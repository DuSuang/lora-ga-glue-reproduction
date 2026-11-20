"""PyTorch implementation of the two initializations in upstream reproduce/run_exp.py."""
import math
from time import perf_counter

import torch
from torch import nn

RANK = 8
ALPHA = 16
GAMMA = 16
EXPECTED_PARAMETERS = 144 * (768 + 768) * RANK + 48 * (768 + 3072) * RANK


METHODS = ("lora", "gaussian", "so", "ga_only", "ga")
CALIBRATION_MEAN = "arithmetic mean of singleton sequence-mean loss gradients"


def scale(method):
    if method not in METHODS:
        raise ValueError(method)
    return ALPHA / (math.sqrt(RANK) if method in ("so","ga") else RANK)


def resolved_init(method):
    gaussian = method in ("gaussian","so")
    svd = method in ("ga_only","ga")
    return dict(method=method, rank=RANK, alpha=ALPHA, scaling=scale(method),
                factor_type="gradient_svd" if svd else "normal" if gaussian else "kaiming_uniform/zero",
                normal_std_A="1/sqrt(d_out)" if gaussian else None,
                normal_std_B="1/sqrt(d_in)" if gaussian else None,
                direction="A=V[:,:8].T; B=U[:,8:16]" if svd else None,
                multiplier="d_out**0.25/sqrt(gamma)" if method in ("so","ga") else "1",
                gamma=GAMMA if method in ("so","ga") else None, compensation="base -= scaling * B @ A",
                calibration_count=8 if svd else 0, gradient_aggregation=CALIBRATION_MEAN if svd else None,
                svd_q=32 if svd else None, svd_niter=4 if svd else None)


def targets(model):
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "lm_head"}
    assert len(layers) == 192 and {n.split(".")[-1] for n in layers} == {"q", "k", "v", "o", "wi", "wo"}
    return layers


def begin(device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return perf_counter()


def measure(start,device):
    torch.cuda.synchronize(device)
    return {"seconds": perf_counter() - start, "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20}


class Adapter(nn.Module):
    def __init__(self, linear, factor):
        super().__init__()
        self.base = linear.requires_grad_(False)
        self.factor = factor
        self.A = nn.Parameter(linear.weight.new_empty(RANK, linear.in_features))
        self.B = nn.Parameter(linear.weight.new_zeros(linear.out_features, RANK))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        # T5DenseActDense checks the output projection dtype before calling it.
        return self.base.weight

    def forward(self, x):
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.A), self.B) * self.factor

    @torch.no_grad()
    def compensate(self):
        self.base.weight.sub_((self.B @ self.A) * self.factor)


def estimate(model, layers, corpus, *, dropout=True, indices=range(8)):
    # Gradients w.r.t. these original weights are identical to their partials in full FT.
    model.requires_grad_(False).train(dropout)
    for layer in layers.values():
        layer.weight.requires_grad_(True)
    grads = {}
    for i in indices:
        batch = corpus.batch([i], initializing=True)
        model(**batch).loss.backward()
        for name, layer in layers.items():
            g = layer.weight.grad.detach().cpu()
            if name in grads:
                grads[name].add_(g)
            else:
                grads[name] = g
            layer.weight.grad = None
    model.requires_grad_(False)
    return {name:g.div_(len(indices)) for name,g in grads.items()}


@torch.no_grad()
def initialize_factors(module, method, gradient=None, svd_seed=None):
    """Table 3 uses normal variances, and V returned by svd_lowrank stores columns."""
    if method == "lora":
        return
    m,n = module.weight.shape
    if method in ("gaussian","so"):
        module.A.normal_(std=m**-0.5)
        module.B.normal_(std=n**-0.5)
    elif method in ("ga_only","ga"):
        g = gradient.to(module.weight)
        def decompose():
            return torch.svd_lowrank(g, q=min(4*RANK,m,n), niter=4)
        if svd_seed is None:
            u,_,v = decompose()
        else:
            devices = [g.device.index] if g.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                generator = torch.cuda.default_generators[g.device.index] if g.is_cuda else torch.default_generator
                generator.manual_seed(svd_seed)
                u,_,v = decompose()
        module.A.copy_(v[:,:RANK].T)
        module.B.copy_(u[:,RANK:2*RANK])
    else:
        raise ValueError(method)
    if method in ("so","ga"):
        amplitude = m**0.25 / math.sqrt(GAMMA)
        module.A.mul_(amplitude)
        module.B.mul_(amplitude)


def install(model, corpus, method, stats, *, svd_seed=None, calibration_dropout=True):
    layers = targets(model)
    model.requires_grad_(False)
    use_gradient = method in ("ga","ga_only")
    if use_gradient:
        start = begin(corpus.device)
        grads = estimate(model, layers, corpus, dropout=calibration_dropout)
        stats["gradient"] = measure(start,corpus.device)
    start = begin(corpus.device)
    adapters, errors, norms = {}, {}, {}
    with torch.no_grad():
        for layer_index, (name, layer) in enumerate(layers.items()):
            module = Adapter(layer, scale(method))
            layer_seed = None if svd_seed is None else svd_seed + layer_index
            initialize_factors(module,method,grads.pop(name) if use_gradient else None,layer_seed)
            if use_gradient and layer_seed is not None:
                stats.setdefault("svd_layer_seeds",{})[name] = layer_seed
            original = layer.weight.clone()
            module.compensate()
            recovered = layer.weight + (module.B @ module.A) * module.factor
            errors[name] = ((recovered - original).norm() / original.norm()).item()
            norms[name] = [module.A.norm().item(), module.B.norm().item()]
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, module)
            adapters[name] = module
    stats["svd_and_install" if use_gradient else "install"] = measure(start,corpus.device)
    stats["weight_relative_errors"] = errors
    stats["factor_norms"] = norms
    assert all(a > 0 and ((b > 0) == (method != "lora")) for a, b in norms.values())
    assert max(errors.values()) < 2e-6
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == EXPECTED_PARAMETERS, n
    stats["trainable"] = n
    return adapters


def snapshot(modules):
    return {name: {k: getattr(m, k).detach().cpu() for k in ("A", "B")} for name, m in modules.items()}


@torch.no_grad()
def restore(model, method, initial, final):
    model.requires_grad_(False)
    layers = targets(model)
    assert set(layers) == set(initial) == set(final)
    for name, layer in layers.items():
        wrapper = Adapter(layer, scale(method))
        wrapper.A.copy_(initial[name]["A"])
        wrapper.B.copy_(initial[name]["B"])
        wrapper.compensate()
        wrapper.A.copy_(final[name]["A"])
        wrapper.B.copy_(final[name]["B"])
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, wrapper)
    return model.eval()
