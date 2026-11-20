"""Dropout-free MRPC first-order SGD diagnostic; no optimizer or epoch training."""

import torch

from glue.adapters import CALIBRATION_MEAN, install, targets
from glue import machine
from glue.model_runtime import base, probe, seed_all, select_device
from glue.glue_data import GlueCorpus
from glue.system_io import save_json


LAYERS=("encoder.block.0.layer.0.SelfAttention.q",
        "encoder.block.0.layer.1.DenseReluDense.wi","encoder.block.0.layer.1.DenseReluDense.wo",
        "decoder.block.0.layer.1.EncDecAttention.v")


def direction(g,a,b,s):
    # Unit learning rate derivative d(W'+sBA)/dt under simultaneous Euclidean gradient flow.
    return -(s*s)*((g@a.T)@a+b@(b.T@g))


def cosine(a,b):
    a,b=a.flatten().double(),b.flatten().double()
    den=a.norm()*b.norm()
    return (torch.dot(a,b)/den).item() if den.item()!=0 else None


def gradients(model,corpus,indices,parameters,calibration):
    model.eval()
    accum=[torch.zeros_like(p,device="cpu") for p in parameters]
    for i in indices:
        batch=corpus.batch([i],initializing=calibration)
        loss=model(**batch).loss/len(indices)
        for dst,g in zip(accum,torch.autograd.grad(loss,parameters)):
            dst.add_(g.detach().cpu())
    return accum


def run(seed,out,device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model diagnostic requires CUDA")
    from glue.prepare import WEIGHT_HASH,digest
    assert digest(machine.current.model_dir/"model.safetensors")==WEIGHT_HASH
    device=select_device(device or machine.current.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True,exist_ok=False)
    c=GlueCorpus("mrpc",device)
    groups={"calibration":list(range(8)),"held_out":list(range(8,16))}
    ids={k:[c.rows["train"][i]["idx"] for i in indices] for k,indices in groups.items()}
    assert set(ids["calibration"]).isdisjoint(ids["held_out"])
    model=base(machine.current.model_dir,device).requires_grad_(False)
    layers=targets(model)
    weights=[layers[n].weight.requires_grad_(True) for n in LAYERS]
    refs={k:gradients(model,c,ii,weights,k=="calibration") for k,ii in groups.items()}
    fixed=c.batch([0,1,2,3],initializing=True)
    before=probe(model,fixed)
    del model,layers,weights
    report=dict(device=str(device),gpu=torch.cuda.get_device_name(device),seed=seed,group_ids=ids,dropout=False,gradient_aggregation=CALIBRATION_MEAN,
                reference="full-model task cross-entropy; unit SGD learning rate; no clipping or AdamW",
                calibration=c.calibration(),metrics=[])
    for method in ("lora","ga"):
        seed_all(seed,device)
        model=base(machine.current.model_dir,device)
        stats={}
        modules=install(model,c,method,stats,svd_seed=seed,calibration_dropout=False)
        after=probe(model,fixed)
        torch.testing.assert_close(after,before,atol=2e-4,rtol=1e-5)
        chosen=[modules[n] for n in LAYERS]
        factors=[p for m in chosen for p in (m.A,m.B)]
        for group,indices in groups.items():
            actual=gradients(model,c,indices,factors,group=="calibration")
            for i,(name,module,g) in enumerate(zip(LAYERS,chosen,refs[group])):
                a,b=module.A.detach().cpu(),module.B.detach().cpu()
                expected=direction(g,a,b,module.factor)
                da,db=actual[2*i:2*i+2]
                observed=-module.factor*(db@a+b@da)
                report["metrics"].append(dict(method=method,group=group,layer=name,scale=module.factor,
                    full_gradient_norm=g.norm().item(),direction_norm=expected.norm().item(),
                    cosine_with_negative_full_gradient=cosine(expected,-g),
                    autograd_sgd_cosine=cosine(observed,-g),init_logits_max_abs=(after-before).abs().max().item()))
        del model,modules,chosen,factors,module
    save_json(out/"diagnostic.json",report)
