"""Shared single-device training loop for the five GLUE tasks and MRPC ablations."""
import csv
import gc
import math
import hashlib
import subprocess
import platform
from importlib.metadata import version
from time import perf_counter

import torch
from transformers import get_cosine_schedule_with_warmup

from glue.adapters import install, restore, snapshot, resolved_init
from glue import machine
from glue.model_runtime import base, probe, seed_all, select_device
from glue.glue_data import GlueCorpus, evaluate_split
from glue.glue_tasks import TASKS
from glue.system_io import save_json
from glue.prepare import ROOT, DATA_REV, MODEL_REV, WEIGHT_HASH, digest
from glue.catalog import eval_steps


def clock(device):
    if torch.device(device).type=="cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def order_for(n,seed):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n,generator=generator).tolist()


def source_version():
    # The normalized code digest also identifies source from extracted ZIPs without .git.
    digest=hashlib.sha256()
    for file in [ROOT/"suite.py",*sorted((ROOT/"glue").glob("*.py"))]:
        digest.update(file.relative_to(ROOT).as_posix().encode()+b"\0")
        digest.update(file.read_text(encoding="utf8").encode()+b"\0")
    source={"sha256":digest.hexdigest(),"commit":None,"dirty":None}
    if not (ROOT/".git").exists():
        return source
    try:
        revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],
                                         stderr=subprocess.DEVNULL,text=True).strip()
        changes=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],
                                        stderr=subprocess.DEVNULL,text=True)
        source.update(commit=revision,dirty=bool(changes.strip()))
    except (OSError,subprocess.CalledProcessError):
        pass
    return source


def backward_group(model,batch,micro):
    """Slice an already padded logical batch; T5's CE includes target pad IDs."""
    labels=batch["labels"]
    loss_sum=labels.new_zeros((),dtype=next(model.parameters()).dtype)
    for start in range(0,len(labels),micro):
        part={k:(v[start:start+micro] if torch.is_tensor(v) else v) for k,v in batch.items()}
        weighted=model(**part).loss*(part["labels"].numel()/labels.numel())
        weighted.backward()
        loss_sum+=weighted.detach()
    return loss_sum


def train_epoch(model,corpus,order,config,out,started,init_seconds):
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config["lr"],betas=(0.9,0.999),eps=1e-8,weight_decay=0)
    scheduler=get_cosine_schedule_with_warmup(optimizer,config["warmup_steps"],config["epoch_steps"])
    train_s=valid_s=0.
    device=corpus.device
    group_size=config["effective_batch"]
    micro=config["microbatch"]
    peak_train=peak_eval=0
    cuda=torch.device(device).type=="cuda"
    model.train()
    with (out/"loss.csv").open("x",newline="",encoding="utf8") as lossfile, (out/"evaluation.csv").open(
            "x",newline="",encoding="utf8") as evalfile:
        losses,evals=csv.writer(lossfile),csv.writer(evalfile)
        losses.writerow(["step","samples","seen","loss","lr","preclip_grad_norm","init_s","train_s","validation_s","wall_s"])
        evals.writerow(["step","seen","split","count","accuracy","elapsed_s","init_s","train_s","validation_s","wall_s"])
        for step in range(1,config["steps"]+1):
            start=clock(device)
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            group=order[(step-1)*group_size:step*group_size]
            seen=min(step*group_size,len(order))
            total_loss=backward_group(model,corpus.batch(group),micro)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            now=clock(device)
            train_s+=now-start
            if cuda:
                peak_train=max(peak_train,torch.cuda.max_memory_allocated(device))
            losses.writerow([step,len(group),seen,total_loss.item(),lr,norm.item(),init_seconds,train_s,valid_s,now-started])
            lossfile.flush()
            if step in config["eval_steps"]:
                for split in corpus.task.validation:
                    start=clock(device)
                    if cuda:
                        torch.cuda.reset_peak_memory_stats(device)
                    accuracy=evaluate_split(model,corpus,split,out/f"pred-{split}-{step}.csv",config["eval_batch"])
                    now=clock(device)
                    elapsed=now-start
                    valid_s+=elapsed
                    if cuda:
                        peak_eval=max(peak_eval,torch.cuda.max_memory_allocated(device))
                    evals.writerow([step,seen,split,len(corpus.rows[split]),accuracy,elapsed,init_seconds,
                                    train_s,valid_s,now-started])
                    evalfile.flush()
                model.train()
    return dict(training_s=train_s,validation_s=valid_s,initialization_s=init_seconds,
                through_final_validation_wall_s=clock(device)-started,
                peak_training_MiB=peak_train/2**20,peak_validation_MiB=peak_eval/2**20)


def execute(job,out,limit,*,smoke,microbatch=8,eval_batch=8,device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    device=select_device(device or machine.current.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)
    if digest(machine.current.model_dir/"model.safetensors")!=WEIGHT_HASH:
        raise ValueError("T5-base checksum mismatch")
    corpus=GlueCorpus(job.task,device)
    seed_all(job.seed,device)
    model=base(machine.current.model_dir,device)
    task=TASKS[job.task]
    epoch_steps=math.ceil(task.train/32)
    init=resolved_init(job.method)
    order=order_for(task.train,job.seed)
    config=dict(key=job.key,task=job.task,method=job.method,train_seed=job.seed,smoke=smoke,source=source_version(),
                model_revision=MODEL_REV,data_revision=DATA_REV,split_files=corpus.file_hashes,
                author_revision="c4cd5372c75b290924214b348008891f744512ef",
                python_version=platform.python_version(),platform=platform.system(),cuda_build=torch.version.cuda,
                transformers_version=version('transformers'),torch_version=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(device),precision="fp32",
                tf32=False,rank=8,alpha=16,adapter_dropout=0,base_dropout=model.config.dropout_rate,
                target_modules=["q","k","v","o","wi","wo"],effective_batch=32,microbatch=microbatch,
                eval_batch=eval_batch,gradient_accumulation=32//microbatch,max_length=128,init_max_length=1024,
                optimizer="AdamW",lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
                scheduler="cosine",warmup_ratio=.03,warmup_steps=math.ceil(.03*epoch_steps),
                epoch_steps=epoch_steps,steps=limit,eval_steps=eval_steps(epoch_steps,limit),
                labels=task.labels,label_token_ids=corpus.targets,score_position=0,init=init,
                label_padding="tokenizer pad ID to longest target in logical batch",
                training_loss="mean CE over all logical batch target positions including pad",
                decoder_mask="one for label through EOS; zero for right padding",
                calibration_sampler="first-eight" if init["calibration_count"] else None,
                sampler_uses_rng=False,
                calibration_dropout_seed=job.seed if init["calibration_count"] else None,
                calibration=corpus.calibration() if init["calibration_count"] else [],
                svd_seed_base=job.seed if init["calibration_count"] else None)
    save_json(out/"config.json",config)
    save_json(out/"order.json",[corpus.rows["train"][i]["idx"] for i in order])
    fixed=corpus.batch(range(4),next(iter(task.validation)))
    before=probe(model,fixed)
    seed_all(job.seed,device)
    started=clock(corpus.device)
    init_stats={}
    modules=install(model,corpus,job.method,init_stats,svd_seed=job.seed)
    init_seconds=clock(corpus.device)-started
    after=probe(model,fixed)
    init_stats.update(max_abs_logits=(before-after).abs().max().item(),probe_split=next(iter(task.validation)),
                      probe_ids=[r["idx"] for r in corpus.rows[next(iter(task.validation))][:4]],
                      finite=bool(torch.isfinite(after).all() and torch.isfinite(before).all()))
    save_json(out/"init.json",init_stats)
    torch.testing.assert_close(after,before,rtol=1e-5,atol=2e-4)
    torch.save(snapshot(modules),out/"initial.pt")
    seed_all(job.seed,device)
    stats=train_epoch(model,corpus,order,config,out,started,init_seconds)
    torch.save(snapshot(modules),out/"final.pt")
    reference=probe(model,fixed)
    del model,modules
    gc.collect()
    torch.cuda.empty_cache()
    initial,final=[torch.load(out/n,map_location="cpu",weights_only=True) for n in ("initial.pt","final.pt")]
    rebuilt=probe(restore(base(machine.current.model_dir,device),job.method,initial,final),fixed)
    torch.testing.assert_close(rebuilt,reference,rtol=1e-5,atol=2e-4)
    assert torch.equal(reference[:,0,corpus.candidates].argmax(-1),rebuilt[:,0,corpus.candidates].argmax(-1))
    stats["reload_max_abs_logits"]=(rebuilt-reference).abs().max().item()
    stats["wall_s_including_reload"]=clock(corpus.device)-started
    save_json(out/"resources.json",stats)
