"""Reconstruct GLUE tables and convergence figures from one campaign's committed artifacts."""
import hashlib
import csv
import json
import math
import os
from statistics import mean, stdev

from glue.glue_tasks import TASKS
from glue.machine import ROOT
from glue import machine
from glue.catalog import MAIN, EXTRA, Job, jobs, locate, DONE

PENDING="尚未完成"
NO_REFERENCE="无论文参照"


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write(path,columns,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def score(rows,labels):
    nclasses=len(labels)
    confusion=[[0]*(nclasses+1) for _ in labels]  # last column = tied maximum
    truth={}
    for row in rows:
        values=[float(row[name]) for name in labels]
        assert all(map(math.isfinite,values))
        idx,gold=int(row["idx"]),int(row["gold"])
        assert 0<=gold<nclasses and idx not in truth
        truth[idx]=gold
        top=max(values)
        pred=values.index(top) if values.count(top)==1 else -1
        assert pred==int(row["pred"])
        confusion[gold][pred]+=1
    n=len(rows)
    correct=sum(confusion[i][i] for i in range(nclasses))
    counts=[sum(r) for r in confusion]+[0]
    predictions=list(map(sum,zip(*confusion)))
    result=dict(count=n,correct=correct,accuracy=100*correct/n,confusion=confusion,
                predicted_0=predictions[0],ties=predictions[-1])
    if nclasses==2:
        result["recall_0"]=100*confusion[0][0]/counts[0] if counts[0] else None
        denominator=counts[1]+predictions[1]
        result["f1_positive"]=2*confusion[1][1]/denominator if denominator else 0.
        denom=math.sqrt((n*n-sum(x*x for x in counts))*(n*n-sum(x*x for x in predictions)))
        result["mcc"]=(correct*n-sum(x*y for x,y in zip(counts,predictions)))/denom if denom else 0.
    return result,truth


def points(rs,metric,axis):
    coords=[(float(r[axis]),float(r[metric])) for r in rs if r.get(axis) and r.get(metric)]
    return ([x for x,_ in coords],[y for _,y in coords])


def comparison(name,split,observations,reference):
    row=dict(task=name,split=split)
    local={}
    for method in MAIN:
        paper,disp=reference[method]
        vals=observations[method]
        complete=None not in vals
        local[method]=mean(vals) if complete else None
        deviation = NO_REFERENCE
        if isinstance(paper, (int, float)):
            deviation = local[method] - paper if complete else PENDING
        row.update({f"paper_{method}_pct":paper,f"paper_{method}_disp_pp":disp,
                    f"local_{method}_pct":local[method] if complete else PENDING,
                    f"local_{method}_sd_pp":stdev(vals) if complete else PENDING,
                    f"{method}_deviation_pp":deviation})
    deltas=[b-a if a is not None and b is not None else PENDING
            for a,b in zip(observations["lora"],observations["ga"])]
    row.update({f"paired_s{i}_pp":v for i,v in enumerate(deltas)})
    row["paired_mean_pp"]=mean(deltas) if PENDING not in deltas else PENDING
    a,b=[reference[m][0] for m in MAIN]
    row["paper_gain_pp"]=b-a if isinstance(a,(int,float)) and isinstance(b,(int,float)) else NO_REFERENCE
    return row


def report_tables(out,values,run_rows,equal):
    paper=json.loads((ROOT/"paper_reference.json").read_text(encoding="utf8"))
    table=[]
    for i,t in enumerate(paper["task_order"]):
        for split in TASKS[t].validation:
            vals={m:[values.get((Job(t,m,s),split)) for s in range(3)] for m in MAIN}
            ref={m:paper["table1"][m][i] if split!="validation_mismatched" else (NO_REFERENCE,NO_REFERENCE)
                 for m in MAIN}
            table.append(comparison(t,split,vals,ref))
    equal_vals={m:equal[m]["per_seed"] for m in MAIN}
    for source in ("paper_printed_average","recomputed_from_five_tasks"):
        ref={m:(paper["reported_average"][m] if source=="paper_printed_average" else
                mean(v[0] for v in paper["table1"][m]),"未报告") for m in MAIN}
        table.append(comparison(source,"five_tasks_mnli_matched",equal_vals,ref))
    write(out/"table1_comparison.csv",table[0],table)
    ablations=[]
    for m,(paper_mean,disp) in paper["table9_mrpc"].items():
        rows=[r for r in run_rows if r["task"]=="mrpc" and r["method"]==m]
        obs=[r["accuracy"] for r in rows if r["status"]==DONE]
        row=dict(method=m,paper_pct=paper_mean,paper_disp_pp=disp,completed_seeds=len(obs))
        for r in rows:
            for field in ("accuracy","confusion","predicted_0","recall_0","predictions"):
                row[f"s{r['seed']}_{field}"]=r.get(field,PENDING)
        row.update(local_mean_pct=mean(obs) if len(obs)==3 else PENDING,
                   local_sample_sd_pp=stdev(obs) if len(obs)==3 else PENDING,
                   deviation_pp=mean(obs)-paper_mean if len(obs)==3 else PENDING)
        ablations.append(row)
    write(out/"table9_mrpc.csv",ablations[0],ablations)


def paired_cases(root,rows):
    sources={(r["task"],r["split"],r["seed"],r["method"]):r for r in rows if r["status"]==DONE}
    for (task,split,seed,method),left in sources.items():
        right=sources.get((task,split,seed,"ga"))
        if method!="lora" or right is None:
            continue
        files=[r["predictions"] for r in (left,right)]
        maps=[{int(r["idx"]):r for r in read(root/f)} for f in files]
        if maps[0].keys()!=maps[1].keys():
            raise ValueError(f"unpaired IDs: {files}")
        for idx in sorted(maps[0]):
            a,b=[lookup[idx] for lookup in maps]
            if int(a["gold"])!=int(b["gold"]):
                raise ValueError(f"inconsistent gold label: {files} / {idx}")
            gold=int(a["gold"])
            correct=(int(a["pred"])==gold,int(b["pred"])==gold)
            category={(True,True):"两者都对",(False,True):"GA 纠错",(True,False):"GA 新增错误",(False,False):"两者都错"}
            record=dict(task=task,split=split,seed=seed,idx=idx,gold=gold,category=category[correct],step=left["step"])
            for m,r,source in zip(MAIN,(a,b),(left,right)):
                record.update({f"{m}_pred":int(r["pred"]),f"{m}_logits":json.dumps({k:float(r[k]) for k in TASKS[task].labels}),
                               f"{m}_attempt":source["attempt"],f"{m}_predictions":source["predictions"]})
            yield record


def summarize(out,root=None):
    root=machine.current.formal if root is None else root
    out.mkdir(parents=True,exist_ok=True)
    run_rows=[]
    completed={}
    values={}
    orders={}
    split_gold={}
    for j in jobs():
        status,folder=locate(j,root)
        steps=math.ceil(TASKS[j.task].train/32)
        for split,size in TASKS[j.task].validation.items():
            row=dict(task=j.task,method=j.method,seed=j.seed,split=split,status=status)
            if status==DONE:
                completed[j]=folder
                ids=json.loads((folder/"order.json").read_text(encoding="utf8"))
                order=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode("ascii")).digest()
                assert orders.setdefault((j.task,j.seed),order)==order, "unpaired sample order"
                metrics,gold=score(read(folder/f"pred-{split}-{steps}.csv"),TASKS[j.task].labels)
                assert len(gold)==size and split_gold.setdefault((j.task,split),gold)==gold
                values[j,split]=metrics["accuracy"]
                row.update(metrics)
                folder_name=folder.relative_to(root).as_posix()
                row.update(attempt=folder_name,step=steps,checkpoint=f"{folder_name}/final.pt",
                           predictions=f"{folder_name}/pred-{split}-{steps}.csv")
                row["confusion"]=json.dumps(row["confusion"])
            run_rows.append(row)
    cols=["task","method","seed","split","status","count","correct","accuracy","predicted_0","ties",
          "recall_0","f1_positive","mcc","confusion","attempt","step","checkpoint","predictions"]
    write(out/"per_seed.csv",cols,run_rows)
    pairs=[]
    means=[]
    for t,task in TASKS.items():
        for split in task.validation:
            for seed in range(3):
                a,b=[values.get((Job(t,m,seed),split)) for m in MAIN]
                pairs.append(dict(task=t,split=split,seed=seed,lora=a,ga=b,
                                  delta_pp=b-a if a is not None and b is not None else PENDING))
            for m in MAIN+(EXTRA if t=="mrpc" else ()):
                obs=[values[Job(t,m,s),split] for s in range(3) if (Job(t,m,s),split) in values]
                means.append(dict(task=t,split=split,method=m,n=len(obs),mean=mean(obs) if len(obs)==3 else PENDING,
                                  sample_sd=stdev(obs) if len(obs)==3 else PENDING))
    write(out/"paired.csv",["task","split","seed","lora","ga","delta_pp"],pairs)
    equal={}
    for m in MAIN:
        averages=[]
        for s in range(3):
            keys=[(Job(t,m,s),next(iter(task.validation))) for t,task in TASKS.items()]
            averages.append(mean(values[k] for k in keys) if all(k in values for k in keys) else None)
        equal[m]=dict(per_seed=averages,mean=mean(averages) if None not in averages else PENDING,
                      sample_sd=stdev(averages) if None not in averages else PENDING)
    write(out/"task_summary.csv",["task","split","method","n","mean","sample_sd"],means)
    (out/"five_task_average.json").write_text(json.dumps(equal,ensure_ascii=False,indent=2),encoding="utf8")
    report_tables(out,values,run_rows,equal)
    case_cols=["task","split","seed","idx","gold","category","step"]
    case_cols.extend(f"{m}_{field}" for m in MAIN for field in ("pred","logits","attempt","predictions"))
    write(out/"paired_cases.csv",case_cols,paired_cases(root,run_rows))
    return completed


def plot(out,completed):
    os.environ.setdefault("MPLCONFIGDIR",str(machine.current.cache_dir/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=dict(zip(MAIN+EXTRA,["#315d8a","#bb5233","#3e794e","#8765a2","#b18a25"]))
    for task,spec in TASKS.items():
        runs=[(j,d) for j,d in completed.items() if j.task==task]
        if not runs:
            continue
        for split in spec.validation:
            fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
            for j,d in runs:
                logs=read(d/"loss.csv")
                evaluations=[r for r in read(d/"evaluation.csv") if r["split"]==split]
                for col,axis in enumerate(("step","wall_s")):
                    for ri,(data,metric) in enumerate(((logs,"loss"),(evaluations,"accuracy"))):
                        ax=axes[ri,col]
                        x,y=points(data,metric,axis)
                        if x:
                            ax.plot(x,y,color=colors[j.method],linewidth=1,alpha=.7,
                                    linestyle=("-","--",":")[j.seed],marker="." if ri else None,
                                    label=f"{j.method} s{j.seed}")
                        ax.set(xlabel="Optimizer update" if col==0 else "Elapsed wall time (s)",
                               ylabel="Token cross-entropy" if ri==0 else "Validation accuracy (0–1)")
            fig.suptitle(f"T5-base · {task.upper()} · {split}")
            axes[0,0].legend(fontsize=7,ncol=2)
            fig.savefig(out/f"{task}-{split}.png",dpi=160)
            plt.close(fig)
