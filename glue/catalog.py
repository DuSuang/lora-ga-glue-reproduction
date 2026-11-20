"""Five-task campaign catalog and local runner. The list command is standard-library only."""
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from zipfile import is_zipfile

from glue.glue_tasks import TASKS

from glue import machine
MAIN = ("lora","ga")
EXTRA = ("gaussian","so","ga_only")
DONE, PARTIAL, NEW = "完整完成","产物不完整","尚未运行"


@dataclass(frozen=True)
class Job:
    task: str
    method: str
    seed: int

    @property
    def key(self):
        return f"{self.task}/{self.method}/seed-{self.seed}"


def jobs(scope="all"):
    core = [Job(t,m,s) for t in TASKS for m in MAIN for s in range(3)]
    extras = [Job("mrpc",m,s) for m in EXTRA for s in range(3)]
    return {"main":core,"ablation":extras,"all":core+extras}[scope]


def eval_steps(total, limit):
    return sorted({1,limit,*[math.ceil(total*i/10) for i in range(1,11) if math.ceil(total*i/10)<=limit],
                   *([total-1] if limit==total and total>1 else [])})


def required(job,limit):
    total=math.ceil(TASKS[job.task].train/32)
    names={"config.json","order.json","initial.pt","final.pt","init.json","loss.csv","evaluation.csv","resources.json"}
    names.update(f"pred-{s}-{k}.csv" for s in TASKS[job.task].validation for k in eval_steps(total,limit))
    return names


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def finish(path,job,limit):
    task=TASKS[job.task]
    cfg=json.loads((path/"config.json").read_text(encoding="utf8"))
    if (cfg["key"],cfg["steps"])!=(job.key,limit):
        raise ValueError("Wrong run identity or update budget")
    order=json.loads((path/"order.json").read_text(encoding="utf8"))
    assert len(order)==len(set(order))==task.train
    with (path/"loss.csv").open(encoding="utf8",newline="") as f:
        logs=list(csv.DictReader(f))
    assert [int(r["step"]) for r in logs]==list(range(1,limit+1))
    assert all(int(r["seen"])==min(32*int(r["step"]),task.train) and math.isfinite(float(r["loss"])) for r in logs)
    with (path/"evaluation.csv").open(encoding="utf8",newline="") as f:
        evaluated={(r["split"],int(r["step"])):float(r["accuracy"]) for r in csv.DictReader(f)}
    for split,count in task.validation.items():
        for step in cfg["eval_steps"]:
            with (path/f"pred-{split}-{step}.csv").open(encoding="utf8",newline="") as f:
                rows=list(csv.DictReader(f))
            assert count==len(rows)==len({int(r["idx"]) for r in rows})
            assert evaluated[split,step]==sum(r["pred"]==r["gold"] for r in rows)/count
    assert all(is_zipfile(path/n) for n in ("initial.pt","final.pt"))
    names=required(job,limit)
    manifest={n:checksum(path/n) for n in sorted(names)}
    temp=path/"complete.tmp"
    temp.write_text(json.dumps(dict(key=job.key,steps=limit,sha256=manifest),indent=2),encoding="utf8")
    temp.replace(path/"complete.json")


def locate(job,root=None, *, limit=None):
    root=machine.current.formal if root is None else root
    parent=root/job.key
    attempts=sorted((p for p in parent.glob("attempt-*") if p.is_dir() and p.name[8:].isdigit()),
                    key=lambda p:int(p.name[8:]),reverse=True)
    expected=limit or math.ceil(TASKS[job.task].train/32)
    for d in attempts:
        try:
            proof=json.loads((d/"complete.json").read_text(encoding="utf8"))
            hashes=proof["sha256"]
            if (proof["key"]==job.key and proof["steps"]==expected and set(hashes)==required(job,expected)
                    and all(checksum(d/n)==h for n,h in hashes.items())):
                return DONE,d
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return (PARTIAL,attempts[0]) if attempts else (NEW,None)
