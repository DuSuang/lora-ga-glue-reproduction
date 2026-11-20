"""Prepare, run and verify the T5-base glue-v1 campaign."""
import argparse
from collections import Counter
import math
from pathlib import Path

from glue import machine
from glue.catalog import DONE, PARTIAL, NEW, MAIN, EXTRA, finish, jobs, locate
from glue.glue_tasks import TASKS
from glue.system_io import file_lock


def run(job,args):
    limit=min(3,math.ceil(TASKS[job.task].train/32)) if args.smoke else math.ceil(TASKS[job.task].train/32)
    root=machine.current.runs_dir/"glue-smoke-v1" if args.smoke else machine.current.formal
    parent=root/job.key
    with file_lock(parent/".running.lock"):
        state,old=locate(job,root,limit=limit)
        latest=max((int(d.name[8:]) for d in parent.glob("attempt-*") if d.is_dir() and d.name[8:].isdigit()),default=0)
        newer_partial=state==DONE and latest>int(old.name[8:])
        if args.retry and state!=PARTIAL and not newer_partial:
            raise ValueError("--retry requires an incomplete latest attempt")
        if args.rerun_complete and state!=DONE:
            raise ValueError("--rerun-complete requires a verified complete run")
        if state!=NEW and not (args.retry or args.rerun_complete):
            print(job.key,state,old,"skipped",flush=True)
            return
        from glue.glue_train import execute
        attempt=parent/f"attempt-{latest+1:04d}"
        attempt.mkdir()
        execute(job,attempt,limit,smoke=args.smoke,microbatch=args.microbatch,eval_batch=args.eval_batch)
        finish(attempt,job,limit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("prepare","audit","list","run","diagnose","analyze","package","test"):
        p=commands.add_parser(name)
        machine.options(p)
        if name in ("list","run"):
            p.add_argument("--scope",choices=["all","main","ablation"],default="all")
            p.add_argument("--task",choices=TASKS)
            p.add_argument("--method",choices=MAIN+EXTRA)
            p.add_argument("--seed",type=int,choices=range(3))
            p.add_argument("--smoke",action="store_true")
        if name=="run":
            repeats=p.add_mutually_exclusive_group()
            repeats.add_argument("--retry",action="store_true")
            repeats.add_argument("--rerun-complete",action="store_true")
        elif name=="prepare":
            p.add_argument("--cpu-only",action="store_true",help="Prepare tokenizer and Parquet only")
        elif name=="diagnose":
            p.add_argument("--seed",type=int,choices=range(3),default=0)
            p.add_argument("--out",type=Path,help="New diagnostic directory")
        elif name=="analyze":
            p.add_argument("--plot",action="store_true")
        elif name=="package":
            p.add_argument("action",choices=["check","build"])
            p.add_argument("--output",type=Path,help="New archive path for build")
            p.add_argument("--report",type=Path,help="Include this finished PDF, DOCX or Markdown report")
        elif name=="test":
            p.add_argument("tests",nargs="*",help="Optional unittest names")
    args=parser.parse_args()
    try:
        site=machine.configure(args)
    except (OSError,TypeError,ValueError) as exc:
        parser.error(str(exc))
    if args.command in ("list","run"):
        chosen=[j for j in jobs(args.scope) if (not args.task or j.task==args.task)
                and (not args.method or j.method==args.method) and (args.seed is None or j.seed==args.seed)]
        if not chosen:
            parser.error("no configurations match")
        if args.command=="list":
            counts=Counter()
            for j in chosen:
                state,path=locate(j,site.runs_dir/"glue-smoke-v1",limit=3) if args.smoke else locate(j,site.formal)
                counts[state]+=1
                print(j.key,state,path or "")
            print(len(chosen),"configs",dict(counts))
            return
        if (args.retry or args.rerun_complete) and (not args.task or not args.method or args.seed is None):
            parser.error("retry and rerun require --task TASK --method METHOD --seed SEED")
        machine.setup_cache()
        args.microbatch,args.eval_batch=site.microbatch,site.eval_batch
        for j in chosen:
            run(j,args)
    elif args.command=="prepare":
        from glue.prepare import prepare_assets
        prepare_assets(TASKS,weights=not args.cpu_only)
    elif args.command=="audit":
        from glue.audit_glue_data import write_audit
        if not write_audit(site.data_dir,site.analysis_dir/"data_audit.json"):
            raise SystemExit(1)
    elif args.command=="diagnose":
        machine.setup_cache()
        from glue.first_step import run as diagnose
        diagnose(args.seed,args.out or site.runs_dir/"diagnostics"/f"mrpc-s{args.seed}",site.device)
    elif args.command=="analyze":
        from glue.analyze_glue import summarize,plot
        completed=summarize(site.analysis_dir,site.formal)
        if args.plot:
            machine.setup_cache()
            plot(site.analysis_dir,completed)
        print(f"{len(completed)}/39 complete; tables: {site.analysis_dir}")
    elif args.command=="test":
        machine.setup_cache()
        from glue.cpu_check import run as test
        if not test(args.tests):
            raise SystemExit(1)
    elif args.command=="package":
        if (args.action=="build")!=(args.output is not None):
            parser.error("use package check, or package build --output NEW.zip")
        machine.setup_cache()
        from glue.package_glue import package
        package(args.output,args.report)


if __name__=="__main__":
    main()
