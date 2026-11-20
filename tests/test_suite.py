import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import sys

import torch
from glue.glue_data import collate
from glue.glue_train import order_for, train_epoch
from glue.catalog import DONE,NEW,PARTIAL,Job,eval_steps,jobs,locate,required,checksum,finish


class SuiteTests(unittest.TestCase):
    def test_attempt_selection_and_retries(self):
        import suite
        j=Job("mrpc","ga",0)
        args=SimpleNamespace(smoke=False,retry=False,rerun_complete=False,
                             microbatch=2,eval_batch=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"glue-v1"
            old=root/j.key/"attempt-9999"
            old.mkdir(parents=True)
            with patch.object(suite.machine,"current",suite.machine.Settings(runs_dir=root.parent)),patch("glue.glue_train.execute") as execute,patch.object(suite,"finish"):
                suite.run(j,args)
                execute.assert_not_called()
                args.retry=True
                suite.run(j,args)
                newer=old.with_name("attempt-10000")
                execute.assert_called_once_with(j,newer,115,smoke=False,microbatch=2,eval_batch=3)
                self.assertEqual(locate(j,root),(PARTIAL,newer))
                for d in (newer,old):
                    for n in required(j,115):
                        (d/n).write_bytes(b"test")
                    proof=dict(key=j.key,steps=115,sha256={n:checksum(d/n) for n in required(j,115)})
                    (d/"complete.json").write_text(json.dumps(proof))
                self.assertEqual(locate(j,root),(DONE,newer))
                with self.assertRaises(ValueError):
                    suite.run(j,args)
                args.retry=False
                suite.run(j,args)
                self.assertEqual(execute.call_count,1)
                args.rerun_complete=True
                suite.run(j,args)
                self.assertEqual(execute.call_args.args[1].name,"attempt-10001")
                self.assertEqual(locate(j,root),(DONE,newer))

    def test_list_has_no_ml_runtime_import(self):
        script=("import sys,suite; sys.argv=['suite.py','list']; suite.main(); "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        run=subprocess.run([sys.executable,"-X","utf8","-c",script],check=True,capture_output=True,
                           encoding="utf8",cwd=Path(__file__).resolve().parents[1])
        self.assertIn("39 configs",run.stdout)

    def test_catalog_and_corruption(self):
        self.assertEqual((len(jobs("main")),len(jobs("ablation")),len(set(jobs()))),(30,9,39))
        job=Job("mrpc","ga",0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(locate(job,root),(NEW,None))
            d=root/job.key/"attempt-0001"
            d.mkdir(parents=True)
            self.assertEqual(locate(job,root),(PARTIAL,d))
            for n in required(job,115):
                (d/n).write_bytes(b"synthetic fixture")
            proof=dict(key=job.key,steps=115,sha256={n:checksum(d/n) for n in required(job,115)})
            (d/"complete.json").write_text(json.dumps(proof))
            self.assertEqual(locate(job,root),(DONE,d))
            (d/"final.pt").write_bytes(b"damaged")
            self.assertEqual(locate(job,root),(PARTIAL,d))

    def test_synthetic_tail_and_timing(self):
        class Corpus:
            task=SimpleNamespace(labels=("a","b"),validation={"validation":2},train=35)
            device="cpu"
            candidates=[0,1]
            targets=[[0],[1,0]]
            rows={"train":[dict(idx=i,label=i%2) for i in range(35)],
                  "validation":[dict(idx=i,label=i) for i in range(2)]}
            def batch(self,indices,split="train"):
                return collate([[1]]*len(indices),[self.targets[self.rows[split][i]["label"]] for i in indices],0,"cpu")
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor([.2,-.3]))
            def forward(self,labels,**unused):
                logits=self.weight.expand(*labels.shape,2)
                return SimpleNamespace(logits=logits,loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),labels.flatten()))
        from time import perf_counter
        order=order_for(35,0)
        torch.rand(120)
        self.assertEqual(order,order_for(35,0))
        with tempfile.TemporaryDirectory() as tmp:
            job=Job("mrpc","lora",0)
            root=Path(tmp)/"runs"
            p=root/job.key/"attempt-0001"
            p.mkdir(parents=True)
            cfg=dict(key=job.key,lr=.001,warmup_steps=1,epoch_steps=2,steps=2,effective_batch=32,microbatch=8,
                     eval_steps=eval_steps(2,2),eval_batch=2)
            model=Tiny()
            torch.save(model.state_dict(),p/"initial.pt")
            stats=train_epoch(model,Corpus(),order,cfg,p,perf_counter()-1,1)
            torch.save(model.state_dict(),p/"final.pt")
            for n,value in [("config",cfg),("init",{"synthetic":True}),("order",order),("resources",stats)]:
                (p/(n+".json")).write_text(json.dumps(value))
            with (p/"loss.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([int(r["seen"]) for r in rows],[32,35])
            self.assertEqual(int(rows[-1]["samples"]),3)
            self.assertGreater(float(rows[-1]["wall_s"]),1.)
            self.assertTrue((p/"pred-validation-2.csv").exists())
            from glue.glue_tasks import TASKS
            from glue.analyze_glue import summarize,plot,read
            with patch.dict(TASKS,{"mrpc":Corpus.task}):
                finish(p,job,2)
                self.assertEqual(locate(job,root),(DONE,p))
                out=Path(tmp)/"summary"
                completed=summarize(out,root)
                self.assertEqual(completed,{job:p})
                self.assertEqual(sum(r["status"]==DONE for r in read(out/"per_seed.csv")),1)
                plot(out,completed)
                self.assertGreater((out/"mrpc-validation.png").stat().st_size,1000)
        self.assertFalse(torch.cuda.is_initialized())
