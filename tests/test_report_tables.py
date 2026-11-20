"""Generate sealed synthetic MRPC attempts only under disposable test directories."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from glue.analyze_glue import PENDING, comparison, read, summarize
from glue.catalog import Job, eval_steps, finish
from glue import machine


def write(path,fields,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root,method="lora",seed=0):
    if root.resolve().is_relative_to(machine.current.runs_dir.resolve()):
        raise ValueError("fixtures require an isolated directory")
    j=Job("mrpc",method,seed)
    folder=root/j.key/"attempt-0001"
    folder.mkdir(parents=True)
    steps=eval_steps(115,115)
    # Sealed attempts from the earlier schema include the unused calibration_sampling_seed field.
    for name,data in dict(config=dict(key=j.key,steps=115,eval_steps=steps,synthetic=True,calibration_sampling_seed=0),
                          order=list(range(3668)),init=dict(synthetic=True),resources=dict(synthetic=True)).items():
        # Identical orders can be serialized on different operating systems.
        body=json.dumps(data,indent=2).replace("\n","\r\n") if name=="order" and method=="ga" else json.dumps(data)
        (folder/f"{name}.json").write_bytes(body.encode("utf8"))
    for name in ("initial","final"):
        torch.save({"synthetic":torch.tensor([0.])},folder/f"{name}.pt")
    write(folder/"loss.csv",["step","seen","loss"],[dict(step=i,seen=min(32*i,3668),loss=1) for i in range(1,116)])
    rows=[]
    for idx in range(408):
        gold=idx%2
        correct=idx%8 in ((0,1,2,3) if method=="lora" else (0,1,4,5,6))
        pred=gold if correct else 1-gold
        rows.append(dict(idx=idx,gold=gold,pred=pred,different=1-pred,equivalent=pred))
    if method=="ga":
        rows.reverse()
    acc=sum(r["gold"]==r["pred"] for r in rows)/408
    write(folder/"evaluation.csv",["step","split","accuracy"],[dict(step=k,split="validation",accuracy=acc) for k in steps])
    for k in steps:
        write(folder/f"pred-validation-{k}.csv",rows[0],rows)
    finish(folder,j,115)
    return folder


class ReportTests(unittest.TestCase):
    def test_paired_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"campaign"
            out=Path(tmp)/"out"
            for seed in range(3):
                for m in ("lora","ga"):
                    fixture(root,m,seed)
            summarize(out,root)
            table=read(out/"table1_comparison.csv")
            mrpc=next(r for r in table if r["task"]=="mrpc")
            self.assertEqual(float(mrpc["paired_mean_pp"]),12.5)
            self.assertEqual(float(mrpc["local_lora_sd_pp"]),0)
            self.assertAlmostEqual(float(mrpc["ga_deviation_pp"]),62.5-85.29)
            self.assertAlmostEqual(float(table[-2]["paper_lora_pct"]),82.08)
            self.assertAlmostEqual(float(table[-1]["paper_lora_pct"]),82.006)
            self.assertEqual(table[-1]["local_ga_pct"],PENDING)
            ablations=read(out/"table9_mrpc.csv")
            self.assertEqual(len(ablations),5)
            ga=next(r for r in ablations if r["method"]=="ga")
            self.assertEqual(float(ga["s0_accuracy"]),62.5)
            self.assertEqual(int(ga["s0_predicted_0"]),sum(r[0] for r in json.loads(ga["s0_confusion"])))
            self.assertEqual(ablations[1]["local_mean_pct"],PENDING)
            cases=read(out/"paired_cases.csv")
            self.assertEqual(len(cases),408*3)
            self.assertEqual({r["category"] for r in cases},{"两者都对","两者都错","GA 纠错","GA 新增错误"})
            fixed=next(r for r in cases if r["idx"]=="4")
            self.assertEqual((fixed["gold"],fixed["lora_pred"],fixed["ga_pred"]),("0","1","0"))
            self.assertEqual(fixed["category"],"GA 纠错")
            self.assertTrue((root/fixed["ga_predictions"]).is_file())
            self.assertEqual(json.loads(fixed["ga_logits"]),{"different":1,"equivalent":0})

    def test_mismatched_deviation_has_no_paper_reference(self):
        scores = {"lora": [85., 86., 87.], "ga": [86., 87., 88.]}
        reference = {method: ("无论文参照", "无论文参照") for method in scores}
        row = comparison("mnli", "validation_mismatched", scores, reference)
        self.assertEqual(row["lora_deviation_pp"], "无论文参照")
        self.assertEqual(row["ga_deviation_pp"], "无论文参照")
        self.assertEqual(row["paired_mean_pp"], 1.)

    def test_sample_sd_and_missing_seed(self):
        r=comparison("mrpc","validation",{"ga":[0.,1.,2.],"lora":[0.,0.,None]},
                     {"ga":[85.29,.24],"lora":[68.38,.01]})
        self.assertEqual(r["lora_deviation_pp"], PENDING)
        self.assertEqual((r["local_ga_sd_pp"],r["paired_s0_pp"],r["paired_s2_pp"],r["paired_mean_pp"]),
                         (1.,0.,PENDING,PENDING))
