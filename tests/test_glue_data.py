import unittest
import ast
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from glue.glue_tasks import TASKS
from glue.glue_data import collate, predict, evaluate_split


class DataTests(unittest.TestCase):
    def test_contract_against_upstream_loader(self):
        source=Path(__file__).resolve().parents[1]/"reference"/"data.py"
        tree=ast.parse(source.read_text(encoding="utf8"))
        for name,task in TASKS.items():
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_"+name)
            fn.decorator_list=[]
            class FakeDataset(dict):
                def map(self,transform):
                    for label,text in enumerate(task.labels):
                        row={field:"field-"+str(i) for i,field in enumerate(task.fields)}
                        row["label"]=label
                        self_test.assertEqual(transform(row),{"x":task.prompt(row),"y":text})
                    return self
            self_test=self
            fake=FakeDataset((s,s) for s in ("train","test",*task.validation))
            scope={"load_dataset":lambda *_:fake}
            exec(compile(ast.Module([fn],[]),str(source),"exec"),scope)
            self.assertEqual(scope[fn.name]()[1],next(iter(task.validation)))

    def test_templates_and_tokens(self):
        from tokenizers import Tokenizer
        from glue import machine
        if not (machine.current.model_dir/"tokenizer.json").exists():
            self.skipTest("local T5 tokenizer unavailable")
        tokenizer=Tokenizer.from_file(str(machine.current.model_dir/"tokenizer.json"))
        for task in TASKS.values():
            self.assertEqual(tuple(tuple(tokenizer.encode(w).ids) for w in task.labels),task.tokens)
            self.assertEqual(task.prompt(dict(zip(task.fields,task.fields))),
                             task.instruction+": "+"\n".join(task.fields)+"\nresult: ")
        self.assertEqual(TASKS["mnli"].fields,("premise","hypothesis"))
        self.assertEqual(TASKS["qnli"].fields,("question","sentence"))

    def test_variable_labels(self):
        b=collate([[11],[20,1]],[[3,35,1],[7163,1]],0,"cpu")
        self.assertEqual(b["labels"].tolist(),[[3,35,1],[7163,1,0]])
        self.assertEqual(b["decoder_attention_mask"].sum().item(),5)
        self.assertEqual(predict([1.,3.,3.]),-1)

    def test_three_way_first_position(self):
        class Dummy(torch.nn.Module):
            def forward(self,**kw):
                return SimpleNamespace(logits=torch.tensor([[[0.,1.,5.],[8.,0.,0.]]]))
        corpus=SimpleNamespace(rows={"validation_matched":[{"idx":9,"label":2}]},
                               task=TASKS["mnli"],candidates=[0,1,2],batch=lambda i,s:{})
        with TemporaryDirectory() as d:
            self.assertEqual(evaluate_split(Dummy(),corpus,"validation_matched",Path(d)/"p.csv"),1)
        self.assertFalse(torch.cuda.is_initialized())
