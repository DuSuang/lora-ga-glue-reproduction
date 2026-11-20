"""Replay the pinned Dataset transform, Trainer collator and T5 CE using a random CPU model."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, Seq2SeqTrainingArguments, Trainer, T5Config,
                          T5ForConditionalGeneration, default_data_collator)

from glue.glue_data import collate
from glue.glue_train import backward_group
from glue.machine import ROOT
from glue import machine


def reference_batch(tokenizer, inputs, texts, model, temp):
    path=ROOT/"reference"/"utils.py"
    names={"SeqToSeqEncode","transform_dataset"}
    nodes=[n for n in ast.parse(path.read_text(encoding="utf8")).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={}
    exec(compile(ast.Module(nodes,[]),str(path),"exec"),scope)
    dataset=Dataset.from_dict({"x":inputs,"y":texts})
    dataset=scope["transform_dataset"]("ConditionalGeneration",tokenizer,dataset,128)

    class FixedOrderTrainer(Trainer):
        def _get_train_sampler(self):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    args=Seq2SeqTrainingArguments(output_dir=str(temp),use_cpu=True,report_to=[],remove_unused_columns=False,
                                  per_device_train_batch_size=32,dataloader_pin_memory=False)
    trainer=FixedOrderTrainer(model=model,args=args,train_dataset=dataset)
    assert trainer.data_collator is default_data_collator
    batch=next(iter(trainer.get_train_dataloader()))
    return trainer,batch


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])


def tiny():
    torch.default_generator.manual_seed(2024)
    config=T5Config(vocab_size=7200,d_model=8,d_kv=4,d_ff=12,num_layers=1,num_decoder_layers=1,
                    num_heads=2,dropout_rate=0.,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,
                    use_cache=False)
    return T5ForConditionalGeneration(config).double().eval()


def compare():
    tok=AutoTokenizer.from_pretrained(machine.current.model_dir,local_files_only=True)
    model=tiny()
    results={}
    with tempfile.TemporaryDirectory() as temp:
        for size in (2,32,19):
            # Homogeneous short-label microbatches make per-microbatch padding observably incorrect.
            texts=["neutral"]*(size-1)+["entailment"]
            inputs=["a"]*size
            trainer,author=reference_batch(tok,inputs,texts,model,temp)
            current=collate(tok(inputs).input_ids,tok(texts).input_ids,tok.pad_token_id,"cpu")
            for k in author:
                torch.testing.assert_close(current[k],author[k],check_dtype=False)
            model.zero_grad()
            loss=trainer.compute_loss(model,dict(author))
            loss.backward()
            expected=flat_grad(model)
            model.zero_grad()
            old=dict(current,labels=current["labels"].masked_fill(~current["decoder_attention_mask"],-100))
            old_loss=model(**old).loss
            old_loss.backward()
            old_grad_diff=(flat_grad(model)-expected).abs().max().item()
            errors={}
            for micro in (1,2,4,8,16,32):
                model.zero_grad()
                actual=backward_group(model,current,micro)
                grad=flat_grad(model)
                torch.testing.assert_close(actual,loss.detach(),rtol=1e-11,atol=1e-11)
                torch.testing.assert_close(grad,expected,rtol=1e-10,atol=1e-11)
                errors[micro]=dict(loss=abs(actual.item()-loss.item()),grad=(grad-expected).abs().max().item())
            results[size]=dict(author_labels=author["labels"].tolist(),mask=author["decoder_attention_mask"].tolist(),
                               old_labels=old["labels"].tolist(),old_count=int(old["labels"].ne(-100).sum()),
                               author_count=author["labels"].numel(),author_loss=loss.item(),old_loss=old_loss.item(),
                               old_grad_max_difference=old_grad_diff,micro_errors=errors)
    return results


class LossProtocolTests(unittest.TestCase):
    def test_author_and_accumulation(self):
        for r in compare().values():
            self.assertGreater(r["author_count"],r["old_count"])
            self.assertGreater(r["old_grad_max_difference"],1e-4)


if __name__=="__main__":
    from glue.cpu_check import guarded
    with guarded():
        print(json.dumps(compare(),indent=2))
