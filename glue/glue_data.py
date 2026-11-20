"""Local Parquet ingestion and first-position candidate classification."""
import csv
from array import array

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from glue.glue_tasks import TASKS
from glue import machine
from glue.prepare import digest, locked_files


def collate(sequences, targets, pad, device):
    def padded(data, value):
        width = max(map(len,data))
        return torch.tensor([list(row)+[value]*(width-len(row)) for row in data],device=device)
    inputs = padded(sequences,pad)
    labels = padded(targets,pad)
    widths = torch.tensor([len(row) for row in targets],device=device)
    return dict(input_ids=inputs,attention_mask=inputs.ne(pad),labels=labels,
                decoder_attention_mask=torch.arange(labels.shape[1],device=device)<widths[:,None],use_cache=False)


class GlueCorpus:
    def __init__(self, name, device):
        self.task = task = TASKS[name]
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(machine.current.model_dir, local_files_only=True)
        self.targets = tokenizer(list(task.labels)).input_ids
        if tuple(map(tuple,self.targets)) != task.tokens:
            raise ValueError("Unexpected label tokenization for " + name)
        self.candidates = [t[0] for t in self.targets]
        assert len(self.candidates)==len(set(self.candidates))
        self.pad_id = tokenizer.pad_token_id
        folder = machine.current.data_dir/name
        files = locked_files(name)
        self.encoded, self.rows = {}, {}
        self.init_ids = []
        self.file_hashes = files
        for split,count in {"train":task.train,**task.validation}.items():
            rows, encoded = [], []
            for part in files[split]:
                path = folder/part["name"]
                if digest(path)!=part["sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
                for chunk in pq.ParquetFile(path).iter_batches(batch_size=512,columns=["idx","label",*task.fields]):
                    raw = chunk.to_pylist()
                    texts = [task.prompt(row) for row in raw]
                    if split=="train" and len(self.init_ids)<8:
                        self.init_ids.extend(tokenizer(texts[:8-len(self.init_ids)],max_length=1024,truncation=True).input_ids)
                    encoded.extend(array("H",t) for t in tokenizer(texts,max_length=128,truncation=True).input_ids)
                    rows.extend({"idx":r["idx"],"label":r["label"]} for r in raw)
            if len(rows)!=count or len({r["idx"] for r in rows})!=count:
                raise ValueError(f"Bad split size or duplicate ID: {name}/{split}")
            assert {r["label"] for r in rows}==set(range(len(task.labels)))
            self.encoded[split], self.rows[split] = encoded, rows

    def batch(self, indices, split="train", initializing=False):
        source = self.init_ids if initializing else self.encoded[split]
        return collate([source[i] for i in indices],
                       [self.targets[self.rows[split][i]["label"]] for i in indices],self.pad_id,self.device)

    def calibration(self):
        return [dict(position=i,idx=r["idx"],label=r["label"],text=self.task.labels[r["label"]],
                     input_tokens=len(self.init_ids[i])) for i,r in enumerate(self.rows["train"][:8])]


def predict(scores):
    top = max(scores)
    return scores.index(top) if scores.count(top)==1 else -1


@torch.inference_mode()
def evaluate_split(model, corpus, split, output, batch_size=8):
    model.eval()
    correct = 0
    rows = corpus.rows[split]
    with output.open("x",encoding="utf8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx","gold","pred",*corpus.task.labels])
        for start in range(0,len(rows),batch_size):
            ids = range(start,min(start+batch_size,len(rows)))
            batch = corpus.batch(ids,split)
            logits = model(**batch).logits[:,0,corpus.candidates]
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite candidate score")
            for i, scores in zip(ids,logits.cpu().tolist()):
                pred = predict(scores)
                r = rows[i]
                correct += pred==r["label"]
                writer.writerow([r["idx"],r["label"],pred,*scores])
    return correct/len(rows)
