"""Fetch immutable Hugging Face assets. Existing downloads are verified before use."""
from glue.system_io import file_lock
import json
import hashlib
from glue.glue_tasks import TASKS
from glue import machine
from glue.machine import ROOT
import time

import requests

MODEL_REV = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
DATA_REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
WEIGHT_HASH = "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"
MODEL_HASH = {"model.safetensors": WEIGHT_HASH,
              "config.json": "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033",
              "generation_config.json": "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
              "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
              "tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(job):
    path, url, expected = job
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected and digest(path) == expected:
        return
    part = path.with_suffix(path.suffix + ".part")
    if part.exists() and digest(part) == expected:
        part.replace(path)
        return
    for attempt in range(8):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416 and start:
                    part.unlink()
                    raise ValueError("Discarded partial file with an invalid resume offset")
                response.raise_for_status()
                append = start and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("server returned an unexpected byte range")
                with part.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if expected and digest(part) != expected:
                part.unlink()
                raise ValueError(f"SHA256 mismatch for {path.name}")
            part.replace(path)
            print("verified", path.name, path.stat().st_size, flush=True)
            return
        except (requests.RequestException, ValueError) as exc:
            print(path.name, str(exc).split("?")[0][:180], flush=True)
            if attempt == 7:
                raise
            time.sleep(min(15, 2 ** attempt))


def prepare_group(folder,jobs):
    # A fully populated shared asset directory can be mounted read-only.
    if all(path.is_file() and digest(path)==sha for path,_,sha in jobs):
        return
    with file_lock(folder/".prepare.lock"):
        for job in jobs:
            fetch(job)


def prepare_assets(tasks,weights=True):
    model_url=f"https://huggingface.co/google-t5/t5-base/resolve/{MODEL_REV}"
    model=machine.current.model_dir
    prepare_group(model,[(model/name,f"{model_url}/{name}?download=true",sha)
                         for name,sha in MODEL_HASH.items() if weights or name!="model.safetensors"])
    repo=f"https://huggingface.co/datasets/nyu-mll/glue/resolve/{DATA_REV}"
    jobs=[]
    for task in tasks:
        for parts in locked_files(task).values():
            for part in parts:
                jobs.append((machine.current.data_dir/task/part["name"],f"{repo}/{part['source']}?download=true",part["sha256"]))
    prepare_group(machine.current.data_dir,jobs)


def locked_files(name):
    meta=json.loads((ROOT/"glue_files.json").read_text(encoding="utf8"))
    assert meta["revision"]==DATA_REV
    files=meta["tasks"][name]
    for split,n in {"train":TASKS[name].train,**TASKS[name].validation}.items():
        assert sum(part["rows"] for part in files[split])==n
    return files
