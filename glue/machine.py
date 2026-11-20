"""Machine-specific settings. Protocol constants live in the task and adapter modules."""
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT/"assets"
    model_dir: Path = ROOT/"assets"/"t5-base"
    runs_dir: Path = ROOT/"runs"
    analysis_dir: Path = ROOT/"analysis_output"/"glue-v1"
    cache_dir: Path = ROOT/".cache"
    device: str = "cuda:0"
    microbatch: int = 8
    eval_batch: int = 8

    @property
    def formal(self):
        return self.runs_dir/"glue-v1"


current = Settings()


def options(parser):
    parser.add_argument("--config",type=Path,help="Machine settings JSON; relative paths resolve beside this file")
    for field in fields(Settings):
        parser.add_argument("--"+field.name.replace("_","-"),type=field.type,default=None)


def configure(args):
    global current
    values={}
    if args.config:
        values=json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(values,dict) or set(values)-{f.name for f in fields(Settings)}:
            raise ValueError("Unknown machine settings")
    for field in fields(Settings):
        value=getattr(args,field.name)
        if value is not None:
            values[field.name]=value.resolve() if field.type is Path else value
        elif field.type is Path and field.name in values:
            values[field.name]=(args.config.resolve().parent/Path(values[field.name]).expanduser()).resolve()
    site=Settings(**values)
    if (not isinstance(site.device,str) or not re.fullmatch(r"cuda:[0-9]+",site.device)
            or type(site.microbatch) is not int or site.microbatch not in (1,2,4,8,16,32)
            or type(site.eval_batch) is not int or site.eval_batch<1):
        raise ValueError("Expected cuda:N, microbatch in {1,2,4,8,16,32} and positive eval_batch")
    current=site
    return site


def setup_cache():
    path=current.cache_dir
    scratch=path/"temp"
    scratch.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(scratch)
    for key in ("TMP","TEMP","TMPDIR"):
        os.environ[key]=str(scratch)
    os.environ["HF_HOME"]=str(path/"hf")
    os.environ["HF_DATASETS_CACHE"]=str(path/"hf"/"datasets")
    os.environ["MPLCONFIGDIR"]=str(path/"matplotlib")
