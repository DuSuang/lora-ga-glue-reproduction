"""Build a portable glue-v1 experiment archive and replay its CPU inventory and tables."""
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from glue.analyze_glue import summarize, plot
from glue.audit_glue_data import audit
from glue.prepare import MODEL_HASH
from glue import machine
from glue.machine import ROOT
from glue.catalog import DONE, jobs, locate, checksum, required
from glue.glue_tasks import TASKS


# Only reviewed source, documentation and pinned reference files enter the archive.
CORE = (
    "suite.py", "README.md", "requirements.txt", "docs/validation/windows-requirements.txt",
    "experiments/glue_files.json", "experiments/paper_reference.json", "experiments/run_a100_4gpu.sh",
    "docs/data/MSR-SSLA.rtf", "docs/data/MRPC-NOTICE.txt", "docs/data/数据来源.md",
    "docs/validation/glue_files_actual.json", "docs/validation/loss_protocol.json",
    "reference/LICENSE","reference/data.py","reference/utils.py","reference/run_exp.py","reference/logTrainer.py",
    "reference/conf/config.yaml","reference/conf/model/t5base.yaml","reference/conf/peft/all.yaml",
    "reference/conf/init/default.yaml","reference/conf/init/gradient.yaml","reference/conf/init/gaussian.yaml",
)
MODULES = ("__init__", "catalog", "machine", "adapters", "first_step", "glue_train", "glue_data", "glue_tasks",
           "model_runtime", "system_io", "prepare", "audit_glue_data", "analyze_glue", "package_glue", "cpu_check")
TESTS = ("__init__", "test_analysis", "test_calibration", "test_first_step", "test_glue_data", "test_initializers",
         "test_loss_protocol", "test_package_glue", "test_portability", "test_report_tables", "test_suite")
CORE += tuple(f"glue/{n}.py" for n in MODULES) + tuple(f"tests/{n}.py" for n in TESTS)


def build(archive,campaign,tables,report=None):
    site=machine.current
    data=audit(site.data_dir)
    if not all(r["status"]=="verified" for r in data["files"]):
        raise ValueError("Verify data with suite.py audit before packaging")
    paths={name:ROOT/name for name in CORE}
    paths["assets/mrpc/MSR-SSLA.rtf"]=ROOT/"docs/data/MSR-SSLA.rtf"
    paths["assets/mrpc/NOTICE.txt"]=ROOT/"docs/data/MRPC-NOTICE.txt"
    if report is not None:
        ext=report.suffix.lower()
        if not report.is_file() or ext not in (".pdf",".docx",".md"):
            raise ValueError("Expected an existing PDF, DOCX or Markdown report")
        paths["report/final"+ext]=report
    for name,digest in MODEL_HASH.items():
        if name=="model.safetensors":
            continue
        source=site.model_dir/name
        if checksum(source)!=digest:
            raise ValueError(f"Tokenizer/config checksum mismatch: {source}")
        paths[f"assets/t5-base/{name}"]=source
    for r in data["files"]:
        relative=Path(r["path"]).relative_to("assets")
        paths[r["path"]]=site.data_dir/relative
    completed=summarize(tables,campaign)
    plot(tables,completed)
    for p in tables.iterdir():
        paths["analysis_output/glue-v1/"+p.name]=p
    for seed in range(3):
        relative=Path("diagnostics")/f"mrpc-s{seed}"/"diagnostic.json"
        diagnostic=campaign.parent/relative
        if diagnostic.is_file():
            paths["runs/"+relative.as_posix()]=diagnostic
    inventory={}
    dirs=[]
    for j in jobs():
        status,attempt=locate(j,campaign)
        inventory[j.key]=status
        if attempt is None:
            continue
        prefix="runs/glue-v1/"+attempt.relative_to(campaign).as_posix()+"/"
        dirs.append(prefix)
        steps=math.ceil(TASKS[j.task].train/32)
        for n in required(j,steps)|{"complete.json"}:
            p=attempt/n
            if status==DONE or p.is_file():
                paths[prefix+n]=p
    manifest=dict(format="glue-v1",inventory=inventory,sha256={n:checksum(p) for n,p in paths.items()})
    with ZipFile(archive,"x",compression=ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for n in dirs:
            z.writestr(n,b"")
        for name,path in sorted(paths.items()):
            z.write(path,name)
        z.writestr("package_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def verify_files(folder,manifest):
    for n,h in manifest["sha256"].items():
        if not (folder/n).is_file() or checksum(folder/n)!=h:
            raise ValueError(f"Missing or changed archive member: {n}")
    missing=set(CORE)-set(manifest["sha256"])
    if missing:
        raise ValueError(f"Missing required source or notice: {missing}")


def table_content(path):
    if path.suffix==".json":
        return json.loads(path.read_text(encoding="utf8"))
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.reader(f))


def replay(folder,manifest):
    verify_files(folder,manifest)
    # These entry points must run using the extracted source and the standard library alone.
    def cli(*args):
        command=[sys.executable,"-E","-X","utf8","-S",*args]
        return subprocess.run(command,cwd=folder,check=True,capture_output=True,encoding="utf8").stdout
    listing=cli("suite.py","list")
    inventory=dict(line.split()[:2] for line in listing.splitlines() if not line.startswith("39 configs"))
    if inventory!=manifest["inventory"]:
        raise ValueError("Source/extracted inventory mismatch")
    cli("suite.py","analyze","--analysis-dir","rebuilt")
    for table in (folder/"analysis_output"/"glue-v1").iterdir():
        rebuilt=folder/"rebuilt"/table.name
        if table.suffix in (".csv",".json") and table_content(table)!=table_content(rebuilt):
            raise ValueError(f"Reconstructed metric or provenance differs: {table.name}")
    return sum(s==DONE for s in inventory.values())


def roundtrip(temp,campaign=None,report=None):
    campaign=machine.current.formal if campaign is None else campaign
    archive=temp/"glue-v1.zip"
    manifest=build(archive,campaign,temp/"tables",report)
    folder=temp/"extracted project"
    with ZipFile(archive) as z:
        z.extractall(folder)
    recorded=json.loads((folder/"package_manifest.json").read_text(encoding="utf8"))
    if recorded!=manifest:
        raise ValueError("Archive manifest mismatch")
    return archive,folder,replay(folder,manifest)


def package(output,report):
    if output is not None and output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=machine.current.cache_dir) as tmp:
        archive,_,count=roundtrip(Path(tmp),report=report)
        print(f"Verified extracted inventory and reconstructed tables: {count}/39 complete")
        if output is not None:
            output.parent.mkdir(parents=True,exist_ok=True)
            with archive.open("rb") as src,output.open("xb") as dest:
                shutil.copyfileobj(src,dest)
            print(output)
