import json
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from glue.analyze_glue import read
from glue import machine
from glue.package_glue import CORE, roundtrip, table_content, verify_files
from glue.prepare import MODEL_HASH
from zipfile import ZipFile
from tests.test_report_tables import fixture


class PackageTests(unittest.TestCase):
    def test_optional_diagnostics_from_external_runs_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            runs=work/"external runs"
            expected={}
            for seed in (0,2):
                relative=Path("diagnostics")/f"mrpc-s{seed}"/"diagnostic.json"
                source=runs/relative
                source.parent.mkdir(parents=True)
                source.write_text(json.dumps({"seed":seed,"synthetic":True}),encoding="utf8")
                expected["runs/"+relative.as_posix()]=source.read_bytes()
                source.with_name("notes.txt").write_text("scratch",encoding="utf8")
            output=work/"materials.zip"
            result=subprocess.run([sys.executable,"-E","-X","utf8",str(machine.ROOT/"suite.py"),"package","build",
                                   "--runs-dir",str(runs),"--cache-dir",str(work/"cache"),
                                   "--model-dir",str(machine.current.model_dir),"--data-dir",str(machine.current.data_dir),
                                   "--output",str(output)],cwd=work,check=True,capture_output=True,encoding="utf8")
            self.assertIn("0/39 complete",result.stdout)
            extracted=work/"extracted"
            with ZipFile(output) as archive:
                self.assertEqual({n for n in archive.namelist() if n.startswith("runs/")},set(expected))
                archive.extractall(extracted)
            manifest=json.loads((extracted/"package_manifest.json").read_text(encoding="utf8"))
            for name,content in expected.items():
                self.assertIn(name,manifest["sha256"])
                self.assertEqual((extracted/name).read_bytes(),content)
            verify_files(extracted,manifest)
            name="runs/diagnostics/mrpc-s2/diagnostic.json"
            (extracted/name).write_text("{}",encoding="utf8")
            with self.assertRaisesRegex(ValueError,"Missing or changed archive member: "+name):
                verify_files(extracted,manifest)

    def test_external_assets_campaign_and_report(self):
        # The clean checkout has no assets or runs; the CLI must resolve both from the config.
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            checkout=work/"clean source"
            for name in CORE:
                dest=checkout/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(machine.ROOT/name,dest)
            for name in ("docs/unrelated.md","glue/unrelated.py","unrequested.pdf","other.zip"):
                (checkout/name).write_bytes(b"excluded")
            config=dict(data_dir="input data",model_dir="model files",runs_dir="experiment runs",
                        analysis_dir="metrics",cache_dir="scratch space")
            (work/"machine.json").write_text(json.dumps(config),encoding="utf8")
            source=machine.current
            manifest=json.loads((checkout/"experiments/glue_files.json").read_text(encoding="utf8"))
            for name,splits in manifest["tasks"].items():
                folder=work/config["data_dir"]/name
                folder.mkdir(parents=True)
                for parts in splits.values():
                    for part in parts:
                        shutil.copyfile(source.data_dir/name/part["name"],folder/part["name"])
            model=work/config["model_dir"]
            model.mkdir()
            for name in MODEL_HASH.keys()-{"model.safetensors"}:
                shutil.copyfile(source.model_dir/name,model/name)
            fixture(work/config["runs_dir"]/"glue-v1")
            report=work/"external report.md"
            report.write_text("fixture report",encoding="utf8")
            def cli(*args):
                return subprocess.run([sys.executable,"-E","-X","utf8",str(checkout/"suite.py"),*args,
                                       "--config","machine.json"],cwd=work,check=True,capture_output=True,
                                      encoding="utf8").stdout
            cli("prepare","--cpu-only")
            self.assertFalse((model/".prepare.lock").exists())
            self.assertIn("完整完成",cli("list","--task","mrpc","--seed","0","--method","lora"))
            self.assertIn("1/39",cli("analyze"))
            observed=[r for r in read(work/"metrics"/"per_seed.csv") if r["accuracy"]]
            self.assertEqual([(r["task"],r["method"],r["accuracy"]) for r in observed],[("mrpc","lora","50.0")])
            self.assertIn("1/39",cli("package","build","--output","materials.zip","--report",report.name))
            with ZipFile(work/"materials.zip") as archive:
                self.assertEqual(archive.read("report/final.md"),report.read_bytes())
                for name in ("docs/unrelated.md","glue/unrelated.py","unrequested.pdf","other.zip","machine.json"):
                    self.assertNotIn(name,archive.namelist())
            self.assertFalse((checkout/"assets").exists())
            self.assertFalse((checkout/"runs").exists())

    def test_regenerated_table_line_endings_and_values(self):
        with tempfile.TemporaryDirectory() as temp:
            p,q=[Path(temp)/name for name in ("p.csv","q.csv")]
            p.write_bytes(b"accuracy\r\n50.0\r\n")
            q.write_bytes(b"accuracy\n50.0\n")
            self.assertEqual(table_content(p),table_content(q))
            q.write_bytes(b"accuracy\n5.0\n")
            self.assertNotEqual(table_content(p),table_content(q))

    def test_extract_completed_attempt_and_detect_omissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            source=d/"synthetic"
            fixture(source)
            _,folder,count=roundtrip(d,source)
            self.assertEqual(count,1)
            manifest=json.loads((folder/"package_manifest.json").read_text(encoding="utf8"))
            rows=read(folder/"rebuilt"/"per_seed.csv")
            scored=[r for r in rows if r["accuracy"]]
            self.assertEqual(len(scored),1)
            self.assertEqual(float(scored[0]["accuracy"]),50.)
            prefix="runs/glue-v1/"+scored[0]["attempt"]+"/"
            for name in [prefix+"initial.pt",prefix+"final.pt","runs/glue-v1/"+scored[0]["predictions"],"glue/glue_train.py"]:
                with self.subTest(name=name):
                    p=folder/name
                    saved=p.read_bytes()
                    p.unlink()
                    with self.assertRaisesRegex(ValueError,"Missing or changed archive member"):
                        verify_files(folder,manifest)
                    p.write_bytes(saved)
