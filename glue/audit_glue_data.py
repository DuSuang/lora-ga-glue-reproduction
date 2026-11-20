"""Check the actual Parquet files against the pinned GLUE manifest, independently of training."""
import json
from pathlib import Path

import pyarrow.parquet as pq

from glue.glue_tasks import TASKS
from glue.machine import ROOT
from glue.catalog import checksum


def audit(data_dir):
    manifest=json.loads((ROOT/"glue_files.json").read_text(encoding="utf8"))
    records=[]
    for name,splits in manifest["tasks"].items():
        task=TASKS[name]
        for split,parts in splits.items():
            seen=set()
            for part in parts:
                rel=Path("assets")/name/part["name"]
                path=data_dir/name/part['name']
                expected={key:part[key] for key in ("sha256","bytes","rows")}
                expected.update(required_fields=["idx","label",*task.fields],labels=list(range(len(task.labels))),
                                unique_ids=part["rows"])
                actual={}
                record=dict(path=rel.as_posix(),split=split,source=part["source"],expected=expected,actual=actual)
                records.append(record)
                try:
                    actual.update(bytes=path.stat().st_size,sha256=checksum(path))
                    table=pq.read_table(path)
                    actual["fields"]=table.column_names
                    ids=table["idx"].to_pylist()
                    distinct=set(ids)
                    actual.update(rows=table.num_rows,labels=sorted(set(table["label"].to_pylist())),
                                  unique_ids=len(distinct),overlap_preceding_files=len(distinct & seen),
                                  nulls={key:table[key].null_count for key in expected["required_fields"]})
                    seen.update(distinct)
                    valid=all(expected[k]==actual[k] for k in ("bytes","sha256","rows","labels","unique_ids"))
                    record["status"]="verified" if valid and not actual["overlap_preceding_files"] and not any(actual["nulls"].values()) else "mismatch"
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    record.update(status="missing" if not path.is_file() else "unreadable",error=str(exc))
    return dict(revision=manifest["revision"],files=records)


def write_audit(data_dir,out):
    report=audit(data_dir)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for r in report["files"]:
        print(r["path"],r["status"],r["actual"].get("rows"),r["actual"].get("sha256"))
    return all(r["status"]=="verified" for r in report["files"])
