"""Recalculate the published GLUE accuracies from the original final predictions."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
TASKS = ("mnli", "sst2", "cola", "qnli", "mrpc")
MAIN = ("lora", "ga")
EXTRA = ("gaussian", "so", "ga_only")


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(actual, reported, label):
    require(math.isclose(actual, float(reported), rel_tol=0, abs_tol=1e-9), label)


def verify_run(archive, run_rows):
    # The published archive contains an audited subset of the original attempt.
    first = run_rows[0]
    attempt = first["attempt"]
    with ZipFile(archive) as z:
        proof = json.loads(z.read(f"{attempt}/complete.json"))
        key = f"{first['task']}/{first['method']}/seed-{first['seed']}"
        require(proof["key"] == key, key)
        for name in z.namelist():
            if name == f"{attempt}/complete.json":
                continue
            rel = Path(name).name
            require(hashlib.sha256(z.read(name)).hexdigest() == proof["sha256"][rel], name)
        config = json.loads(z.read(f"{attempt}/config.json"))
        require(config["key"] == key and not config["smoke"], key)
        require(config["steps"] == proof["steps"], key)
        logs = rows(z.read(f"{attempt}/evaluation.csv"))
        scores = {}
        for row in run_rows:
            correct = 0
            seen = set()
            for case in rows(z.read(row["predictions"])):
                values = [float(case[label]) for label in config["labels"]]
                require(all(math.isfinite(v) for v in values), row["predictions"])
                top = max(values)
                prediction = values.index(top) if values.count(top) == 1 else -1
                require(prediction == int(case["pred"]) and case["idx"] not in seen, case["idx"])
                seen.add(case["idx"])
                correct += prediction == int(case["gold"])
            size = len(seen)
            require(size == int(row["count"]) and correct == int(row["correct"]), row["predictions"])
            end = [r for r in logs if r["split"] == row["split"] and int(r["step"]) == config["steps"]]
            require(len(end) == 1 and int(row["step"]) == config["steps"], key)
            close(correct / size, end[0]["accuracy"], key)
            accuracy = 100 * correct / size
            close(accuracy, row["accuracy"], row["predictions"])
            scores[row["split"]] = accuracy
        return scores


def main():
    per_seed = rows((ROOT / "per_seed.csv").read_bytes())
    expected = {(t, m, s) for t in TASKS for m in MAIN for s in range(3)}
    expected.update(("mrpc", m, s) for m in EXTRA for s in range(3))
    runs = {}
    for row in per_seed:
        require(row["status"] == "完整完成", row)
        runs.setdefault((row["task"], row["method"], int(row["seed"])), []).append(row)
    require(set(runs) == expected, "Unexpected run inventory")
    observations = {}
    for (task, method, seed), run_rows in runs.items():
        archive = ROOT / "evidence" / task / method / f"seed-{seed}.zip"
        scores = verify_run(archive, run_rows)
        splits = {"validation_matched", "validation_mismatched"} if task == "mnli" else {"validation"}
        require(splits == set(scores) and len(run_rows) == len(splits), str(archive))
        for split, accuracy in scores.items():
            observations[task, method, split, seed] = accuracy

    for row in rows((ROOT / "table1_comparison.csv").read_bytes()):
        task, split = row["task"], row["split"]
        paired = {}
        for m in MAIN:
            if task in TASKS:
                values = [observations[task, m, split, s] for s in range(3)]
            else:
                require(task in ("paper_printed_average", "recomputed_from_five_tasks"), task)
                values = [mean(observations[t, m, "validation_matched" if t == "mnli" else "validation", s]
                               for t in TASKS) for s in range(3)]
            close(mean(values), row[f"local_{m}_pct"], task)
            close(stdev(values), row[f"local_{m}_sd_pp"], task)
            paired[m] = values
        gains = [g - l for l, g in zip(paired["lora"], paired["ga"])]
        close(mean(gains), row["paired_mean_pp"], task)
        for s, gain in enumerate(gains):
            close(gain, row[f"paired_s{s}_pp"], task)
    for row in rows((ROOT / "table9_mrpc.csv").read_bytes()):
        values = [observations["mrpc", row["method"], "validation", s] for s in range(3)]
        close(mean(values), row["local_mean_pct"], row["method"])
        close(stdev(values), row["local_sample_sd_pp"], row["method"])
    print("39/39 verified; final predictions match per-seed scores, Table 1 and Table 9.")


if __name__ == "__main__":
    main()
