"""Rebuild the report's main table from archived predictions using only stdlib.

python3 report/analyze_main.py                # verify the table and print evidence
python3 report/analyze_main.py --write-table  # update only the marked table
"""
import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/glue-v1"
REPORT = ROOT / "report/report.md"
START = "<!-- main-table:start -->"
END = "<!-- main-table:end -->"
NAMES = {"mnli": "MNLI matched", "sst2": "SST-2", "cola": "CoLA", "qnli": "QNLI", "mrpc": "MRPC"}


def analyze():
    spec = importlib.util.spec_from_file_location("archive_verifier", RESULTS / "verify_results.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    paper = json.loads((ROOT / "paper_reference.json").read_text(encoding="utf8"))
    records = verifier.rows((RESULTS / "per_seed.csv").read_bytes())
    observations, predictions = {}, {}
    for task in paper["task_order"]:
        split = "validation_matched" if task == "mnli" else "validation"
        for method in ("lora", "ga"):
            for seed in range(3):
                rows = [r for r in records if (r["task"], r["method"], r["seed"]) == (task, method, str(seed))]
                archive = RESULTS / "evidence" / task / method / f"seed-{seed}.zip"
                # Verifies member hashes and recomputes accuracy from candidate scores.
                scores = verifier.verify_run(archive, rows)
                observations[task, method, seed] = scores[split]
                row = next(r for r in rows if r["split"] == split)
                with ZipFile(archive) as z:
                    cases = csv.DictReader(io.StringIO(z.read(row["predictions"]).decode("utf-8-sig")))
                    predictions[task, method, seed] = {int(r["idx"]): (int(r["gold"]), int(r["pred"])) for r in cases}
        gold = {idx: p[0] for idx, p in predictions[task, "lora", 0].items()}
        for method in ("lora", "ga"):
            for seed in range(3):
                verifier.require({idx: p[0] for idx, p in predictions[task, method, seed].items()} == gold,
                                 f"Validation IDs or labels differ: {task}/{method}/{seed}")

    table = ["| Method | MNLI-m | SST-2 | CoLA | QNLI | MRPC | Avg. |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    gains = {}
    for i, task in enumerate(paper["task_order"]):
        values = {m: [observations[task, m, s] for s in range(3)] for m in ("lora", "ga")}
        gains[task] = [g - l for l, g in zip(values["lora"], values["ga"])]
    averages = {m: [mean(observations[t, m, s] for t in NAMES) for s in range(3)] for m in ("lora", "ga")}
    delta = [g - l for l, g in zip(averages["lora"], averages["ga"])]
    printed = paper["reported_average"]
    for source in ("Paper", "Ours"):
        for method, label in (("lora", "LoRA"), ("ga", "LoRA-GA")):
            cells = [f"{label} ({source})"]
            for i, task in enumerate(paper["task_order"]):
                if source == "Paper":
                    average, dispersion = paper["table1"][method][i]
                else:
                    values = [observations[task, method, s] for s in range(3)]
                    average, dispersion = mean(values), stdev(values)
                cells.append(f"{average:.2f} ± {dispersion:.2f}")
            average = printed[method] if source == "Paper" else mean(averages[method])
            cells.append(f"{average:.2f}" + ("*" if source == "Paper" and method == "lora" else ""))
            table.append("| " + " | ".join(cells) + " |")

    print("30/30 主实验已从归档预测复算；同任务的验证 ID 和真实标签一致。")
    for task, values in gains.items():
        print(f"{NAMES[task]}：seed 0/1/2 配对提升 = " + ", ".join(f"{x:+.4f}" for x in values) + " pp")
    print("五任务等权平均的配对提升：" + ", ".join(f"{x:.4f}" for x in delta) + " pp")
    print(f"CoLA + MRPC 对五任务平均提升的贡献：{(mean(gains['cola']) + mean(gains['mrpc'])) / 5:.6f} pp")
    print(f"其余三个任务的平均提升：{mean(mean(gains[t]) for t in ('mnli', 'sst2', 'qnli')):.6f} pp")
    for seed in range(3):
        cases = {m: predictions['sst2', m, seed] for m in ('lora', 'ga')}
        counts = {m: sum(g == p for g, p in rows.values()) for m, rows in cases.items()}
        print(f"SST-2 seed {seed}：LoRA {counts['lora']}/872，GA {counts['ga']}/872，答对数差 {counts['ga'] - counts['lora']:+d}")
    for method in ("lora", "ga"):
        deviations = [mean(observations[t, method, s] for s in range(3)) - paper['table1'][method][i][0]
                      for i, t in enumerate(paper['task_order'])]
        print(f"{method} 相对论文各任务偏差：" + ", ".join(f"{x:+.4f}" for x in deviations) + " pp")
    for seed in range(3):
        cases = predictions['cola', 'ga', seed]
        print(f"CoLA GA seed {seed}：{sum(g == p for g, p in cases.values())}/{len(cases)}")
    first = predictions['cola', 'ga', 0]
    print("CoLA GA seed 0 与 seed 1/2 的预测不一致条数：" + ", ".join(
        str(sum(first[idx][1] != predictions['cola', 'ga', seed][idx][1] for idx in first)) for seed in (1, 2)))
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-table", action="store_true")
    args = parser.parse_args()
    table = analyze()
    text = REPORT.read_text(encoding="utf8")
    before, remainder = text.split(START)
    existing, after = remainder.split(END)
    replacement = "\n\n" + table + "\n\n"
    if args.write_table:
        REPORT.write_text(before + START + replacement + END + after, encoding="utf8")
        print("已更新 report/report.md 的主实验表；其他正文未改动。")
    elif existing != replacement:
        raise SystemExit("报告表格与归档预测不一致，请检查或运行 --write-table。")
    else:
        print("报告主实验表与归档预测一致。")


if __name__ == "__main__":
    main()
