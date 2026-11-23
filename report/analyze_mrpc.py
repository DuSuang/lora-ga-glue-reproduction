"""Verify MRPC ablation results and class predictions from archived evidence.

python3 report/analyze_mrpc.py
python3 report/analyze_mrpc.py --write-table
"""
import argparse
import csv
import importlib.util
import io
import json
from statistics import mean, stdev
from zipfile import ZipFile

from analyze_main import ROOT, RESULTS, REPORT


NAMES = {"lora": "LoRA", "gaussian": "Gaussian", "so": "+SO",
         "ga_only": "+GA", "ga": "LoRA-GA"}
START = "<!-- mrpc-table:start -->"
END = "<!-- mrpc-table:end -->"


def analyze():
    spec = importlib.util.spec_from_file_location("archive_verifier", RESULTS / "verify_results.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    paper = json.loads((ROOT / "paper_reference.json").read_text(encoding="utf8"))["table9_mrpc"]
    records = verifier.rows((RESULTS / "per_seed.csv").read_bytes())
    scores, cases = {}, {}
    gold = None
    for method in NAMES:
        scores[method] = []
        for seed in range(3):
            rows = [r for r in records if (r['task'], r['method'], r['seed']) == ('mrpc', method, str(seed))]
            verifier.require(len(rows) == 1, f"Unexpected MRPC inventory: {method}/{seed}")
            archive = RESULTS / "evidence/mrpc" / method / f"seed-{seed}.zip"
            scores[method].append(verifier.verify_run(archive, rows)['validation'])
            with ZipFile(archive) as z:
                predictions = csv.DictReader(io.StringIO(z.read(rows[0]['predictions']).decode('utf-8-sig')))
                data = {int(r['idx']): (int(r['gold']), int(r['pred'])) for r in predictions}
            actual_gold = {idx: p[0] for idx, p in data.items()}
            if gold is None:
                gold = actual_gold
            verifier.require(actual_gold == gold, f"Different validation IDs or gold labels: {method}/{seed}")
            cases[method, seed] = data
    verifier.require(len(gold) == 408 and sum(v == 0 for v in gold.values()) == 129
                     and sum(v == 1 for v in gold.values()) == 279, "Unexpected MRPC label distribution")
    print("15/15 MRPC 结果已从归档预测复算，包含复用的 6 项主实验。")
    table = ["| Method | Paper | Ours |", "|---|---:|---:|"]
    for method, name in NAMES.items():
        values = scores[method]
        gain = mean(values) - mean(scores['lora'])
        table.append(f"| {name} | {paper[method][0]:.2f} ± {paper[method][1]:.2f} | "
                     f"{mean(values):.2f} ± {stdev(values):.2f} |")
        print(f"{name} 相对本组 LoRA 的平均增益：{gain:.6f} pp")
        recalls = []
        for seed in range(3):
            data = cases[method, seed]
            cm = [[0, 0, 0], [0, 0, 0]]  # rows: gold; columns: different/equivalent/tie
            for actual, predicted in data.values():
                cm[actual][predicted] += 1
            old = cases['lora', seed]
            corrected = sum(old[i][1] != actual and pred == actual for i, (actual, pred) in data.items())
            introduced = sum(old[i][1] == actual and pred != actual for i, (actual, pred) in data.items())
            verifier.close((corrected - introduced) / 408 * 100, values[seed] - scores['lora'][seed], name)
            recalls.append((cm[0][0] / 129 * 100, cm[1][1] / 279 * 100))
            print(f"{name} seed {seed}：混淆矩阵 {cm}；相对 LoRA 纠正 {corrected} 条、新增错误 {introduced} 条")
        print(f"{name}：平均准确率 {mean(values):.6f}%，样本 SD {stdev(values):.6f} pp；"
              f"非等价/等价类平均召回率 {mean(r[0] for r in recalls):.4f}%/{mean(r[1] for r in recalls):.4f}%")
    for method in ('so', 'ga_only'):
        print(f"完整 GA 相对 {NAMES[method]} 的平均增益：{mean(scores['ga']) - mean(scores[method]):.6f} pp")
    print("完整 GA 在每个种子下是否均为最佳：", all(
        scores['ga'][seed] > scores[m][seed] for seed in range(3) for m in NAMES if m != 'ga'))
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-table', action='store_true')
    args = parser.parse_args()
    table = analyze()
    text = REPORT.read_text(encoding='utf8')
    if START not in text and not args.write_table:
        print(table)
        print('MRPC 正文尚未审核并入报告；本次仅复算，不修改报告。')
        return
    before, remainder = text.split(START)
    existing, after = remainder.split(END)
    replacement = '\n\n' + table + '\n\n'
    if args.write_table:
        REPORT.write_text(before + START + replacement + END + after, encoding='utf8')
        print('已更新报告 MRPC 消融表；其他正文未改动。')
    elif existing != replacement:
        raise SystemExit('报告 MRPC 消融表与归档预测不一致，请检查或运行 --write-table。')
    else:
        print('报告 MRPC 消融表与归档预测一致。')


if __name__ == '__main__':
    main()
