"""Plot archived CoLA and MRPC training curves without rerunning training.

Install report/plot-requirements.txt, then run:
    python report/plot_training.py --output report/figures

Lines and bands are the arithmetic mean and sample SD across seeds 0, 1, 2
at each recorded step. Loss is not smoothed. Validation markers show actual
evaluation steps; connecting lines do not represent additional evaluations.
"""

import argparse
import csv
import importlib.util
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/glue-v1"
METHODS = {
    "lora": ("LoRA", "#0072B2", "-", "o"),
    "gaussian": ("Gaussian", "#7B8492", "--", "x"),
    "so": ("+SO", "#009E73", "--", "^"),
    "ga_only": ("+GA", "#AA4499", "-.", "s"),
    "ga": ("LoRA-GA", "#D55E00", "-", "D"),
}
TASKS = {"cola": ("lora", "ga"), "mrpc": tuple(METHODS)}


def load_curves():
    spec = importlib.util.spec_from_file_location(
        "archive_verifier", RESULTS / "verify_results.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    records = verifier.rows((RESULTS / "per_seed.csv").read_bytes())
    curves = {}
    for task, methods in TASKS.items():
        task_grids = {}
        for method in methods:
            series = {"loss": [], "accuracy": []}
            for seed in range(3):
                run_rows = [r for r in records if
                            (r["task"], r["method"], r["seed"]) ==
                            (task, method, str(seed))]
                verifier.require(len(run_rows) == 1, f"Missing run: {task}/{method}/{seed}")
                archive = RESULTS / "evidence" / task / method / f"seed-{seed}.zip"
                # Checks every archived file hash and recomputes final accuracy.
                scores = verifier.verify_run(archive, run_rows)
                attempt = run_rows[0]["attempt"]
                with ZipFile(archive) as z:
                    for metric, filename in (("loss", "loss.csv"),
                                             ("accuracy", "evaluation.csv")):
                        rows = list(csv.DictReader(io.StringIO(
                            z.read(f"{attempt}/{filename}").decode("utf-8-sig"))))
                        if metric == "accuracy":
                            rows = [r for r in rows if r["split"] == "validation"]
                        steps = np.array([int(r["step"]) for r in rows])
                        values = np.array([float(r[metric]) for r in rows])
                        verifier.require(np.all(np.isfinite(values)), str(archive))
                        verifier.require(np.all(np.diff(steps) > 0), str(archive))
                        verifier.require(steps[-1] == int(run_rows[0]["step"]), str(archive))
                        if metric == "loss":
                            verifier.require(np.array_equal(steps, np.arange(1, steps[-1] + 1)),
                                             f"Missing loss steps: {archive}")
                            verifier.require(np.all(values > 0), f"Nonpositive loss: {archive}")
                        else:
                            verifier.require(np.all((values >= 0) & (values <= 1)), str(archive))
                            values *= 100
                            verifier.close(values[-1], scores["validation"], str(archive))
                        if metric not in task_grids:
                            task_grids[metric] = steps
                        verifier.require(np.array_equal(steps, task_grids[metric]),
                                         f"Different recorded steps: {archive}/{metric}")
                        series[metric].append(values)
            curves[task, method] = {}
            for metric, runs in series.items():
                values = np.stack(runs)
                curves[task, method][metric] = (
                    task_grids[metric], values.mean(axis=0), values.std(axis=0, ddof=1))
    return curves


def draw_task(task, curves, output):
    methods = TASKS[task]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24,
                        top=0.74, wspace=0.23)
    title = "CoLA: LoRA vs. LoRA-GA" if task == "cola" else "MRPC: Initialization ablation"
    fig.suptitle(title, y=0.97, fontsize=17, fontweight="semibold")
    handles = []
    for method in methods:
        label, color, style, marker = METHODS[method]
        handles.append(Line2D([], [], color=color, linestyle=style, marker=marker,
                              markerfacecolor="white", markersize=5, linewidth=2,
                              label=label))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.9),
               ncol=len(methods), frameon=False, columnspacing=2.0, handlelength=2.6)
    for ax, metric, title in zip(axes, ("loss", "accuracy"),
                                  ("(a) Training loss", "(b) Validation accuracy")):
        for index, method in enumerate(methods):
            label, color, style, marker = METHODS[method]
            steps, avg, sd = curves[task, method][metric]
            lower, upper = avg - sd, avg + sd
            if metric == "loss" and np.any(lower <= 0):
                raise ValueError("Loss SD band crosses zero; reconsider the log-axis display.")
            ax.fill_between(steps, lower, upper, color=color, alpha=0.12, linewidth=0,
                            zorder=1)
            ax.plot(steps, avg, color=color, linestyle=style,
                    linewidth=2.4 if method == "lora" else 1.9,
                    marker=marker if metric == "accuracy" else None,
                    markersize=4.5, markerfacecolor="white", markeredgewidth=1.2,
                    zorder=3 + index * 0.1)
        last = int(steps[-1])
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_xlabel("Training step", labelpad=7)
        ax.set_xlim(0, last * (1.19 if metric == "accuracy" else 1.02))
        ax.set_xticks([0, 50, 100, 150, 200, 250] if task == "cola"
                     else [0, 20, 40, 60, 80, 100, 115])
        ax.grid(axis="y", color="#E2E6EB", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C5CBD2")
        ax.tick_params(length=3, color="#A0A8B2")
        if metric == "loss":
            ax.set_ylabel("Token cross-entropy (log scale)", labelpad=8)
            ax.set_yscale("log")
            ax.set_ylim(0.03, 12)
            ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylabel("Accuracy (%)", labelpad=8)
            ax.set_ylim((45, 86) if task == "cola" else (25, 90))
            for method in methods:
                if method == "gaussian":
                    continue  # Its endpoint is identical to LoRA's.
                _, color, _, _ = METHODS[method]
                value = curves[task, method][metric][1][-1]
                text_y = value
                if task == "mrpc":
                    text_y += {"so": -1.8, "ga_only": 1.6}.get(method, 0)
                ax.annotate(f"{value:.2f}", xy=(last, value),
                            xytext=(last * 1.055, text_y), color=color, fontsize=10,
                            va="center", fontweight="medium",
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
                            zorder=10)
    note = "Mean ± 1 sample SD across 3 seeds. No smoothing; markers show recorded evaluations."
    if task == "mrpc":
        note += "\nLoRA and Gaussian accuracy curves overlap from step 35 onward."
    fig.text(0.075, 0.055, note, fontsize=9.4, color="#56616F", linespacing=1.6)
    for extension in ("png", "pdf"):
        path = output / f"{task}-training.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.labelsize": 11, "text.color": "#202A35",
                         "axes.labelcolor": "#202A35", "xtick.color": "#44515E",
                         "ytick.color": "#44515E", "pdf.fonttype": 42})
    curves = load_curves()
    for task in TASKS:
        draw_task(task, curves, args.output)
    print("Verified 21 archived runs; step grids and final accuracies match.")


if __name__ == "__main__":
    main()
