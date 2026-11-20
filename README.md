# 自然语言处理课程期中作业：LoRA-GA 论文复现

本作业复现 Shaowen Wang、Linxi Yu 和 Jian Li 发表于 NeurIPS 2024 的论文 *LoRA-GA: Low-Rank Adaptation with Gradient Approximation*。我们选取论文表 1 中 **T5-base 在五个 GLUE 任务上的 LoRA 和 LoRA-GA 两行**作为主实验，并选取附录表 9 的 **MRPC 列**做初始化消融。[论文全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf) · [作者代码（固定提交）](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef/reproduce)

## 作者源码与本组适配

本组在论文作者开源的 `Outsider565/LoRA-GA`（提交 `c4cd5372`）基础上适配了 T5-base 五任务实验。`reference/` 保留作者源码节选及许可证。作者的梯度估计与初始化位于 [reference/run_exp.py](reference/run_exp.py)；本组正式训练调用的核心方法位于 [glue/adapters.py](glue/adapters.py)。

| 算法步骤 | 作者实现（`reference/run_exp.py`） | 本组实现（`glue/adapters.py`） |
|---|---|---|
| 估计校准梯度 | `estimate_gradient` | `estimate` 对前 8 条训练样本逐条反向传播，再对各层梯度等权取平均。 |
| 初始化低秩矩阵 | `reinit_lora_modules` | 普通 LoRA 由 `Adapter.__init__` 随机初始化 A、将 B 置零；LoRA-GA 由 `initialize_factors` 对校准梯度做低秩 SVD，用奇异向量初始化 A、B 并调整尺度。 |
| 补偿初始权重 | `reinit_lora_modules` 中的 `offset` | `Adapter.compensate` 从基座权重减去初始低秩增量，使安装适配器时的有效权重与预训练权重一致。 |

运行 `python suite.py run` 时，训练函数 `execute` 调用 `adapters.install` 安装所选方法，再进入 `train_epoch`。本组为课程实验适配了 MNLI 三分类评测、变长标签的微批训练，并保存逐条预测，供三种子配对和论文表格核查。[训练流程](glue/glue_train.py) · [数据与评测](glue/glue_data.py) · [结果汇总](glue/analyze_glue.py)

## 主实验：表 1 的五任务分类准确率

普通 LoRA 冻结预训练权重，用两个低秩矩阵学习权重增量。LoRA-GA 用少量训练样本估计梯度，据此初始化两个矩阵，让第一次更新的方向更接近全参数微调。本组检验的是：在相同的训练预算下，这种初始化能否提高分类准确率，提升主要来自哪些任务？

下表是**论文表 1 报告的验证集 Accuracy 均值（%）**。最后一列按论文两行之差计算，单位为百分点。

| 任务 | 判断什么 | LoRA（论文） | LoRA-GA（论文） | 差值 |
|---|---|---:|---:|---:|
| MNLI | 两句话之间是蕴含、中立还是矛盾 | 85.30 | 85.70 | +0.40 |
| SST-2 | 句子的情感是正面还是负面 | 94.04 | 94.11 | +0.07 |
| CoLA | 英语句子在语法上是否可接受 | 69.35 | 80.57 | +11.22 |
| QNLI | 给定句子是否包含问题的答案 | 92.96 | 93.18 | +0.22 |
| MRPC | 两句话是否表达相同含义 | 68.38 | 85.29 | +16.91 |
| **Average（论文印刷值）** | **五任务平均** | **82.08** | **87.77** | **+5.69** |

复现时，每个任务分别运行 LoRA 和 LoRA-GA，采用相同的三个种子（0、1、2），共 **5 × 2 × 3 = 30 次训练**。比较时既看五任务平均，也看能否观察到论文中 CoLA、MRPC 提升较大的现象。本组汇总表并列论文值、三种子实测均值、样本标准差及逐种子差值。论文中 LoRA 的五项均值按展示值重算为 82.006%，汇总时保留这个重算值和论文印出的 82.08%。[论文数值清单](paper_reference.json)

## 补充实验：表 9 的 MRPC 初始化消融

表 1 中 MRPC 的两种方法相差 16.91 个百分点，因此选取它考察梯度方向与尺度调整各自的作用。初始化定义见论文表 3，以下参照值取自附录表 9 的 MRPC 列。

| 初始化方式 | LoRA | Gaussian | +SO（尺度调整） | +GA（梯度方向） | LoRA-GA（两者结合） |
|---|---:|---:|---:|---:|---:|
| 论文 MRPC Accuracy（%） | 68.38 | 68.38 | 67.97 | 66.09 | 85.29 |

LoRA 与 LoRA-GA 复用主实验的 MRPC 结果，中间三种方法各增加三个种子，共补充 **9 次训练**。主实验与消融合计 **39 次**。报告还可以用同一批训练日志绘制 loss 随更新步数变化的曲线，对照论文图 3 的收敛趋势。

## 实验设置与核验

- 使用 T5-base，在每个任务的完整训练集上训练 1 轮。两种主方法使用相同的样本顺序、更新预算和验证时点。主要参数为秩 8、LoRA α = 16、有效 batch 32、FP32、AdamW 和学习率 1e-4。
- 五个任务的主指标均为验证集 Accuracy。MNLI 在解码首位置比较三个候选标签，用 matched 验证集计入主表，并单列 mismatched。主表采用完整一轮的最后一步；作者公共训练脚本配置为按验证损失选择检查点。
- 测试直接执行作者的数据映射与编码函数，逐项检查五任务的提示词、标签和验证划分，并在微型 T5 上比较整批与微批累积的损失和梯度。[复现实验记录](docs/复现实验记录.md) · [数据来源](docs/data/数据来源.md)

## 如何运行

在项目根目录使用 Python 3.11。首次运行先创建并激活环境；Windows PowerShell 和 Linux Bash 分别执行对应的两行。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

安装项目测试使用的依赖。GPU 机器需要兼容 CUDA 12.1 的 NVIDIA 驱动；仅运行 CPU 验收时，将第一行末尾的 `cu121` 改为 `cpu`。[已验证的 Windows 环境](docs/validation/windows-requirements.txt)

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

**准备与检查。** `prepare` 校验已有的模型和数据，并补齐缺少的文件。只准备 CPU 验收材料时使用 `prepare --cpu-only`。`audit` 核验五任务的 11 份数据文件；`list` 显示当前批次进度。

```bash
python suite.py prepare
python suite.py audit
python suite.py list
python suite.py test
```

**GPU 实验。** 先用 MRPC 做三步短测，输出保存在 `runs/glue-smoke-v1/`。后两条命令分别执行表 1 的 30 项和表 9 额外的 9 项，正式输出保存在 `runs/glue-v1/`。

```bash
python suite.py run --smoke --task mrpc --method ga --seed 0
python suite.py run --scope main
python suite.py run --scope ablation
```

默认设备为 `cuda:0`，训练与验证的单次 batch 均为 8。换设备或使用共享存储时，可在子命令后指定 `--device`、`--microbatch`、`--eval-batch`、`--model-dir`、`--data-dir`、`--runs-dir`、`--analysis-dir` 和 `--cache-dir`，也可使用 `--config machine.local.json`。正式训练中断时，用 `python suite.py run --retry --task mrpc --method ga --seed 0` 这样的命令从头重跑对应配置，旧记录保留。可选的 `python suite.py diagnose --seed 0` 单独检查 MRPC 的首步更新方向。

## 实验结果与复核

`analyze --plot` 从已完成运行的逐条预测重建对照表，并根据训练日志绘制收敛曲线。`package check` 检查实验材料解压后能否重建相同的结果。

```bash
python suite.py analyze --plot
python suite.py package check
```

| 要查的内容 | 文件位置 |
|---|---|
| 表 1：五任务的论文值、本组实测值和偏差 | `analysis_output/glue-v1/table1_comparison.csv` |
| 表 9：MRPC 五种初始化的论文值和本组实测值 | `analysis_output/glue-v1/table9_mrpc.csv` |
| 逐种子指标、配对样本和收敛曲线 | 同目录下的 `per_seed.csv`、`paired_cases.csv` 和 `<任务>-<划分>.png` |
| 核对某个数字所用的配置、训练日志、耗时和逐条预测 | `runs/glue-v1/<任务>/<方法>/seed-N/attempt-NNNN/` |
| 可选的 MRPC 首步诊断 | `runs/diagnostics/mrpc-sN/diagnostic.json`，材料 ZIP 保留相同路径并记录 SHA-256 |

以核对表 1 的 MRPC 准确率为例，在 `per_seed.csv` 中按任务和方法筛选，沿 `predictions` 列给出的相对路径，在 `runs/glue-v1/` 下打开各个种子的预测文件。每个文件中 `pred` 与 `gold` 相同的样本数除以验证集样本总数，即为该种子的准确率。表 1 报告三个种子准确率的均值和样本标准差，单位分别为百分比和百分点。三个种子尚未齐全时，汇总栏显示“尚未完成”。
