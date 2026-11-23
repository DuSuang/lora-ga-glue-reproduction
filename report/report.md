# LoRA-GA 复现报告：T5-base 五任务对比与 MRPC 初始化消融

> 报告草稿：正文已补齐，交付材料及组内贡献待完成。

## 1. 复现目标与范围

全参数微调需要更新模型的全部参数，训练时还要保存对应的梯度和优化器状态，资源开销较大。LoRA 冻结预训练权重，只训练两个低秩矩阵来表示权重增量，从而减少可训练参数及相关存储开销。但低秩参数化也改变了权重的更新方式，在相同训练预算下，效果可能落后于全参数微调。

LoRA-GA 从初始化入手改善这一问题。普通 LoRA 随机初始化一个矩阵、将另一个置零；LoRA-GA 则先用少量训练样本估计梯度，再利用梯度方向和尺度调整初始化两个矩阵，使训练初期的权重更新更接近全参数微调的梯度方向。本文关注这种初始化能否提高有限训练预算下的效果，以及梯度方向和尺度调整分别起到什么作用。

本次主实验对应论文表 1，比较 T5-base 在 MNLI、SST-2、CoLA、QNLI、MRPC 上的 LoRA 与 LoRA-GA，每种配置运行三个种子，共 30 次训练。消融实验对应附录表 9 的 MRPC 列，比较五种初始化；复用主实验中的 LoRA 和完整 LoRA-GA 结果后，额外增加 9 次训练，合计 39 次。本次以普通 LoRA 为同条件基线，未安排全参数微调实验。

除最终准确率外，我们还分析 MRPC 的逐条预测，并绘制 CoLA、MRPC 的训练曲线，观察不同初始化如何影响分类结果和训练过程。

## 2. 方法与本组实现

LoRA 将有效权重写为 `W_eff = C + sBA`。其中，基座权重 `C` 冻结，只训练低秩矩阵 `A`、`B`，`s` 为缩放系数。普通 LoRA 随机初始化 `A`，将 `B` 置零，因此模型从原预训练权重开始训练。

LoRA-GA 则先用训练数据计算梯度，再据此初始化两个矩阵。本组使用训练集原始顺序中的前 8 条样本，逐条计算损失对目标层权重的梯度并取平均。对各层梯度做低秩 SVD 后，用前 8 个右奇异向量构造 `A₀`，第 9—16 个左奇异向量构造 `B₀`，再调整两者的尺度。

由于 `B₀A₀` 非零，安装适配器时将基座权重改为 `C = W₀ − sB₀A₀`，使初始有效权重仍等于预训练权重 `W₀`。梯度估计只在初始化时执行，正式训练仍然只更新 `A`、`B`。

代码基于官方 LoRA-GA 实现改写。本组完成了五任务训练流程、MNLI 三分类评测和五种初始化的统一实现，并保存逐条预测用于复核结果。变长标签的损失计算也与作者代码进行了对照，保证微批划分前后使用相同的训练目标。

五种初始化的区别如下：

| Method | Initialization | Scale adjustment |
|---|---|---|
| LoRA | Random A, zero B | No |
| Gaussian | Random A and B | No |
| +SO | Random A and B | Yes |
| +GA | Gradient singular vectors | No |
| LoRA-GA | Gradient singular vectors | Yes |

Gaussian 在非方形矩阵上的方差按论文表 3 设置，与官方代码该分支的维度对应不同。具体公式和实现差异见仓库的复现实验记录。

## 3. 实验设置

实验使用 T5-base，每个任务训练一轮，种子为 0、1、2。同一任务、同一种子下，各方法使用相同的训练样本顺序、更新步数和验证时点。

| Task | Training samples | Validation samples | Training steps |
|---|---:|---:|---:|
| MNLI | 392,702 | 9,815 matched | 12,272 |
| SST-2 | 67,349 | 872 | 2,105 |
| CoLA | 8,551 | 1,043 | 268 |
| QNLI | 104,743 | 5,463 | 3,274 |
| MRPC | 3,668 | 408 | 115 |

适配器安装在注意力层的 q、k、v、o 和前馈层的 wi、wo，共训练 3,244,032 个参数。主要配置如下：

| Setting | Value |
|---|---|
| Rank / Alpha | 8 / 16 |
| Training / Evaluation batch size | 32 / 32 |
| Optimizer / Learning rate | AdamW / 1e−4 |
| Learning-rate schedule | Cosine, 3% warmup |
| Weight decay / Gradient clipping | 0 / 1 |
| Maximum sequence length | 128 |
| Calibration samples / Gamma | 8 / 16 |
| Precision | FP32, TF32 disabled |

正式实验在 RTX 3090 上运行，使用 Python 3.11.18、PyTorch 2.3.0+cu121 和 Transformers 4.40.2。

为与论文对照，五项任务统一报告验证集 Accuracy，包括 CoLA。评测时比较解码首位置各候选标签的分数；MNLI 使用三个候选类别，matched 划分计入主表，mismatched 结果另行保存。各项结果取三个种子的均值和样本标准差，五任务平均采用等权计算。

本组报告完整一轮训练结束时的结果。官方脚本按验证损失选择检查点，两者的取值方式有所不同。

## 4. 实验结果与分析

### 4.1 五任务主实验

**Table 1. GLUE accuracy (%) on T5-base.**

<!-- main-table:start -->

| Method | MNLI-m | SST-2 | CoLA | QNLI | MRPC | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| LoRA (Paper) | 85.30 ± 0.04 | 94.04 ± 0.11 | 69.35 ± 0.05 | 92.96 ± 0.09 | 68.38 ± 0.01 | 82.08* |
| LoRA-GA (Paper) | 85.70 ± 0.09 | 94.11 ± 0.18 | 80.57 ± 0.20 | 93.18 ± 0.06 | 85.29 ± 0.24 | 87.77 |
| LoRA (Ours) | 85.41 ± 0.05 | 94.07 ± 0.26 | 69.35 ± 0.06 | 92.87 ± 0.18 | 68.38 ± 0.00 | 82.02 |
| LoRA-GA (Ours) | 85.81 ± 0.13 | 93.81 ± 0.40 | 80.35 ± 0.00 | 93.14 ± 0.28 | 83.42 ± 1.35 | 87.30 |

<!-- main-table:end -->

复现结果为三个种子的均值 ± 样本标准差，Avg. 为五任务等权平均；论文结果引自原文表 1 [1]。

\* 论文 LoRA 的五项展示值重算平均为 82.006%，本表保留其印刷值 82.08%。

表 1 对比了论文与本组的五任务结果。LoRA-GA 的平均准确率为 87.30%，比普通 LoRA 的 82.02% 提高了 5.28 个百分点，与论文报告的 5.69 个百分点接近，复现了 LoRA-GA 在这一训练预算下优于普通 LoRA 的整体结果。

不同任务的收益差异明显。CoLA 和 MRPC 分别提高了 10.99 和 15.03 个百分点，是平均分提升的主要来源，这与论文中的结果分布一致。两项任务在三个种子下都获得了较大提升：CoLA 的增益为 10.93–11.03 个百分点，MRPC 为 13.48–15.93 个百分点。相比之下，MNLI 和 QNLI 仅提高了 0.40 和 0.26 个百分点；SST-2 平均下降了 0.27 个百分点，三个种子中有两个下降、一个上升。因此，本次实验中 GA 的优势主要体现在 CoLA 和 MRPC，其他任务的变化较小。

与论文数值相比，本组普通 LoRA 的五项结果均相差不到 0.12 个百分点。LoRA-GA 的主要偏差出现在 MRPC：本组准确率为 83.42%，低于论文的 85.29%；其余四项偏差均不到 0.31 个百分点。MRPC 上 GA 相对 LoRA 的提升由论文的 16.91 个百分点降至本组的 15.03 个百分点，仍是五个任务中收益最大的一项。

### 4.2 MRPC 初始化消融

**Table 2. Initialization ablation on MRPC. Accuracy (%).**

<!-- mrpc-table:start -->

| Method | Paper | Ours |
|---|---:|---:|
| LoRA | 68.38 ± 0.01 | 68.38 ± 0.00 |
| Gaussian | 68.38 ± 0.00 | 68.38 ± 0.00 |
| +SO | 67.97 ± 0.75 | 75.00 ± 1.49 |
| +GA | 66.09 ± 11.32 | 75.82 ± 1.39 |
| LoRA-GA | 85.29 ± 0.24 | 83.42 ± 1.35 |

<!-- mrpc-table:end -->

SO 表示尺度调整，GA 表示梯度方向初始化。复现结果为三个种子的均值 ± 样本标准差，论文结果引自附录表 9 [1]。

表 2 比较了五种初始化在 MRPC 上的表现。普通 LoRA 和 Gaussian 的准确率均为 68.38%，非零随机初始化没有带来收益。仅尺度调整和仅梯度方向分别达到 75.00% 和 75.82%，比普通 LoRA 提高 6.62 和 7.43 个百分点。完整 LoRA-GA 达到 83.42%，又分别高出两个单组件版本 8.42 和 7.60 个百分点，并在三个种子下均取得最高准确率。这组结果说明，梯度方向和尺度调整各自都能改善本次训练的效果，两者结合进一步提高了准确率。

GA 的提升主要来自对不等价句对的识别。MRPC 验证集包含 279 条语义等价句对和 129 条不等价句对。普通 LoRA 和 Gaussian 在三个种子下都将全部样本预测为“等价”，只答对了前面的 279 条，准确率因此为 68.38%。完整 GA 则平均能识别出 66.93% 的不等价句对，对等价句对的正确识别率为 91.04%。虽然它也将部分等价句对判错，但识别出了更多原本被全部错判的不等价句对，最终提高了整体准确率。

论文中同样是完整 LoRA-GA 得分最高，但单独使用两个组件时的结果与本组不同。论文的 +SO 和 +GA 分别为 67.97% 和 66.09%，均低于普通 LoRA 的 68.38%；本组则分别达到 75.00% 和 75.82%，都获得了提升。因此，我们复现了两种组件结合后效果最好的结果，同时也观察到：在本次实验设置下，单独调整尺度或使用梯度方向，就已经能改善普通 LoRA 的表现。

### 4.3 训练过程与收敛表现

![CoLA training curves](figures/cola-training.png)

**Figure 1. Training loss and validation accuracy on CoLA.** Lines show the mean across three seeds; shaded regions show ±1 sample standard deviation. Loss uses a logarithmic scale without smoothing. Markers indicate recorded validation evaluations.

在 CoLA 上，LoRA-GA 的优势在训练前期就已出现。第 54 步时，其验证准确率达到 72.64%，已经超过普通 LoRA 训练结束时的 69.35%。此后，LoRA-GA 继续提升至 80.35%，普通 LoRA 则逐渐停留在 69% 左右。结合 loss 曲线可以看到，GA 更快降低了训练损失，并且较早取得的准确率优势一直保持到训练结束。

![MRPC training curves](figures/mrpc-training.png)

**Figure 2. Training loss and validation accuracy for five initializations on MRPC.** Lines show the mean across three seeds; shaded regions show ±1 sample standard deviation. Loss uses a logarithmic scale without smoothing. Markers indicate recorded validation evaluations. LoRA and Gaussian accuracy curves overlap from step 35 onward.

MRPC 更明显地表现出训练 loss 与分类准确率的区别。普通 LoRA 和 Gaussian 的 loss 持续下降，但从第 35 步起，两者在各次评估中的准确率都停在 68.38%。结合上一节的最终预测结果，这说明训练损失的降低，并没有让它们在本轮训练结束时摆脱只预测多数类的情况。完整 LoRA-GA 则在第 81 步达到 83.42%，之后的平均准确率保持在 83% 左右。

两个单组件版本虽然最终都优于普通 LoRA，但训练中途出现了明显波动：仅梯度方向版本在第 35 步的平均准确率降至 42.57%，仅尺度调整版本在第 46 步降至 56.70%，之后才逐步回升。完整 GA 在早期也有波动，但从第 35 步起，在后续各次评估中均高于两个单组件版本。这补充了最终消融表没有展示的过程差异：在本次训练中，方向与尺度结合不仅取得了更高的最终准确率，也更早取得了持续领先的表现。

## 5. 结论与局限

本次复现中，LoRA-GA 在 T5-base 五项任务上的平均准确率比普通 LoRA 提高了 5.28 个百分点，主要收益来自 CoLA 和 MRPC。训练曲线显示，在这两项任务上，GA 较早取得了更高的准确率，并将优势保持到训练结束。这说明，在一轮训练的预算下，初始化方式对模型最终能学到多少有明显影响。不过，MNLI、QNLI 的提升较小，SST-2 略有下降，收益大小仍取决于具体任务。

MRPC 消融实验中，完整 LoRA-GA 在三个种子下均取得最高准确率，复现了梯度方向与尺度调整结合使用效果最好的结果。与论文不同的是，本组两个单组件版本也都获得了提升。逐条预测则进一步解释了完整 GA 的收益：它提高了对不等价句对的识别能力，使模型摆脱了普通 LoRA 在训练结束时只预测多数类的情况。

本次复现集中在 T5-base 的五项 GLUE 任务，比较的是一轮训练内的表现。这留下了一个值得进一步研究的问题：延长训练后，普通 LoRA 会逐渐追上，还是 GA 仍能保持优势？更长的训练可以帮助区分，初始化带来的收益主要是更快学会任务，还是充分训练后仍有性能提升。此外，加入全参数微调作为参照，可以进一步比较两种低秩方法与全参训练之间的差距。

## 6. 复现与交付说明

当前工作分支为 `reproduction/analysis`。结果位于 `results/glue-v1/`，其中 39 份实验记录包含配置、训练与验证日志、最终逐条预测。

在仓库根目录运行以下命令，可重新计算全部实验结果及报告中的两张表，无需 GPU：

```bash
python3 results/glue-v1/verify_results.py
python3 report/analyze_main.py
python3 report/analyze_mrpc.py
```

全部实验已通过复核，输出为 `39/39 verified`。

绘图使用单独的环境，避免改变训练依赖：

```bash
python -m pip install -r report/plot-requirements.txt
python report/plot_training.py --output report/figures
```

重新训练时，按 README 安装 Python 3.11 环境和 GPU 依赖，再准备模型与数据。以下命令运行 MRPC 的三步最小样例：

```bash
python suite.py prepare
python suite.py run --smoke --task mrpc --method ga --seed 0 \
  --device cuda:0 --microbatch 32 --eval-batch 32
python suite.py list --smoke --task mrpc --method ga --seed 0
```

去掉 `--smoke` 即运行该配置的正式实验。模型和数据版本、下载地址及校验值已固定在代码中。

最终提交包包括源代码、实验数据与日志、环境和运行说明，以及报告和配图。打包时还需从实验服务器补入原始数据和完整运行记录，并将新增的分析脚本、绘图脚本和报告图片加入打包清单。

## 参考文献

[1] Wang, S., Yu, L., & Li, J. LoRA-GA: Low-Rank Adaptation with Gradient Approximation. NeurIPS, 2024. [论文全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf).

## 组内贡献说明（Contribution）

待各成员核实实际职责和产出后填写，并附对应代码路径、commit 或实验编号。
