# LoRA-GA 课程复现 · 小组交接

**39/39 项正式实验已完成，接下来可以直接分工做结果分析、制图和报告撰写。** 接受 GitHub 邀请后，打开 `reproduction/glue-v1` 分支。先读[结果总览](../results/glue-v1/README.md)，再按自己负责的问题打开下方材料。分析和写作都无需登录服务器或申请 GPU。

## 已完成的工作

- 固定了作者代码、T5-base 和五任务 GLUE 数据版本，适配了 LoRA、LoRA-GA 训练与评估，核验了 5 份模型相关文件和 11 份数据文件。[方法与运行说明](../README.md)
- 在 RTX 3090 上完成五任务 × 两种方法 × 三个种子的 30 项主实验，以及 MRPC 的 9 项额外初始化消融。MRPC 的普通 LoRA 和完整 LoRA-GA 复用主实验结果。每项实验在一张 GPU 上运行，使用 FP32，关闭 TF32，训练与验证 batch 均为 32；LoRA-GA 用 8 条校准样本逐条求梯度后等权平均。[实验配置与日志](../results/glue-v1/evidence/)
- 从最终步逐条预测重新算出了所有准确率。SST-2 / LoRA-GA / seed 2 在修正 FP32 初始化检查后，于第三次尝试跑完；该次训练和验证设置与另外两个种子相同。[结果复核与修正记录](../results/glue-v1/README.md)

## 剩余任务怎么分

可以在群里按下表认领，每块由一人牵头，人数少时可兼任。每位同学把自己负责的**正文、表或图、数据来源**一并交给统稿人。

| 可认领的部分 | 需要回答的问题 | 建议交付 |
|---|---|---|
| 方法与实验设置 | LoRA-GA 的梯度初始化如何区别于普通 LoRA？五任务、三个种子和单卡参数怎样保证比较口径一致？ | 一段方法说明和一张设置表。[作者实现](../reference/) · [本组适配记录](复现实验记录.md) |
| 主实验分析 | 哪些任务提升最大？哪些与论文不同？同一种子下的变化是否一致？ | 一张五任务论文对照表，一段结合逐种子差值的分析。[表 1 对照](../results/glue-v1/table1_comparison.csv) · [配对差值](../results/glue-v1/paired.csv) |
| MRPC 消融与误差分析 | 单独加尺度调整、单独加梯度方向、两者结合分别怎样？为什么两个基线恰好都是 68.38%？ | 一张消融表，加一个从混淆矩阵或预测中核实的例子。[表 9 对照](../results/glue-v1/table9_mrpc.csv) · [逐种子指标](../results/glue-v1/per_seed.csv) |
| 图表、讨论与统稿 | 曲线能支持什么收敛结论？哪些偏差还有多种可能解释？ | 选取 1–2 张能支持正文的图，写清坐标和种子口径，合并各节并核对数字及引用。[CoLA 曲线](../results/glue-v1/cola-validation.png) · [MRPC 曲线](../results/glue-v1/mrpc-validation.png) |

## 写报告时统一的口径

- 主指标是验证集 **Accuracy（%）**，两种方法之差用**百分点**。主表取完整一轮的最后一步；MNLI 用 matched，mismatched 单列。每种方法报告三个种子的均值，需要描述波动时使用样本标准差。五任务平均给每个任务相同权重。
- 五任务平均准确率为 LoRA **82.02%**、LoRA-GA **87.30%**，平均提升 **5.28 个百分点**，按论文印刷均值计算的增益为 **5.69 个百分点**。CoLA 和 MRPC 分别提升 10.99 和 15.03 个百分点，SST-2 下降 0.27 个百分点。正文应同时交代一致处和偏差。[论文表 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf#page=7)
- MRPC 上，普通 LoRA 和 Gaussian 的三个种子都把 408 条样本全部预测为“语义等价”，其中 279 条正确，因此都得到 68.38%。完整 LoRA-GA 仍排第一，但仅尺度调整、仅梯度方向分别比论文高 7.03 和 9.73 个百分点。现有数据可以描述这种差异；原因若未经对照实验验证，请写成可能解释。[MRPC 证据包](../results/glue-v1/evidence/mrpc/) · [论文表 9](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf#page=18)
- 旧适配记录里的 `0/39` 是运行前状态。正式数字以 `results/glue-v1/` 为准。若报告要展示误判句子，可用预测 CSV 的 `idx` 在同任务、同验证划分的原始数据中查找。[数据来源](data/数据来源.md)

在本地核对数字时，进入仓库根目录运行 `python results/glue-v1/verify_results.py`。它只用 Python 标准库，成功时会显示 `39/39 verified`。
