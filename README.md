# LoRA-GA 课程复现

使用 T5-base 比较 LoRA 与 LoRA-GA 在 MNLI、SST-2、CoLA、QNLI 和 MRPC 上的验证集准确率，同时复现 MRPC 初始化消融。五任务主实验使用三个种子，共 30 项；消融复用两种基线，另加 9 项。

[NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf) · [固定版本的作者实现](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef/reproduce)

本仓库按复现步骤整理课程源码快照 `df76fc9b336c8d96e2ef6969eae489acbabbae27` 和服务器实测结果。提交顺序用于说明各项工作，原始开发仓库的逐次提交未随源码包提供。
