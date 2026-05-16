# 2026.06-HW2-3-23301020080
计算机视觉Homework2任务三
# 图像分割模型的像素级训练：从零搭建 U-Net 与损失函数工程

本仓库包含了“从零搭建与损失函数工程”实验的完整代码。本项目基于 PyTorch 框架，从零开始（无预训练权重）手写搭建了经典的语义分割网络 **U-Net**，并在轻量级三分类数据集 **Oxford-IIIT Pet Dataset** 上进行了训练与评估。

本实验的重点在于**对比不同损失函数对处理像素极度不平衡问题的影响**，分别实现了标准交叉熵损失（CE Loss）、手动编写的 Dice Loss 以及两者的组合损失，并通过 WandB 进行了完整的训练可视化。


##  环境依赖与配置

本项目代码已在 Python 3.9 环境下测试通过。为了避免 Windows 下多线程 DataLoader 导致的死锁问题，代码默认使用 `num_workers=0` 且针对 CPU 训练进行了尺寸优化（128x128）。

请使用以下命令安装必要的依赖库：

```bash
# 推荐使用 conda 或 venv 创建虚拟环境
pip install torch torchvision
pip install wandb tqdm matplotlib numpy
```

##  项目文件

unet.py: U-Net 模型架构的核心实现文件（包含下采样、上采样与 Skip Connection）。

train_ce.py: 仅使用标准 Cross-Entropy Loss 进行训练的执行脚本。

train_dice.py: 仅使用手动实现的 Dice Loss 进行训练的执行脚本。

train_combined.py: 使用 组合损失 (CE + Dice) 进行训练的执行脚本。

data/: 运行代码后自动下载的数据集存放目录（首次运行自动生成）。


## 运行指南
由于数据集会自动下载（如果本地没有的话），你只需要按顺序直接运行以下三个训练脚本即可。每个脚本默认运行 15 个 epoch。

1. 运行交叉熵损失实验：

```Bash
python train_ce.py
```
2. 运行纯 Dice Loss 实验：

```Bash
python train_dice.py
```
3. 运行组合损失实验：

```Bash
python train_combined.py
```
