# NURBS Meta-Atoms Transformer 模型更新工作记录

## 原文内容 (论文摘录)

> For numerical processing efficiency, we discretize the Cartesian coordinate system with grid points, where the control points of NURBS curves are mapped to binary encoded grid indices. The trained Transformer model replaces conventional numerical solvers, enabling parallel accelerated structural optimization of NURBS meta-atoms.
>
> **Computational Framework.** The training dataset was generated through rigorous finite-difference time-domain (FDTD) simulations using MEEP. We simulated 500,000 unique NURBS meta-atom geometries spanning curvature radii of 50 - 300nm and aspect ratios of 0.2 - 5.0. Each simulation solved Maxwell's equations on a 10nm mesh grid across 400 – 700 nm spectrum, extracting full-field solutions with <0.5% residual error. Geometric parameters were encoded as feature vectors containing control point coordinates, weights, and knot vectors. A Transformer-based surrogate model was implemented in PyTorch 2.0 using **12 attention heads** and **8 encoder layers** as shown in Fig. 1c. The model was trained on NVIDIA A100 GPUs with 500,000 samples (**90% training, 10% validation**), optimized via **Adam (lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98)**. Automatic differentiation enabled end-to-end optimization of the dual-output architecture predicting both complex optical response (amplitude/phase) and parametric gradients. Training convergence occurred at **10,000 epochs** with validation loss plateauing at **mean absolute error 0.0187**.

---

## 论文规格要求

| 参数 | 论文描述 |
|------|----------|
| **架构类型** | Encoder-Decoder Transformer (Fig. 1c) |
| **Attention Heads** | 12 |
| **Encoder Layers** | 8 |
| **优化器** | Adam |
| **学习率** | 5×10⁻⁵ |
| **β₁, β₂** | 0.9, 0.98 |
| **训练样本** | 500,000 |
| **训练/验证划分** | 90% / 10% |
| **训练Epochs** | 10,000 |
| **目标MAE** | 0.0187 |
| **输出** | 双输出 (optical response + parametric gradients) |
| **控制点编码** | Binary encoded grid indices |
| **光谱范围** | 400-700nm |
| **网格精度** | 10nm mesh |
| **曲率半径** | 50-300nm |
| **长宽比** | 0.2-5.0 |

---

## 已完成的工作

### 1. 创建新的匹配论文的模型文件

**文件**: `paper_matched_transformer.py` (待重命名为 `meta_transformer.py`)

包含:
- `NURBSEncoderDecoderTransformer`: 完整的Encoder-Decoder架构，匹配Fig. 1c
  - 12 attention heads
  - 8 encoder/decoder layers
  - Masked Multi-Head Attention (Decoder)
  - Binary encoded grid indices (`BinaryGridEncoder`)
  - 双输出头: `optical_head` + `gradient_head`
- `PaperMatchedNURBSModel`: 模型包装类
  - Adam优化器 (lr=5e-5, betas=(0.9, 0.98))
  - 90/10 训练验证划分
  - 10,000 epochs训练配置
  - 目标MAE 0.0187
- `NURBSMetaAtomDataset`: 数据集类
- `create_data_loaders`: 数据加载器工厂函数

### 2. 创建配套训练脚本

**文件**: `train_paper_matched_model.py`

功能:
- 合成数据生成 (用于测试)
- 加载MEEP仿真数据
- 完整训练流程
- 训练曲线可视化
- 支持快速测试模式

使用方法:
```bash
# 快速测试
python train_paper_matched_model.py --quick_test

# 完整训练 (论文规格)
python train_paper_matched_model.py --n_samples 500000 --epochs 10000

# 使用MEEP数据
python train_paper_matched_model.py --data_dir ./meep_data
```

### 3. 更新原有模型文件的默认参数

已修改的文件:
- `transformer_nurbs_model.py`: nhead 8→12, num_layers 6→8, 优化器改为Adam
- `wavelength_conditioned_transformer.py`: nhead 8→12, 优化器改为Adam
- `train_transformer_model.py`: 训练划分改为90/10

---

## 待完成的工作 (在WSL中继续)

### 1. 重命名文件
```bash
git mv paper_matched_transformer.py meta_transformer.py
```

### 2. 更新import引用
`train_paper_matched_model.py` 中:
```python
from meta_transformer import (
    PaperMatchedNURBSModel,
    NURBSMetaAtomDataset,
    create_data_loaders,
    print_paper_specs
)
```
**注意**: 这一步已在Windows中完成

### 3. 删除与论文不匹配的原模型文件
```bash
git rm transformer_nurbs_model.py
git rm wavelength_conditioned_transformer.py  
git rm train_transformer_model.py
```

### 4. 提交并推送到GitHub
```bash
git add -A
git commit -m "Refactor: Replace models with paper-matched Encoder-Decoder Transformer

- Rename paper_matched_transformer.py to meta_transformer.py
- Remove non-matching model files (transformer_nurbs_model.py, wavelength_conditioned_transformer.py, train_transformer_model.py)
- New model matches paper specifications:
  - 12 attention heads
  - 8 encoder/decoder layers
  - Adam optimizer (lr=5e-5, β₁=0.9, β₂=0.98)
  - Encoder-Decoder architecture (Fig. 1c)
  - Dual-output: optical response + parametric gradients
  - Binary encoded grid indices
  - 90/10 train/val split
  - Target MAE: 0.0187"

git push origin main
```

---

## 远程仓库信息

- **GitHub URL**: https://github.com/kaipengliuHIT/nurbs-meta-atoms
- **分支**: main (假设)

---

## 文件结构 (更新后)

```
nurbs-meta-atoms-main/
├── meta_transformer.py           # 主模型文件 (匹配论文)
├── train_paper_matched_model.py  # 训练脚本
├── nurbs_atoms_data.py           # NURBS数据处理
├── generate_training_data_parallel.py  # 并行数据生成
├── inference_transformer_model.py      # 推理脚本
├── metalens_optimization.py      # 金属透镜优化
├── example_usage.py              # 使用示例
├── visualize_field.py            # 场可视化
├── test_nurbs_simulation.py      # 测试脚本
├── README.md
├── .gitignore
├── training_data/                # 训练数据目录
└── *.npy                         # 数据文件
```

---

## 模型架构对比

### 原代码 (不匹配)
- Encoder-only Transformer
- 8 attention heads
- 6 layers
- AdamW optimizer (lr=0.001)
- 80/20 train/val split
- 单输出 (phase, transmittance)

### 新代码 (匹配论文)
- **Encoder-Decoder Transformer**
- **12 attention heads**
- **8 encoder + 8 decoder layers**
- **Adam optimizer (lr=5e-5, β₁=0.9, β₂=0.98)**
- **90/10 train/val split**
- **双输出 (optical_head + gradient_head)**
- **Binary encoded grid indices**
- **Masked Multi-Head Attention in Decoder**

---

## 注意事项

1. Windows CMD中git命令输出可能有编码问题，建议在WSL或Git Bash中操作
2. 确保GitHub已配置SSH密钥或使用HTTPS认证
3. 删除文件前确认本地有备份或确实不需要
4. 推送前检查 `git remote -v` 确认远程仓库地址正确

---

*文档创建时间: 2026-01-26*
