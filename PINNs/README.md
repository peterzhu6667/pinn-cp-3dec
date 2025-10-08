# Physics Informed Neural Networks - 弹性力学应用

## 关于本项目

本文件夹包含PINN算法的参考实现，用于求解**圆柱孔弹性力学问题**。

原始PINN框架由 Raissi et al. (2019) 开发，本项目将其应用于土木工程领域。

---

## 项目结构

```
PINNs/
├── main/
│   ├── continuous_time_inference (Schrodinger)/
│   │   └── Schrodinger.py          # 连续时间PINN参考代码
│   └── Data/
│       ├── NLS.mat                  # .mat文件格式参考
│       └── README.md                # 数据格式说明
├── Utilities/
│   └── plotting.py                  # 可视化工具
├── IMPLEMENTATION_GUIDE.md          # 弹性力学PINN实现指南
└── README.md                        # 本文件
```

---

## 使用说明

1. **参考代码**: `continuous_time_inference (Schrodinger)/Schrodinger.py`
   - 展示连续时间PINN的完整实现
   - 适用于静态弹性力学问题（无时间演化）

2. **数据准备**: 参见 `Data/README.md`
   - 说明弹性力学问题需要的数据格式
   - 包含从3DEC导出数据的指引

3. **实现指南**: `IMPLEMENTATION_GUIDE.md`
   - 详细的代码修改步骤
   - 物理方程的TensorFlow实现

---

## 原始PINN论文引用

本项目基于以下研究工作：

**主要论文**:
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). 
  *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* 
  Journal of Computational Physics, 378, 686-707.
  [论文链接](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

**预印本**:
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). 
  *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.* 
  arXiv:1711.10561.
  [arXiv链接](https://arxiv.org/abs/1711.10561)

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). 
  *Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations.* 
  arXiv:1711.10566.
  [arXiv链接](https://arxiv.org/abs/1711.10566)

---

## BibTeX引用格式

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}
```

---

## 原始代码库

原始PINN实现（TensorFlow 1.x）: https://github.com/maziarraissi/PINNs

> **注意**: 原始代码库已不再维护。推荐使用现代框架的实现：
> - PyTorch版本: https://github.com/rezaakb/pinns-torch
> - JAX版本: https://github.com/rezaakb/pinns-jax  
> - TensorFlow 2.x版本: https://github.com/rezaakb/pinns-tf2

---

**本项目用途**: 将PINN应用于土木工程弹性力学问题的研究
