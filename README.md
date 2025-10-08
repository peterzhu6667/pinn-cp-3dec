# PINN在土木工程弹性力学中的应用研究

## 项目简介

本项目旨在研究**物理信息神经网络（Physics-Informed Neural Networks, PINNs）**在土木工程领域的适应性，通过将PINN算法应用于经典的**圆柱孔弹性力学问题**，对比传统数值方法（3DEC）与深度学习方法的性能。

## 项目结构

```
pinn-cp-3dec/
├── Elastic_Hole/           # 3DEC基准测试问题
│   ├── ElasticHole.fis     # 解析解计算脚本
│   ├── ElasticHole-*.dat   # 3DEC模型文件（不同网格类型）
│   └── project.md          # 问题描述和理论解
│
├── PINNs/                  # PINN算法实现
│   ├── main/
│   │   └── continuous_time_inference (Schrodinger)/
│   │       └── Schrodinger.py  # 连续时间PINN参考代码
│   ├── Utilities/
│   │   └── plotting.py     # 可视化工具
│   └── IMPLEMENTATION_GUIDE.md  # 实现指南
│
├── papers.md               # PINN核心理论（精简版）
└── .gitignore              # Git配置
```

## 研究问题

### 圆柱孔弹性力学问题

**物理背景**：
- 无限弹性介质中的1.0m半径圆柱孔
- 非均匀应力场：垂直应力 30 MPa，水平应力 15 MPa
- 材料参数：
  - 剪切模量 G = 2.8 GPa
  - 体积模量 K = 3.9 GPa
  - 泊松比 ν ≈ 0.167

**控制方程**（线弹性平衡方程）：
```
∇·σ = 0                    (平衡方程)
σ = C : ε                  (本构方程)
ε = ½(∇u + ∇u^T)          (几何方程)
```

**边界条件**：
- 孔边界：自由表面（σ·n = 0）
- 远场边界：应力 σ_∞ = diag(15, 30, 15) MPa

**解析解**：Kirsch解（详见 `Elastic_Hole/project.md`）

## PINN方法

### 连续时间模型（适用于静态弹性问题）

本项目采用**连续时间PINN**求解静态弹性力学问题：

1. **神经网络逼近**：u(x,y,z) ≈ NN(x,y,z; θ)
2. **物理损失**：L_physics = ||∇·σ(u)||²（平衡方程残差）
3. **边界损失**：L_boundary = ||BC违反项||²
4. **总损失**：L = L_physics + λ_b · L_boundary

### 实现参考

- 参考代码：`PINNs/main/continuous_time_inference (Schrodinger)/Schrodinger.py`
- 实现指南：`PINNs/IMPLEMENTATION_GUIDE.md`
- 理论基础：`papers.md`（论文精简版）

## 使用方法

### 环境配置

```bash
# 安装依赖
pip install tensorflow numpy scipy matplotlib
```

### 运行3DEC基准测试

使用3DEC软件运行：
```
3DEC > call ElasticHole-Hex.dat
```

### 实现PINN模型

按照 `PINNs/IMPLEMENTATION_GUIDE.md` 的指引实现弹性力学PINN求解器。

## 研究目标

1. **精度对比**：PINN vs 3DEC vs 解析解
2. **计算效率**：训练时间 vs 3DEC求解时间
3. **适应性分析**：
   - 网格独立性（PINN无需网格）
   - 边界条件处理能力
   - 复杂几何的适应性
4. **工程应用潜力**：在岩土工程中的可行性评估

## 研究意义

评估深度学习方法在土木工程数值模拟中的可行性，为复杂岩土工程问题提供新的求解思路。

## 参考文献

- Kirsch, G. (1898). Die Theorie der Elastizität.
- Jaeger, J.C., Cook, N.G.W. (1976). Fundamentals of Rock Mechanics.
- Raissi, M., et al. (2019). Physics-informed neural networks. *J. Comput. Phys.*

---

**项目作者**：Bo Sihang  
**研究目的**：学术研究
