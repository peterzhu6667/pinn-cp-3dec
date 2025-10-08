# 快速开始指南

## 📋 完整操作流程

本指南将引导你完成从3DEC数据导出到PINN训练验证的全过程。

---

## 第一步：准备3DEC数据

### 1.1 运行3DEC模型

```bash
# 1. 启动3DEC软件
3DEC

# 2. 在3DEC命令行中运行模型
3DEC > call ElasticHole-Hex.dat
3DEC > model cycle 2000

# 3. 等待模拟完成，确保收敛
```

### 1.2 导出3DEC数据

**方法一：自动生成FISH脚本（推荐）**

```bash
# 1. 打开命令行，进入项目目录
cd D:\bo4syhang\pinn-cp-3dec\Elastic_Hole

# 2. 运行导出脚本（第一次运行）
python export_to_pinn.py
```

**输出**：
```
==============================================================
3DEC数据导出工具 - PINN训练数据准备
==============================================================

已创建3DEC导出脚本: D:\bo4syhang\pinn-cp-3dec\Elastic_Hole\export_3dec_data.fis
请在3DEC中运行此脚本以导出数据

==============================================================
使用步骤:
==============================================================
1. 在3DEC中打开模型并运行到收敛
2. 在3DEC中执行: call D:\bo4syhang\pinn-cp-3dec\Elastic_Hole\export_3dec_data.fis
3. 确认生成了以下文本文件:
   - nodes.txt
   - displacement.txt
   - stress.txt
   - boundary_hole.txt
   - boundary_far.txt
4. 再次运行此Python脚本完成.mat文件生成
==============================================================

⚠ 尚未检测到3DEC导出文件
   请先在3DEC中运行导出脚本
```

**3. 在3DEC中运行导出脚本**

```bash
# 在3DEC命令行中
3DEC > call export_3dec_data.fis
```

**输出**：
```
数据导出完成!
```

**4. 检查生成的文本文件**

在 `Elastic_Hole` 文件夹中应该看到：
- ✅ `nodes.txt` - 节点坐标
- ✅ `displacement.txt` - 位移数据
- ✅ `stress.txt` - 应力数据  
- ✅ `boundary_hole.txt` - 孔边界节点
- ✅ `boundary_far.txt` - 远场边界节点

**5. 转换为.mat格式**

```bash
# 再次运行Python脚本
python export_to_pinn.py
```

**输出**：
```
检测到3DEC导出文件，开始转换...
读取 XXXX 个节点坐标
读取 XXXX 个位移数据
读取 XXXX 个应力数据
读取 XXX 个孔边界点
读取 XXX 个远场边界点

数据已导出到: D:\bo4syhang\pinn-cp-3dec\PINNs\main\Data\elastic_hole_3dec.mat
文件大小: XX.XX KB

✓ 转换完成！
```

---

## 第二步：生成解析解数据

```bash
cd D:\bo4syhang\pinn-cp-3dec\Elastic_Hole
python generate_analytical_solution.py
```

**输出**：
```
======================================================================
Kirsch解析解生成器 - 圆柱孔弹性力学问题
======================================================================
材料参数:
  剪切模量 G = 2.80 GPa
  体积模量 K = 3.90 GPa
  泊松比 ν = 0.1667

边界条件:
  孔半径 a = 1.00 m
  水平应力 σ_x = 30.0 MPa
  垂直应力 σ_y = 15.0 MPa

生成解析解网格:
  径向: 1.00 - 5.00 m, 50 点
  环向: 0 - 2π, 100 点
  总点数: 5000

======================================================================
✓ 解析解数据已导出到: D:\bo4syhang\pinn-cp-3dec\PINNs\main\Data\elastic_hole_analytical.mat
  文件大小: XXX.XX KB
======================================================================

可视化图片已保存: D:\bo4syhang\pinn-cp-3dec\Elastic_Hole\analytical_solution.png

数据字典包含的键:
  far_field_stress_x       : shape (1,)
  far_field_stress_y       : shape (1,)
  hole_radius              : shape (1,)
  nr                       : shape (1,)
  ntheta                   : shape (1,)
  poissons_ratio           : shape (1,)
  r                        : shape (5000,)
  r_max                    : shape (1,)
  r_min                    : shape (1,)
  shear_modulus            : shape (1,)
  sigma_r                  : shape (5000,)
  sigma_theta              : shape (5000,)
  sigma_xx                 : shape (5000,)
  sigma_xy                 : shape (5000,)
  sigma_xz                 : shape (5000,)
  sigma_yy                 : shape (5000,)
  sigma_yz                 : shape (5000,)
  sigma_zz                 : shape (5000,)
  theta                    : shape (5000,)
  u_r                      : shape (5000,)
  u_theta                  : shape (5000,)
  u_x                      : shape (5000,)
  u_y                      : shape (5000,)
  u_z                      : shape (5000,)
  x                        : shape (5000,)
  y                        : shape (5000,)
  z                        : shape (5000,)
```

---

## 第三步：验证数据

检查数据文件是否正确生成：

```bash
cd D:\bo4syhang\pinn-cp-3dec\PINNs\main\Data
dir
```

应该看到：
- ✅ `elastic_hole_3dec.mat` - 3DEC训练数据
- ✅ `elastic_hole_analytical.mat` - Kirsch解析解
- ✅ `NLS.mat` - 格式参考

**验证数据内容**（可选）：

```python
import scipy.io as sio

# 加载3DEC数据
data_3dec = sio.loadmat('elastic_hole_3dec.mat')
print("3DEC数据包含的键:")
for key in data_3dec.keys():
    if not key.startswith('__'):
        print(f"  {key}")

# 加载解析解数据
data_analytical = sio.loadmat('elastic_hole_analytical.mat')
print("\n解析解数据包含的键:")
for key in data_analytical.keys():
    if not key.startswith('__'):
        print(f"  {key}")
```

---

## 第四步：实现PINN模型

现在你可以开始实现PINN模型了。

### 4.1 参考代码

参考 `PINNs/main/continuous_time_inference (Schrodinger)/Schrodinger.py`

### 4.2 实现指南

详见 `PINNs/IMPLEMENTATION_GUIDE.md`

### 4.3 创建PINN训练脚本

建议创建：`PINNs/main/ElasticHole_PINN.py`

**基本结构**：

```python
import tensorflow as tf
import numpy as np
import scipy.io as sio

# 1. 加载3DEC训练数据
train_data = sio.loadmat('Data/elastic_hole_3dec.mat')
X_train = np.stack([train_data['x'].flatten(), 
                    train_data['y'].flatten(), 
                    train_data['z'].flatten()], axis=1)
U_train = train_data['u_x'].flatten()  # 位移

# 2. 定义PINN模型
class ElasticPINN:
    def __init__(self, layers):
        # 神经网络定义
        pass
    
    def net_u(self, x, y, z):
        # 预测位移
        pass
    
    def net_f(self, x, y, z):
        # 计算物理残差 (平衡方程)
        pass
    
    def loss(self, X_data, U_data, X_physics):
        # 数据损失 + 物理损失
        loss_data = MSE(U_pred - U_data)
        loss_physics = MSE(equilibrium_residual)
        return loss_data + lambda_physics * loss_physics

# 3. 训练
model = ElasticPINN(layers=[3, 50, 50, 50, 3])
model.train(epochs=10000)

# 4. 预测和验证
analytical = sio.loadmat('Data/elastic_hole_analytical.mat')
U_pred = model.predict(analytical['x'], analytical['y'], analytical['z'])
error = np.linalg.norm(U_pred - analytical['u_x']) / np.linalg.norm(analytical['u_x'])
print(f"L2相对误差: {error:.4f}")
```

---

## 常见问题

### Q1: 3DEC导出的文本文件在哪里？
**A**: 在 `Elastic_Hole` 文件夹中，与 `.dat` 文件同级。

### Q2: 如果3DEC模型没有收敛怎么办？
**A**: 增加迭代步数：`3DEC > model cycle 5000`

### Q3: 生成的.mat文件太大怎么办？
**A**: 可以在FISH脚本中采样部分节点，不需要导出所有节点。

### Q4: 如何检查解析解是否正确？
**A**: 查看生成的 `analytical_solution.png` 图片，检查应力和位移分布是否合理。

---

## 数据文件说明

详见：`PINNs/main/Data/README.md`

---

## 下一步

数据准备完成后，开始实现PINN模型：

1. 阅读 `PINNs/IMPLEMENTATION_GUIDE.md`
2. 参考 `PINNs/main/continuous_time_inference (Schrodinger)/Schrodinger.py`
3. 创建 `PINNs/main/ElasticHole_PINN.py`
4. 训练、验证、对比

---

**祝顺利！** 🚀
