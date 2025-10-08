# PINN数据文件说明

## 文件夹用途

此文件夹用于存放PINN训练和验证所需的数据文件。

## 数据格式参考

### 现有文件
- `NLS.mat` - 非线性Schrödinger方程的数据示例（**仅作为.mat文件格式参考**）

### .mat文件格式说明

MATLAB `.mat` 文件是一种二进制格式，可以使用Python的`scipy.io.loadmat()`读取。

**典型结构**：
```python
import scipy.io as sio
data = sio.loadmat('filename.mat')
# data是一个字典，包含各种numpy数组
```

**NLS.mat 示例结构**：
```python
{
    't': array of time points,           # 时间坐标
    'x': array of spatial points,        # 空间坐标  
    'usol': solution array (Nx x Nt)     # 解的数组（空间×时间）
}
```

---

## 弹性力学问题所需数据

### ⚠️ 重要说明：工作流程

**本项目采用数据驱动 + 物理约束的PINN方法**：

```
1. 3DEC数值模拟
   ↓ 生成数值解（位移、应力）
   
2. 使用3DEC数据训练PINN
   ├── 数据损失: ||u_PINN - u_3DEC||²
   └── 物理损失: ||∇·σ||²（平衡方程残差）
   
3. PINN预测
   ↓ 在新的点上预测位移和应力
   
4. 三方对比验证
   ├── PINN预测结果
   ├── 3DEC数值解
   └── Kirsch解析解（理论真值）
   
5. 评估PINN在土木工程中的适应性
   ├── 精度对比
   ├── 计算效率
   └── 泛化能力
```

---

### 从Elastic_Hole项目导出的数据

#### 1. **训练数据（来自3DEC数值模拟）**

```python
{
    # 节点坐标
    'x_train': 3DEC节点x坐标 (N_train,),
    'y_train': 3DEC节点y坐标 (N_train,),
    'z_train': 3DEC节点z坐标 (N_train,),
    
    # 3DEC计算的位移（训练标签）
    'u_x_train': x方向位移 (N_train,),
    'u_y_train': y方向位移 (N_train,),
    'u_z_train': z方向位移 (N_train,),
    
    # 3DEC计算的应力（训练标签）
    'sigma_xx_train': xx应力 (N_train,),
    'sigma_yy_train': yy应力 (N_train,),
    'sigma_zz_train': zz应力 (N_train,),
    'sigma_xy_train': xy剪应力 (N_train,),
    'sigma_xz_train': xz剪应力 (N_train,),
    'sigma_yz_train': yz剪应力 (N_train,),
    
    # 边界信息（用于物理损失）
    'hole_boundary_indices': 孔边界节点索引,
    'far_field_boundary_indices': 远场边界节点索引,
    
    # 材料参数
    'shear_modulus': 2.8e9,
    'bulk_modulus': 3.9e9,
    'poissons_ratio': 0.167,
    'hole_radius': 1.0,
    'far_field_stress_x': 15e6,
    'far_field_stress_y': 30e6
}
```

**说明**：
- 可以使用**全部3DEC节点**作为训练数据
- 也可以**采样部分节点**来测试PINN的泛化能力
- 物理损失确保PINN学习的解满足平衡方程

#### 2. **验证数据（用于三方对比）**

##### a) Kirsch解析解（理论真值）
```python
{
    # 验证点坐标（与3DEC/PINN使用相同的点）
    'x_test': x坐标 (N_test,),
    'y_test': y坐标 (N_test,),
    'z_test': z坐标 (N_test,),
    
    # Kirsch解析解
    'u_x_analytical': x位移解析解 (N_test,),
    'u_y_analytical': y位移解析解 (N_test,),
    'u_z_analytical': z位移解析解 (N_test,),
    'sigma_xx_analytical': xx应力解析解 (N_test,),
    'sigma_yy_analytical': yy应力解析解 (N_test,),
    'sigma_zz_analytical': zz应力解析解 (N_test,),
    'sigma_xy_analytical': xy剪应力解析解 (N_test,),
    'sigma_xz_analytical': xz剪应力解析解 (N_test,),
    'sigma_yz_analytical': yz剪应力解析解 (N_test,)
}
```

##### b) 3DEC完整结果（数值解基准）
```python
{
    # 所有3DEC节点的完整结果
    'x_3dec': 所有节点x坐标 (N_3dec,),
    'y_3dec': 所有节点y坐标 (N_3dec,),
    'z_3dec': 所有节点z坐标 (N_3dec,),
    
    # 3DEC完整结果
    'u_x_3dec': 完整x位移 (N_3dec,),
    'u_y_3dec': 完整y位移 (N_3dec,),
    'u_z_3dec': 完整z位移 (N_3dec,),
    'sigma_xx_3dec': 完整应力 (N_3dec,),
    # ... 等等
}
```

---

## 推荐的数据文件组织

```
Data/
├── elastic_hole_3dec_train.mat    # 3DEC结果（训练用）
├── elastic_hole_3dec_full.mat     # 3DEC完整结果（验证对比）
└── elastic_hole_analytical.mat    # Kirsch解析解（验证对比）
```

---

## 完整训练和验证流程

```python
import scipy.io as sio
import numpy as np

# ========== 1. 训练阶段 ==========
# 加载3DEC训练数据
train_data = sio.loadmat('elastic_hole_3dec_train.mat')

# 训练PINN（数据驱动 + 物理约束）
model = PINN(layers=[3, 50, 50, 50, 9])  # 输入(x,y,z) → 输出(u_x,u_y,u_z,σ_xx,...)
model.train(
    X_train=train_data['x_train'],
    U_train=train_data['u_x_train'],  # 3DEC结果作为标签
    physics_weight=0.1,  # 物理损失权重
    epochs=10000
)

# ========== 2. 预测阶段 ==========
# 在测试点上预测
X_test = ... # 测试点坐标
U_pred = model.predict(X_test)  # PINN预测

# ========== 3. 验证阶段（三方对比）==========
# 加载解析解和3DEC完整结果
analytical = sio.loadmat('elastic_hole_analytical.mat')
full_3dec = sio.loadmat('elastic_hole_3dec_full.mat')

# 计算误差
error_vs_analytical = np.linalg.norm(U_pred - analytical['u_x_analytical'])
error_vs_3dec = np.linalg.norm(U_pred - full_3dec['u_x_3dec'])

# 可视化三方对比
plot_comparison(U_pred, analytical, full_3dec)
```

---

## 数据导出脚本

### 已创建的脚本：

1. **`Elastic_Hole/generate_analytical_solution.py`** ✅ 
   - 生成Kirsch解析解（验证对比用）
   - 输出: `elastic_hole_analytical.mat`

2. **`Elastic_Hole/export_to_pinn.py`** ✅ 
   - 从3DEC导出训练数据
   - 输出: `elastic_hole_3dec_train.mat` 或 `elastic_hole_3dec_full.mat`

### 使用步骤：

```bash
# 1. 在3DEC中运行模型
3DEC > call ElasticHole-Hex.dat
3DEC > model cycle 2000

# 2. 导出3DEC数据
cd Elastic_Hole
python export_to_pinn.py  # 按提示先在3DEC中运行导出脚本

# 3. 生成解析解数据
python generate_analytical_solution.py

# 4. 训练PINN并对比
cd ../PINNs
python train_elastic_pinn.py
```

---

## 评估指标

对比PINN、3DEC、解析解时可以计算：

1. **L2相对误差**
```python
L2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
```

2. **最大绝对误差**
```python
max_error = np.max(np.abs(u_pred - u_exact))
```

3. **均方根误差（RMSE）**
```python
rmse = np.sqrt(np.mean((u_pred - u_exact)**2))
```

---

## 使用scipy读取.mat文件示例

```python
import scipy.io as sio
import numpy as np

# 读取数据
data = sio.loadmat('elastic_hole_3dec.mat')

# 提取坐标
x = data['x'].flatten()
y = data['y'].flatten()
z = data['z'].flatten()

# 提取位移
u_x = data['u_x'].flatten()
u_y = data['u_y'].flatten()
u_z = data['u_z'].flatten()

# 提取应力
sigma_xx = data['sigma_xx'].flatten()
sigma_yy = data['sigma_yy'].flatten()
# ... 等等
```

---

## 注意事项

1. **坐标系统**：确保3DEC和PINN使用相同的坐标系统（笛卡尔坐标或柱坐标）
2. **单位一致性**：所有数据使用SI单位（m, Pa）
3. **数据归一化**：PINN训练前通常需要对数据进行归一化处理
4. **采样密度**：边界条件采样点要足够密集，内部配点数量要充足

---

**参考**: `continuous_time_inference (Schrodinger)/Schrodinger.py` 中的数据加载部分
