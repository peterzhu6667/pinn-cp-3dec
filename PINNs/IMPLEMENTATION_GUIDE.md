# 弹性力学PINN实现指南

## 基于现有代码的实现建议

### 参考代码选择

**推荐参考**: `main/continuous_time_inference (Schrodinger)/Schrodinger.py`

**原因**：
1. ✅ 连续时间模型（无时间演化）
2. ✅ 多维空间推理问题
3. ✅ 包含复杂PDE约束
4. ✅ 代码结构清晰，易于修改

### 核心修改点

#### 1. 问题定义

```python
# 原 Schrödinger 问题: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + V(x)ψ
# 新 弹性力学问题: ∇·σ = 0, σ = C:ε, ε = ∇_s u
```

**修改**：
- 输入维度：(x, y, z) ∈ [-10, 10] × [0, 1] × [-10, 10]
- 输出维度：位移场 (u_x, u_y, u_z)
- PDE约束：平衡方程 + 本构关系 + 几何方程

#### 2. 神经网络架构

```python
class ElasticPINN:
    def __init__(self, layers):
        # 输入: [x, y, z]
        # 输出: [u_x, u_y, u_z]
        self.layers = layers  # 例如: [3, 50, 50, 50, 50, 3]
        
    def net_u(self, x, y, z):
        """位移场预测"""
        return u_x, u_y, u_z
    
    def net_stress(self, x, y, z):
        """应力场计算（通过自动微分）"""
        # ε = ∇_s u (对称梯度)
        # σ = C : ε (各向同性弹性)
        return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz
    
    def net_equilibrium(self, x, y, z):
        """平衡方程残差"""
        # ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z = 0
        # ∂σ_xy/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z = 0
        # ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z = 0
        return residual_x, residual_y, residual_z
```

#### 3. 损失函数

```python
def loss_function():
    # 物理损失 (内部点)
    loss_eq = MSE(equilibrium_residual)
    
    # 边界损失 (孔边界: 自由表面)
    loss_hole = MSE(sigma_n_hole)
    
    # 边界损失 (远场边界: 应力边界条件)
    loss_far = MSE(sigma - sigma_prescribed)
    
    # 平面应变约束 (u_y = 0)
    loss_plane_strain = MSE(u_y)
    
    # 总损失
    loss = loss_eq + lambda_hole * loss_hole + \
           lambda_far * loss_far + lambda_ps * loss_plane_strain
    
    return loss
```

#### 4. 训练数据采样

```python
# 内部点 (拉丁超立方采样)
N_interior = 10000
X_interior = sample_interior_points(N_interior)

# 孔边界 (圆柱面 r=1)
N_hole = 1000
theta = np.random.uniform(0, 2*np.pi, N_hole)
z = np.random.uniform(0, 1, N_hole)
X_hole = np.column_stack([np.cos(theta), z, np.sin(theta)])

# 远场边界 (r=10)
N_far = 1000
X_far = sample_far_field_boundary(N_far)

# 对称面 (y=0, y=1, x=0, z=0)
N_symmetry = 500
X_symmetry = sample_symmetry_planes(N_symmetry)
```

#### 5. 材料参数

```python
# 弹性参数
K = 3.9e9  # 体积模量 (Pa)
G = 2.8e9  # 剪切模量 (Pa)

# 拉梅常数
nu = (3*K - 2*G) / (6*K + 2*G)  # 泊松比
lmbda = K - 2*G/3  # 第一拉梅常数
mu = G             # 第二拉梅常数

# 应力-应变关系 (各向同性)
def constitutive(epsilon):
    """
    sigma_ij = lambda * delta_ij * tr(epsilon) + 2 * mu * epsilon_ij
    """
    trace_eps = epsilon_xx + epsilon_yy + epsilon_zz
    sigma_xx = lmbda * trace_eps + 2*mu * epsilon_xx
    sigma_yy = lmbda * trace_eps + 2*mu * epsilon_yy
    sigma_zz = lmbda * trace_eps + 2*mu * epsilon_zz
    sigma_xy = 2*mu * epsilon_xy
    sigma_yz = 2*mu * epsilon_yz
    sigma_xz = 2*mu * epsilon_xz
    return sigma
```

### 关键技术要点

#### 自动微分

TensorFlow提供自动微分计算应变和应力散度：

```python
import tensorflow as tf

# 一阶导数（应变）
du_dx = tf.gradients(u, x)[0]
du_dy = tf.gradients(u, y)[0]
# ...

# 二阶导数（平衡方程）
dsigma_xx_dx = tf.gradients(sigma_xx, x)[0]
# ...
```

#### 训练策略

1. **Adam优化器**: 快速收敛到局部最优
2. **L-BFGS优化器**: 精细调整到全局最优
3. **学习率衰减**: 初始 1e-3, 指数衰减
4. **权重平衡**: 动态调整各损失项权重

#### 验证指标

```python
# 与解析解比较
def validate_against_analytical():
    # Kirsch解析解（见 Elastic_Hole/ElasticHole.fis）
    u_analytical = kirsch_displacement(x, y, z)
    sigma_analytical = kirsch_stress(x, y, z)
    
    # 误差计算
    L2_error_u = np.linalg.norm(u_pred - u_analytical) / \
                 np.linalg.norm(u_analytical)
    L2_error_sigma = np.linalg.norm(sigma_pred - sigma_analytical) / \
                     np.linalg.norm(sigma_analytical)
    
    return L2_error_u, L2_error_sigma
```

### 实现步骤

**第一步**: 复制 `Schrodinger.py` 为 `ElasticHole.py`
```bash
cp main/continuous_time_inference\ (Schrodinger)/Schrodinger.py \
   main/ElasticHole_PINN/ElasticHole.py
```

**第二步**: 修改网络结构（3输入3输出）

**第三步**: 实现应变计算（对称梯度）

**第四步**: 实现应力计算（本构关系）

**第五步**: 实现平衡方程残差

**第六步**: 定义边界条件和损失函数

**第七步**: 配置训练参数和采样点

**第八步**: 训练和验证

**第九步**: 可视化结果对比

### 预期输出

1. **训练曲线**: 损失函数随迭代次数的变化
2. **应力场云图**: σ_rr, σ_θθ 沿径向分布
3. **位移场云图**: u_r 沿径向分布
4. **误差分析**: 与解析解和3DEC结果对比
5. **收敛性分析**: 不同网络深度/宽度的性能

### 扩展方向

- **几何泛化**: 椭圆孔、多孔、任意形状孔
- **材料非线性**: 弹塑性、损伤本构
- **动力学**: 波传播、振动分析
- **耦合问题**: 热-力耦合、流-固耦合

---

## 参考文献

1. Raissi et al. (2019) - 原始PINN论文
2. Haghighat et al. (2021) - 弹性力学PINN应用
3. Samaniego et al. (2020) - 能量方法PINN

## 相关代码示例

查看项目中的以下文件作为参考：
- `main/continuous_time_inference (Schrodinger)/Schrodinger.py` - 网络架构
- `Utilities/plotting.py` - 可视化工具
- `Elastic_Hole/ElasticHole.fis` - 解析解实现
