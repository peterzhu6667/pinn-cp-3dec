# Physics-Informed Neural Networks (PINNs) - 核心理论

> **原始论文**: Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

本文档提取了PINN论文中与**弹性力学应用**相关的核心内容。

---

## 1. PINN核心思想

### 1.1 基本原理

物理信息神经网络（PINNs）是一种将物理定律（偏微分方程）嵌入神经网络训练过程的深度学习方法。其核心思想是：

**传统神经网络**：
```
Loss = Data_Loss (仅拟合数据)
```

**物理信息神经网络**：
```
Loss = Data_Loss + Physics_Loss (拟合数据 + 满足物理定律)
```

### 1.2 通用形式

考虑一般的非线性偏微分方程：

$$u_t + \mathcal{N}[u] = 0, \quad x \in \Omega, \quad t \in [0,T]$$

其中：
- $u(t,x)$ 是待求解的未知函数（如位移、应力等）
- $\mathcal{N}[\cdot]$ 是非线性微分算子
- $\Omega$ 是空间域

---

## 2. 连续时间模型（适用于弹性力学）

### 2.1 方法描述

对于**静态或准静态问题**（如弹性力学平衡问题），使用连续时间模型：

1. **神经网络逼近**：
   $$u(x) \approx \text{NN}(x; \theta)$$
   
2. **物理残差**：
   $$f(x) := \mathcal{N}[u(x)]$$
   
3. **自动微分计算导数**：通过TensorFlow/PyTorch的自动微分功能计算 $\nabla u, \nabla^2 u$ 等

### 2.2 损失函数

$$\text{Loss} = \lambda_u \cdot \text{MSE}_u + \lambda_f \cdot \text{MSE}_f + \lambda_b \cdot \text{MSE}_b$$

其中：
- **MSE_u**: 数据拟合损失（已知数据点）
  $$\text{MSE}_u = \frac{1}{N_u}\sum_{i=1}^{N_u} |u(x_i) - u_{\text{data}}^i|^2$$

- **MSE_f**: 物理方程残差（配点法）
  $$\text{MSE}_f = \frac{1}{N_f}\sum_{i=1}^{N_f} |f(x_i)|^2 = \frac{1}{N_f}\sum_{i=1}^{N_f} |\mathcal{N}[u](x_i)|^2$$

- **MSE_b**: 边界条件损失
  $$\text{MSE}_b = \frac{1}{N_b}\sum_{i=1}^{N_b} |\text{BC}(x_i)|^2$$

### 2.3 TensorFlow实现示例

```python
def neural_net(X, weights, biases):
    """前向传播"""
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def net_u(x, y, z):
    """位移场预测"""
    u = neural_net(tf.concat([x,y,z], 1), weights, biases)
    return u

def net_f(x, y, z):
    """物理方程残差（使用自动微分）"""
    u = net_u(x, y, z)
    # 计算一阶导数
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    # 计算二阶导数
    u_xx = tf.gradients(u_x, x)[0]
    # 物理方程残差
    f = ... # 根据具体PDE定义
    return f

# 损失函数
loss = tf.reduce_mean(tf.square(u_pred - u_data)) + \
       tf.reduce_mean(tf.square(f_pred))
```

---

## 3. 关键技术要点

### 3.1 采样策略

**拉丁超立方采样（Latin Hypercube Sampling, LHS）**：
- 在计算域内均匀分布配点
- 比随机采样更高效，覆盖更均匀

```python
import scipy.stats as stats

def sample_lhs(N, D):
    """
    N: 采样点数
    D: 维度
    """
    return stats.qmc.LatinHypercube(d=D).random(n=N)
```

### 3.2 网络架构

**推荐配置**（论文中经验值）：
- **深度**: 8-10层
- **宽度**: 每层20-50个神经元
- **激活函数**: tanh 或 sin（周期性问题）
- **初始化**: Xavier初始化

**示例**：
```python
layers = [3, 50, 50, 50, 50, 50, 50, 50, 50, 3]
# 输入维度=3 (x,y,z), 输出维度=3 (u_x,u_y,u_z)
```

### 3.3 优化策略

**两阶段训练**（论文推荐）：

1. **阶段1 - Adam优化器**：
   - 学习率: 1e-3
   - 迭代次数: 10,000 - 50,000
   - 作用: 快速收敛到较好的局部最优

2. **阶段2 - L-BFGS优化器**：
   - 迭代次数: 直到收敛
   - 作用: 精细调整，达到高精度

```python
# Adam优化
optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op_Adam = optimizer_Adam.minimize(loss)

# L-BFGS优化
optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(
    loss, 
    method='L-BFGS-B',
    options={'maxiter': 50000,
             'maxfun': 50000,
             'maxcor': 50,
             'maxls': 50,
             'ftol': 1.0 * np.finfo(float).eps})
```

### 3.4 权重平衡

损失函数中的权重系数 $\lambda_u, \lambda_f, \lambda_b$ 需要平衡：

**方法1 - 手动调整**：
```python
lambda_u = 1.0
lambda_f = 1.0
lambda_b = 10.0  # 边界条件更重要
```

**方法2 - 自适应权重**（论文未详述，但实践中有用）：
```python
# 归一化各项损失
loss = MSE_u / MSE_u_0 + MSE_f / MSE_f_0 + MSE_b / MSE_b_0
```

---

## 4. 适用于弹性力学的要点

### 4.1 控制方程

线弹性平衡方程（无体力）：

$$\nabla \cdot \sigma = 0$$

其中应力-应变-位移关系：

$$\begin{aligned}
\epsilon_{ij} &= \frac{1}{2}(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}) \quad &\text{(几何方程)} \\
\sigma_{ij} &= \lambda \epsilon_{kk} \delta_{ij} + 2\mu \epsilon_{ij} \quad &\text{(本构方程)}
\end{aligned}$$

**平衡方程展开**（3D）：
$$\begin{aligned}
\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + \frac{\partial \sigma_{xz}}{\partial z} &= 0 \\
\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + \frac{\partial \sigma_{yz}}{\partial z} &= 0 \\
\frac{\partial \sigma_{xz}}{\partial x} + \frac{\partial \sigma_{yz}}{\partial y} + \frac{\partial \sigma_{zz}}{\partial z} &= 0
\end{aligned}$$

### 4.2 PINN实现要点

1. **输入**: 空间坐标 $(x, y, z)$
2. **输出**: 位移场 $(u_x, u_y, u_z)$
3. **物理损失**: 平衡方程残差 $|\nabla \cdot \sigma|^2$
4. **边界损失**: 
   - 位移边界: $|u - u_{\text{prescribed}}|^2$
   - 应力边界: $|\sigma \cdot n - t_{\text{prescribed}}|^2$

### 4.3 与有限元的对比

| 特性 | 有限元 (FEM) | PINN |
|------|-------------|------|
| 离散化 | 需要网格 | 无网格（配点） |
| 几何复杂度 | 网格生成困难 | 容易处理复杂边界 |
| 高阶导数 | 需要高阶单元 | 自动微分容易获得 |
| 计算效率 | 稀疏矩阵求解快 | 训练时间长 |
| 参数化研究 | 每次重新计算 | 训练后推理快 |

---

## 5. 典型案例参考

### 案例1: Burgers方程（1D非线性扩散）

**方程**：
$$u_t + u u_x - \nu u_{xx} = 0$$

**PINN实现**：
```python
def net_f(t, x):
    u = net_u(t, x)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f = u_t + u*u_x - nu*u_xx
    return f
```

**结果**：
- 配点数: 10,000
- 训练时间: 60秒（单GPU）
- 精度: L2误差 < 1%

### 案例2: Schrödinger方程（复数值PDE）

**方程**：
$$i h_t + 0.5 h_{xx} + |h|^2 h = 0$$

**PINN实现要点**：
- 复数分解: $h = u + iv$
- 双输出网络: $[u(t,x), v(t,x)]$
- 周期边界条件处理

**结果**：
- 配点数: 20,000
- 网络结构: [2, 100, 100, 100, 100, 100, 2]
- 成功预测复杂的孤子演化

---

## 6. 实践建议

### 6.1 调试技巧

1. **可视化损失分量**：
   ```python
   print(f"MSE_u: {MSE_u:.2e}, MSE_f: {MSE_f:.2e}, MSE_b: {MSE_b:.2e}")
   ```
   - 如果某项始终很大，增加其权重或配点数

2. **检查梯度**：
   ```python
   grad_u_x = tf.gradients(u, x)[0]
   print(f"Max gradient: {tf.reduce_max(tf.abs(grad_u_x))}")
   ```
   - 梯度爆炸/消失说明网络深度或学习率有问题

3. **逐步增加复杂度**：
   - 先用简单几何测试
   - 再加复杂边界条件

### 6.2 常见问题

**Q1: 损失不下降？**
- 检查学习率（尝试1e-4到1e-2）
- 增加网络宽度/深度
- 检查损失函数权重平衡

**Q2: 过拟合数据但物理方程不满足？**
- 增加 $\lambda_f$（物理损失权重）
- 增加配点数 $N_f$
- 检查自动微分实现是否正确

**Q3: 边界条件不满足？**
- 增加 $\lambda_b$（边界损失权重）
- 在边界附近增加配点密度
- 使用硬约束（直接在网络输出中编码边界条件）

### 6.3 性能优化

1. **GPU加速**：
   ```python
   with tf.device('/GPU:0'):
       # 构建模型
   ```

2. **批处理训练**：
   ```python
   batch_size = 1000
   for epoch in range(epochs):
       for batch in get_batches(X_train, batch_size):
           train_step(batch)
   ```

3. **并行采样**：
   ```python
   from multiprocessing import Pool
   with Pool(4) as p:
       samples = p.map(sample_function, ranges)
   ```

---

## 7. 弹性力学应用的扩展

### 7.1 高级本构模型

- **弹塑性**: $\sigma = f(\epsilon, \epsilon^p)$
- **损伤力学**: $\sigma = (1-D) C : \epsilon$
- **超弹性**: $\sigma = \frac{\partial W}{\partial \epsilon}$

### 7.2 多物理场耦合

- **热-力耦合**: 同时求解温度场和位移场
- **流-固耦合**: 流体与结构相互作用
- **多尺度**: 宏观-细观耦合模拟

### 7.3 参数反演

PINN的一大优势是可以同时推断未知参数：

```python
# 将材料参数作为可训练变量
E = tf.Variable(initial_value=1e9, trainable=True)  # 弹性模量
nu = tf.Variable(initial_value=0.3, trainable=True)  # 泊松比

# 损失函数同时优化网络参数和材料参数
loss = MSE_u + MSE_f(E, nu) + MSE_b
```

---

## 8. 参考文献

### 核心论文

1. **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2017).** Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. *arXiv:1711.10561*.

3. **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2017).** Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations. *arXiv:1711.10566*.

### 弹性力学应用

4. **Haghighat, E., Raissi, M., Moure, A., Gomez, H., & Juanes, R. (2021).** A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. *Computer Methods in Applied Mechanics and Engineering*, 379, 113741.

5. **Samaniego, E., Anitescu, C., Goswami, S., Nguyen-Thanh, V.M., Guo, H., Hamdia, K., Zhuang, X., & Rabczuk, T. (2020).** An energy approach to the solution of partial differential equations in computational mechanics via machine learning: Concepts, implementation and applications. *Computer Methods in Applied Mechanics and Engineering*, 362, 112790.

---

## 附录：推荐超参数（基于论文经验）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 网络深度 | 8-10层 | 太深可能梯度消失 |
| 网络宽度 | 20-50神经元/层 | 根据问题复杂度调整 |
| 激活函数 | tanh | sin用于周期问题 |
| 初始化 | Xavier/Glorot | 标准初始化 |
| Adam学习率 | 1e-3 | 可尝试1e-4到1e-2 |
| Adam迭代 | 10,000-50,000 | 直到损失平稳 |
| L-BFGS迭代 | 50,000 | 或设置收敛容差 |
| 配点数 $N_f$ | 10,000-100,000 | 越多越好，但计算成本高 |
| 数据点 $N_u$ | 100-1,000 | 根据可用数据 |
| 边界点 $N_b$ | 100-1,000 | 边界密度要足够 |

---

**本文档整理自原始论文，聚焦于与弹性力学应用相关的核心内容。完整论文请参考上述文献。**
