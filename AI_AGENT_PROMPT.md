# AI代理任务提示词

## 任务概述

你是一个具有网页浏览能力的AI代理。我需要你帮我修正一个Python脚本中关于3DEC FISH语言的理解错误。

---

## 背景信息

### 项目目标
研究物理信息神经网络（PINN）在土木工程弹性力学中的适应性，通过将PINN应用于圆柱孔弹性力学问题，对比PINN、3DEC数值解和Kirsch解析解的结果。

### 项目结构
请先阅读以下文件了解项目背景：
1. **`README.md`** - 项目总体介绍
2. **`Elastic_Hole/project.md`** - 圆柱孔弹性力学问题的详细描述
3. **`Elastic_Hole/ElasticHole.fis`** - 3DEC FISH脚本，包含Kirsch解析解的实现
4. **`QUICKSTART.md`** - 项目使用指南

### 当前问题
文件 **`Elastic_Hole/generate_analytical_solution.py`** 中，我尝试将FISH脚本（`ElasticHole.fis`）中的Kirsch解析解公式转换为Python代码，但可能由于对FISH语言理解不准确，导致公式转换有误。

---

## 你的任务

### 第一步：查阅3DEC FISH语言手册

**手册链接**：
- 3DEC FISH脚本语言参考手册：https://docs.itascacg.com/3dec700/common/fish/doc/fish_language.html
- 3DEC FISH内置函数：https://docs.itascacg.com/3dec700/common/fish/doc/fish_functions.html
- 3DEC Zone命令：https://docs.itascacg.com/3dec700/3dec/docproject/source/manual/zone_commands/zone_commands.html

**重点查看**：
1. FISH语言的语法规则（特别是数学运算符）
2. 内置函数的定义（如 `math.sqrt()`, `math.atan2()`, `cos()`, `sin()` 等）
3. 变量赋值和计算的逻辑

### 第二步：理解Kirsch解析解的物理背景

参考文献（已在 `Elastic_Hole/project.md` 中说明）：
- Kirsch, G. (1898). Die Theorie der Elastizität
- Jaeger, J.C., Cook, N.G.W. (1976). Fundamentals of Rock Mechanics

**物理问题**：
- 无限弹性介质中的圆柱孔
- 远场应力：σ_x = 15 MPa（水平），σ_y = 30 MPa（垂直）
- 材料：剪切模量 G = 2.8 GPa，体积模量 K = 3.9 GPa

**Kirsch解（柱坐标系）**：
```
径向应力: σ_r(r,θ) = f1(r,θ)
环向应力: σ_θ(r,θ) = f2(r,θ)
径向位移: u_r(r,θ) = f3(r,θ)
```

### 第三步：对比和修正

**对比内容**：
1. 打开 `Elastic_Hole/ElasticHole.fis`，找到以下FISH函数：
   - `nastr` - 计算应力的函数
   - `nadis` - 计算位移的函数

2. 对比 `Elastic_Hole/generate_analytical_solution.py` 中的对应Python实现：
   - `KirschSolution.stress_radial()`
   - `KirschSolution.stress_tangential()`
   - `KirschSolution.displacement_radial()`
   - `KirschSolution.displacement_tangential()`

3. **特别注意**：
   - 变量命名对应关系（如 FISH中的 `p1`, `p2`, `o_r`, `o_r2` 等）
   - 数学公式的正确性（括号、运算符优先级）
   - 坐标系统（FISH中使用 `x`, `y`, `z`，是否对应Python中的柱坐标 `r`, `θ`）
   - **远场应力的对应关系**：FISH中 `p1` 和 `p2` 分别对应哪个方向的应力？

### 第四步：修正Python脚本

根据手册和FISH代码，修正 `Elastic_Hole/generate_analytical_solution.py` 中的错误。

**需要检查的关键点**：
1. ✅ 公式是否与FISH代码完全一致？
2. ✅ 变量对应关系是否正确？（如 `p1` vs `p2`，哪个是水平应力？）
3. ✅ 坐标系转换是否正确？（柱坐标 ↔ 笛卡尔坐标）
4. ✅ 泊松比计算公式是否正确？
5. ✅ 位移公式中的环向分量 `u_θ` 是否有FISH实现？如果没有，需要从理论推导

### 第五步：验证修正结果

修正后，确保：
1. Python脚本可以正常运行
2. 生成的 `analytical_solution.png` 图片合理（应力集中在孔边缘，远场趋于均匀）
3. 数值量级合理（位移约毫米级，应力约几十MPa）

---

## 输出要求

### 1. 问题诊断报告
```markdown
## 发现的问题

### 问题1: [问题描述]
- FISH代码: [相关代码行]
- Python代码: [对应代码行]
- 错误原因: [说明]
- 修正方法: [说明]

### 问题2: ...
```

### 2. 修正后的代码
提供完整修正后的 `generate_analytical_solution.py` 文件

### 3. 验证说明
说明如何验证修正是否正确（预期的输出结果）

---

## 注意事项

1. **坐标系统**：3DEC中的 `x`, `y`, `z` 与Python中的柱坐标 `r`, `θ` 的对应关系
2. **FISH语法**：FISH中的运算符优先级可能与Python不同
3. **角度单位**：确认FISH中使用弧度还是角度
4. **应力符号约定**：压应力为正还是负？

---

## 后续任务提示

修正完成后，我还需要你帮助：
1. 创建PINN网络结构（基于TensorFlow/PyTorch）
2. 实现弹性力学的物理损失函数
3. 优化超参数和网络架构

请在完成当前任务后，等待进一步指示。

---

## 开始任务

现在请：
1. 访问3DEC FISH手册链接
2. 阅读项目相关文档
3. 对比FISH代码和Python代码
4. 提供问题诊断和修正方案

谢谢！
