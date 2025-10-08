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
3. **`Elastic_Hole/ElasticHole-Hex.dat`** - 3DEC模型文件
4. **`QUICKSTART.md`** - 项目使用指南
5. **`PINNs/main/Data/README.md`** - 数据格式说明

### 当前问题
文件 **`Elastic_Hole/export_to_pinn.py`** 中，我尝试创建一个FISH脚本来从3DEC导出数据，但由于对3DEC FISH语言理解不准确，生成的FISH脚本可能有语法错误或逻辑错误，无法正常导出数据。

**具体问题**：
- `export_to_pinn.py` 中的 `create_sample_fish_export_script()` 函数生成的FISH脚本可能有语法错误
- 不确定3DEC的数据结构访问方式是否正确（如 `block.gp.pos.x()`, `block.zone.stress.xx()` 等）
- 文件操作（`file.open()`, `file.write()`, `file.close()`）的语法可能不正确
- 循环结构（`loop foreach`）的写法可能有误

---

## 你的任务

### 第一步：查阅3DEC FISH语言手册

**手册链接（请使用浏览器访问）**：
- **3DEC FISH脚本语言参考**：https://docs.itascacg.com/3dec700/common/fish/doc/fish_language.html
- **3DEC FISH内置函数**：https://docs.itascacg.com/3dec700/common/fish/doc/fish_functions.html
- **3DEC Block命令参考**：https://docs.itascacg.com/3dec700/3dec/docproject/source/manual/block/block.html
- **3DEC Zone相关命令**：https://docs.itascacg.com/3dec700/3dec/docproject/source/manual/zone_commands/zone_commands.html
- **FISH文件I/O操作**：https://docs.itascacg.com/3dec700/common/fish/doc/fish_functions.html (搜索 "file" 相关函数)

**重点查看**：
1. **FISH语法规则**：
   - 变量声明（`local` 关键字）
   - 循环结构（`loop foreach`, `loop while`, `end_loop`）
   - 条件语句（`if`, `then`, `end_if`）
   - 数学运算和函数（`math.sqrt()`, `math.abs()` 等）

2. **3DEC数据结构访问**：
   - 如何遍历所有gridpoint（网格点）：`block.gp.list`
   - 如何访问gridpoint的坐标：`block.gp.pos.x()`, `block.gp.pos.y()`, `block.gp.pos.z()`
   - 如何访问gridpoint的位移：`block.gp.disp.x()`, `block.gp.disp.y()`, `block.gp.disp.z()`
   - 如何遍历所有zone（单元）：`block.zone.list`
   - 如何访问zone的应力：`block.zone.stress.xx()`, `block.zone.stress.yy()` 等

3. **文件I/O操作**：
   - `file.open()` 的正确语法和参数
   - `file.write()` 如何写入数据
   - `file.close()` 的用法
   - 如何写入换行符

### 第二步：分析现有的FISH脚本示例

查看项目中可能存在的其他FISH脚本（如果有），了解正确的FISH代码风格。

### 第三步：对比和修正 `export_to_pinn.py`

**打开文件**：`Elastic_Hole/export_to_pinn.py`

**找到函数**：`create_sample_fish_export_script(output_path)`

**需要检查的关键点**：

1. **FISH函数定义语法**：
   ```fish
   fish define export_for_pinn
       ; 函数体
   end
   ```
   - 是否缺少或多余关键字？
   - 缩进是否正确？

2. **变量声明**：
   ```fish
   local fp_nodes = file.open('nodes.txt', 1, 0)
   ```
   - `local` 关键字是否必需？
   - `file.open()` 的参数是否正确？（查手册确认参数含义）

3. **循环遍历gridpoint**：
   ```fish
   loop foreach local igp_ block.gp.list
       ; 访问坐标
       file.write(fp_nodes, block.gp.pos.x(igp_))
       ; ...
   end_loop
   ```
   - `loop foreach` 语法是否正确？
   - `block.gp.list` 是否是正确的列表获取方式？
   - `block.gp.pos.x(igp_)` 的调用方式是否正确？

4. **循环遍历zone**：
   ```fish
   loop foreach local iz_ block.zone.list
       file.write(fp_stress, block.zone.stress.xx(iz_))
       ; ...
   end_loop
   ```
   - Zone的应力访问方式是否正确？
   - 应力分量名称是否正确？（`xx`, `yy`, `zz`, `xy`, `xz`, `yz`）

5. **文件写入操作**：
   ```fish
   file.write(fp_nodes, block.gp.pos.x(igp_))
   file.write(fp_nodes, ' ')
   file.write(fp_nodes, 13)  ; 换行
   ```
   - 写入空格和换行符的方式是否正确？
   - `13` 是否是正确的换行符代码？（查手册确认）

6. **条件判断**：
   ```fish
   if math.abs(rad - 1.0) < 0.1 then
       ; ...
   end_if
   ```
   - `if` 语句语法是否正确？
   - `then` 关键字位置是否正确？

7. **数学计算**：
   ```fish
   local rad = math.sqrt(block.gp.pos.x(igp_)^2 + block.gp.pos.z(igp_)^2)
   ```
   - 幂运算符 `^` 是否正确？
   - 函数调用方式是否正确？

### 第四步：修正Python脚本

根据手册，修正 `Elastic_Hole/export_to_pinn.py` 中 `create_sample_fish_export_script()` 函数生成的FISH脚本。

**修正重点**：
1. ✅ FISH语法完全符合3DEC 7.0规范
2. ✅ 数据访问方式正确
3. ✅ 文件I/O操作正确
4. ✅ 循环和条件语句正确
5. ✅ 输出格式正确（用于后续Python读取）

### 第五步：验证修正结果

修正后的FISH脚本应该：
1. 在3DEC中可以正常运行（无语法错误）
2. 成功导出以下文本文件：
   - `nodes.txt` - 所有gridpoint的坐标
   - `displacement.txt` - 所有gridpoint的位移
   - `stress.txt` - 所有zone的应力
   - `boundary_hole.txt` - 孔边界节点（r ≈ 1.0m）
   - `boundary_far.txt` - 远场边界节点（r ≈ 5.0m）
3. 导出的数据格式正确（空格分隔，每行一个数据点）

---

## 输出要求

### 1. 问题诊断报告
```markdown
## 发现的问题

### 问题1: [问题描述]
- 当前FISH代码: [错误的代码行]
- 3DEC手册说明: [引用手册内容或链接]
- 错误原因: [说明]
- 修正方法: [说明]

### 问题2: ...
```

### 2. 修正后的代码
提供完整修正后的 `export_to_pinn.py` 文件，特别是 `create_sample_fish_export_script()` 函数中生成的FISH脚本部分。

### 3. 验证说明
说明如何在3DEC中测试修正后的FISH脚本：
```bash
# 在3DEC中
3DEC > call ElasticHole-Hex.dat
3DEC > model cycle 2000
3DEC > call export_3dec_data.fis
# 应该看到 "数据导出完成!" 消息
# 检查生成的文本文件
```

---

## 注意事项

1. **3DEC版本**：本项目使用3DEC 7.0，请参考对应版本的手册
2. **FISH语法严格性**：FISH对缩进、关键字大小写可能有要求
3. **数据类型**：确认gridpoint和zone的属性访问方式
4. **文件路径**：FISH中的文件路径可能需要特殊处理
5. **注释符号**：FISH使用 `;` 作为注释符号

---

## 参考资料

### 3DEC命令参考
- Block Gridpoint: https://docs.itascacg.com/3dec700/3dec/docproject/source/manual/block/block_gridpoint.html
- Block Zone: https://docs.itascacg.com/3dec700/3dec/docproject/source/manual/block/block_zone.html

### FISH编程示例
在手册中搜索 "FISH example" 或 "loop foreach example"

---

## 后续任务提示

修正完成后，我还需要你帮助：
1. 验证导出的数据格式是否符合PINN训练要求
2. 创建PINN网络结构（基于TensorFlow/PyTorch）
3. 实现弹性力学的物理损失函数
4. 优化超参数和网络架构

请在完成当前任务后，等待进一步指示。

---

## 开始任务

现在请：
1. ✅ 访问3DEC FISH手册链接，重点查看：
   - FISH语言语法
   - Block gridpoint/zone 数据访问
   - File I/O 操作
   - Loop 循环结构

2. ✅ 打开 `Elastic_Hole/export_to_pinn.py`，分析 `create_sample_fish_export_script()` 函数

3. ✅ 对比手册和代码，找出所有语法/逻辑错误

4. ✅ 提供详细的问题诊断报告和修正后的完整代码

谢谢！🚀
