# 气动系统状态空间分析工具

这个项目提供了一套用于气动系统状态空间分析的工具，可以进行线性化分析、状态反馈控制器设计、非线性系统模拟以及PID控制模拟。

## 项目结构

```
ss/
├── main.py               # 主程序入口
├── modules/              # 模块目录
│   ├── __init__.py       # 包初始化文件
│   ├── pid_controller.py # PID控制器实现
│   ├── state_space_model.py # 状态空间模型
│   ├── system_dynamics.py # 系统动力学
│   └── visualization.py  # 可视化工具
└── results/              # 结果输出目录（自动创建）
```

## 功能特性

- **线性系统分析**：线性化系统模型，分析稳定性、可控性和可观性
- **状态反馈控制器设计**：使用极点配置法设计状态反馈控制器
- **非线性系统模拟**：模拟完整的非线性系统动态行为
- **线性与非线性比较**：比较线性化模型和非线性模型的响应差异
- **PID控制模拟**：使用模糊PID控制器进行位置控制模拟
- **可视化工具**：生成系统响应、相空间轨迹、根轨迹和Bode图

## 安装依赖

确保已安装以下Python库：

```bash
pip install numpy matplotlib scipy control
```

## 使用方法

### 基本用法

```bash
python main.py
```

默认情况下，程序将运行所有分析模式。

### 指定分析模式

```bash
# 仅运行线性分析
python main.py --mode linear

# 仅运行非线性分析
python main.py --mode nonlinear

# 运行线性与非线性比较
python main.py --mode compare

# 运行PID控制模拟
python main.py --mode pid
```

### 自定义系统参数

```bash
python main.py --m_ball 0.05 --A 0.001 --f_v 0.1 --V_tube 0.01 --L_tube 0.5 --P_s 500000
```

### 自定义控制器参数

```bash
python main.py --desired_poles "-3,-3.5,-4,-4.5" --kp 0.3 --ki 0.0 --kd 20.0
```

### 修改模拟参数

```bash
python main.py --t_span 2.0 --dt 0.01 --x0 "0.26,0,200000,200000" --pid_duration 28.0
```

### 保存和显示图表

```bash
# 保存图表并生成报告
python main.py --save_figs --output_dir results

# 显示图表
python main.py --show_plots
```

## 自定义实验

### 比较不同摩擦系数

```bash
python main.py --mode compare --f_v 0.05 --save_figs --output_dir "results/low_friction"
python main.py --mode compare --f_v 0.2 --save_figs --output_dir "results/high_friction"
```

### 比较不同PID参数

```bash
python main.py --mode pid --kp 0.1 --kd 10.0 --save_figs --output_dir "results/pid_low"
python main.py --mode pid --kp 0.5 --kd 30.0 --save_figs --output_dir "results/pid_high"
```

### 调整模拟时间和步长

```bash
# 长时间模拟
python main.py --mode pid --pid_duration 84.0 --save_figs

# 高精度模拟
python main.py --mode compare --dt 0.001 --save_figs
```

## 系统参数说明

- `m_ball`: 小球质量 (kg)
- `A`: 管道横截面积 (m²)
- `f_v`: 粘性摩擦系数
- `V_tube`: 管道总体积 (m³)
- `L_tube`: 管道长度 (m)
- `P_s`: 供气压力 (Pa)
- `P_atm`: 大气压力 (Pa)
- `x_eq`: 平衡位置 (m)
- `P_l_eq`: 左气室平衡压力 (Pa)
- `P_r_eq`: 右气室平衡压力 (Pa)

## 控制器参数说明

- `desired_poles`: 期望极点 (逗号分隔)
- `kp`: PID控制器比例增益
- `ki`: PID控制器积分增益
- `kd`: PID控制器微分增益

## 生成的图表

- 线性/非线性模型响应
- 状态空间轨迹
- 根轨迹
- Bode图
- 线性与非线性模型比较
- PID控制响应
