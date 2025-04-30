#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气动系统状态空间分析主程序
========================
此程序集成所有模块，运行气动系统的状态空间分析和控制模拟。

作者: Claude AI
日期: 2025-04-21
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import traceback

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入模块
try:
    from modules.state_space_model import PneumaticStateSpaceModel
    from modules.simulation_v1 import PneumaticBallSystem # Add import for basic simulation
    from modules.visualization import (
    plot_response,
    plot_state_space,
    plot_root_locus,
    plot_bode,
)
    print("成功导入所需模块")
except ImportError as e:
    # Check if the error is specifically about pid_controller and ignore if so,
    # otherwise raise the error. This allows other modules to be imported.
    if "No module named 'modules.pid_controller'" in str(e):
        print(f"导入pid_controller模块失败 (已忽略): {e}")
    else:
        print(f"模块导入错误: {e}")
        sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='气动系统状态空间分析程序')

    # 分析模式选择
    parser.add_argument('--mode', type=str, choices=['basic', 'linear', 'all'], # Update choices
                        default='all', help='分析模式: 基础物理, 线性分析, 或全部')

    # 系统参数
    parser.add_argument('--m_ball', type=float, default=0.003, help='小球质量 (kg)') # Updated default
    parser.add_argument('--A', type=float, default=0.0001, help='管道横截面积 (m²)') # Updated default
    parser.add_argument('--f_v', type=float, default=0.05, help='粘性摩擦系数') # Updated default
    parser.add_argument('--V_tube', type=float, default=0.0015, help='管道总体积 (m³)') # Updated default
    parser.add_argument('--L_tube', type=float, default=0.15, help='管道长度 (m)') # Updated default
    parser.add_argument('--P_s', type=float, default=500000, help='供气压力 (Pa)')
    parser.add_argument('--P_atm', type=float, default=101325, help='大气压力 (Pa)')

    # System reference position
    parser.add_argument('--x_s', type=float, default=0.1, help='小球初始参考位置 (m)') # Added x_s argument

    # 平衡点参数
    parser.add_argument('--x_eq', type=float, default=0.075, help='平衡位置 (m)') # Updated default
    parser.add_argument('--P_l_eq', type=float, default=180000, help='左气室平衡压力 (Pa)') # Updated default
    parser.add_argument('--P_r_eq', type=float, default=180000, help='右气室平衡压力 (Pa)') # Updated default

    # 控制器参数
    parser.add_argument('--desired_poles', type=str, default='-3,-3.5,-4,-4.5',
                        help='期望极点 (逗号分隔)')

    # 模拟参数 - 减少模拟时间方便测试
    parser.add_argument('--t_span', type=float, default=15.0, help='模拟时间 (s)') # Increased simulation time
    parser.add_argument('--dt', type=float, default=0.005, help='时间步长 (s)')
    parser.add_argument('--x0', type=str, default='-0.099,0,101325,101325', # Adjusted initial state to avoid starting exactly at 0 position
                        help='初始状态 (逗号分隔: x_0, v_0, P_l, P_r)')

    # 输出参数
    parser.add_argument('--save_figs', action='store_true', default=True, help='保存图表')
    parser.add_argument('--output_dir', type=str, default='ss/results', help='图表保存目录')
    parser.add_argument('--show_plots', action='store_true', help='显示图表')

    return parser.parse_args()

def build_params(args):
    """从命令行参数构建系统参数"""
    params = {
        'm_ball': args.m_ball,
        'A': args.A,
        'f_v': args.f_v,
        'V_tube': args.V_tube,
        'L_tube': args.L_tube,
        'P_s': args.P_s,
        'P_atm': args.P_atm,
        'x_s': args.x_s, # Add x_s to params
        'x_eq': args.x_eq,
        'P_l_eq': args.P_l_eq,
        'P_r_eq': args.P_r_eq,
        't_span': (0, args.t_span) # Add t_span to params
    }

    # Parse initial state string
    x0_values = [float(x) for x in args.x0.split(',')]
    # Initial absolute position = initial deviation (from x0 arg) + initial reference (x_s param)
    params['x_b'] = x0_values[0] + params['x_s']
    params['v_b'] = x0_values[1]
    params['P_l'] = x0_values[2]
    params['P_r'] = x0_values[3]

    return params

def ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def save_figure(fig, filename, output_dir):
    """保存图表到指定目录"""
    if not os.path.exists(output_dir):
        ensure_output_dir(output_dir)

    filepath = os.path.join(output_dir, filename)
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save chart: {e}")
def run_basic_simulation(params, save_animation=False, output_dir='ss/results'):
    """运行基础物理模拟"""
    print("========== Running Basic Physical Simulation ==========")

    # 创建系统实例
    system = PneumaticBallSystem(params)

    # List to store duty cycle values
    duty_cycles = []

    # 定义一个模拟 PWM 控制的阀门开度函数
    def pwm_spool_func(t):
        # Implement user's duty cycle logic
        if t <= 2.00:
            duty_cycle = 100
        elif t <= 8.00:
            duty_cycle = 50
        elif t <= 8.10:
            duty_cycle = 0
        else:
            duty_cycle = 50
        
        # Store the duty cycle
        # This approach of appending inside the ODE function can be unreliable
        # Let's rely on recalculating based on simulation times later
        # duty_cycles.append(duty_cycle)

        # Map duty cycle [0, 100] to spool position [-1, 1]
        s_spool = (duty_cycle / 50.0) - 1.0
        return s_spool

    # 运行模拟，传入阀门开度函数
    print("Running physical simulation...")
    start_time = time.time()
    # Pass the time-varying spool function to simulate
    result = system.simulate(s_spool_func=pwm_spool_func)
    sim_time = time.time() - start_time
    print(f"Simulation complete, calculation time: {sim_time:.3f} seconds")

    # Recalculate duty cycles based on the simulation times for consistency
    calculated_duty_cycles = []
    for t in system.times:
        if t <= 2.00:
            calculated_duty_cycles.append(100)
        elif t <= 8.00:
            calculated_duty_cycles.append(50)
        elif t <= 8.10:
            calculated_duty_cycles.append(0)
        else:
            calculated_duty_cycles.append(50)
    duty_cycles_array = np.array(calculated_duty_cycles)


    # 绘制结果 - 绘制位置图和控制输入 (PWM 占空比) 图
    print("Drawing simulation results plots...")

    # 计算实际位置 (转换为 mm)
    x_b_mm = (system.states[0] + system.x_s) * 1000

    # 绘制位置时间图 (单独的图)
    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
    ax_pos.plot(system.times, x_b_mm, 'b-')
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (mm)')
    ax_pos.set_title('Ball Position') # Simplified title
    ax_pos.grid(True)
    # Add minor grid lines for better readability
    ax_pos.minorticks_on()
    ax_pos.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax_pos.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')


    # 绘制控制输入 (PWM 占空比) 图 (单独的图)
    fig_pwm, ax_pwm = plt.subplots(figsize=(10, 5))
    ax_pwm.plot(system.times, duty_cycles_array, 'b-') # Use blue line as in the example image
    ax_pwm.set_xlabel('Time (s)')
    ax_pwm.set_ylabel('PWM Duty Cycle (%)') # Update label
    ax_pwm.set_title('PWM Duty Cycle') # Simplified title
    ax_pwm.grid(True)
    ax_pwm.set_ylim(-10, 110) # Set reasonable limits for duty cycle
    # Add minor grid lines for better readability
    ax_pwm.minorticks_on()
    ax_pwm.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax_pwm.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')


    plt.tight_layout() # Adjust layout for separate figures

    # 保存图表
    if output_dir:
        ensure_output_dir(output_dir)
        fig_pos_path = os.path.join(output_dir, 'basic_simulation_position.png') # Save position plot
        fig_pwm_path = os.path.join(output_dir, 'basic_simulation_pwm.png') # Save PWM plot

        fig_pos.savefig(fig_pos_path)
        print(f"Basic simulation position chart saved to: {fig_pos_path}")

        fig_pwm.savefig(fig_pwm_path)
        print(f"Basic simulation PWM duty cycle chart saved to: {fig_pwm_path}")


    # Close figures to free up memory
    plt.close(fig_pos)
    plt.close(fig_pwm)

    # Return system and None for figures/animation as they are saved separately
    return system, None, None, None


def run_linear_analysis(model, args):
    """运行线性系统分析"""
    print("========== Linear System Analysis ==========")

    # 线性化系统
    print("Linearizing system...")
    try:
        model.linearize_system()
    except Exception as e:
        print(f"Error linearizing system: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 分析系统
    print("Analyzing system characteristics...")
    try:
        analysis = model.analyze_system()
    except Exception as e:
        print(f"Error analyzing system characteristics: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 设计状态反馈控制器
    print("Designing state feedback controller...")
    try:
        desired_poles = [float(p) for p in args.desired_poles.split(',')]
        K, sys_cl = model.design_state_feedback(desired_poles)
    except Exception as e:
        print(f"Error designing state feedback controller: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 打印摘要
    try:
        model.print_summary()
    except Exception as e:
        print(f"Error printing system summary: {e}")

    # 解析初始状态
    try:
        # 对于线性模型，需要转换为偏差状态
        x0_values = [float(x) for x in args.x0.split(',')]
        x0_linear = np.array([
            x0_values[0] - model.x_eq,
            x0_values[1] - model.v_eq,
            x0_values[2] - model.P_l_eq,
            x0_values[3] - model.P_r_eq
        ])
    except Exception as e:
        print(f"Error parsing initial state: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 模拟线性系统响应
    print("Simulating linear system response...")
    try:
        linear_response = model.simulate_response(
            x0=x0_linear,
            t_span=(0, args.t_span),
            dt=args.dt
        )
    except Exception as e:
        print(f"Error simulating linear system response: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 绘制图表
    print("Generating charts...")
    try:
        fig_response = plot_response(linear_response, title_prefix="Linear Model")
        fig_state_space = plot_state_space(linear_response)
        fig_root_locus = plot_root_locus(model.sys)
        fig_bode = plot_bode(model.sys)
    except Exception as e:
        print(f"Error drawing charts: {e}")
        traceback.print_exc()
        return linear_response, None, None, None, None

    # 保存图表
    if args.save_figs:
        try:
            save_figure(fig_response, "linear_response.png", args.output_dir)
            save_figure(fig_state_space, "linear_state_space.png", args.output_dir)
            save_figure(fig_root_locus, "root_locus.png", args.output_dir)
            save_figure(fig_bode, "bode_plot.png", args.output_dir)
        except Exception as e:
            print(f"Error saving charts: {e}")

    # 显示图表
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')

    return linear_response, fig_response, fig_state_space, fig_root_locus, fig_bode

def generate_report(args, results=None):
    """生成分析报告"""
    print("========== Generating Analysis Report ==========")

    report_file = os.path.join(args.output_dir, "analysis_report.txt")
    ensure_output_dir(args.output_dir)

    try:
        with open(report_file, 'w') as f:
            f.write("=========================================\n")
            f.write("  气动系统状态空间分析报告\n") # Keep report title in Chinese
            f.write("=========================================\n\n")

            # 参数部分
            f.write("系统参数:\n")
            f.write("-----------------------------------------\n")
            params = build_params(args)
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # 控制器参数
            f.write("控制器参数:\n")
            f.write("-----------------------------------------\n")
            f.write(f"期望极点: {args.desired_poles}\n")
            f.write("\n")

            # 结果部分
            if results:
                f.write("分析结果摘要:\n")
                f.write("-----------------------------------------\n")

                if 'linear_analysis' in results:
                    analysis = results['linear_analysis'].get('analysis', {})
                    f.write("线性系统分析:\n")
                    if analysis:
                        f.write(f"  系统稳定性: {'稳定' if analysis.get('is_stable', False) else '不稳定'}\n")
                        f.write(f"  系统可控性: {'可控' if analysis.get('is_controllable', False) else '不可控'}\n")
                        f.write(f"  系统可观性: {'可观' if analysis.get('is_observable', False) else '不可观'}\n")
                    f.write("\n")

            # 生成的图表列表
            f.write("生成的图表文件:\n")
            f.write("-----------------------------------------\n")
            if args.save_figs and os.path.exists(args.output_dir):
                for filename in os.listdir(args.output_dir):
                    if filename.endswith(('.png', '.pdf', '.jpg')):
                        f.write(f"  {filename}\n")
            f.write("\n")

            # 结束语
            f.write("=========================================\n")
            f.write("分析报告生成完成\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Analysis report saved to: {report_file}")
        return report_file

    except Exception as e:
        print(f"Failed to generate report: {e}")
        traceback.print_exc()
        return None

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()

    print("Program started, parsing command line arguments...")

    # 解析命令行参数
    args = parse_args()

    print(f"Using run mode: {args.mode}")

    # 构建系统参数
    params = build_params(args)

    # 创建状态空间模型
    try:
        model = PneumaticStateSpaceModel(params)
        print("Successfully created state space model")
    except Exception as e:
        print(f"Failed to create state space model: {e}")
        traceback.print_exc()
        return

    # 存储结果
    results = {}

    # 根据模式运行相应的分析
    if args.mode in ['basic', 'all']: # Add basic mode
        try:
            # For basic simulation, we don't need the state space model instance
            # We pass the parameters directly
            system, fig, anim, anim_fig = run_basic_simulation(
                params, args.save_figs, args.output_dir) # Use save_figs for animation
            results['basic_simulation'] = system
        except Exception as e:
            print(f"运行基础物理模拟失败: {e}")
            traceback.print_exc()

    if args.mode in ['linear', 'all']:
        try:
            linear_response, fig_response, fig_state_space, fig_root_locus, fig_bode = \
                run_linear_analysis(model, args)
            if linear_response is not None:
                results['linear_analysis'] = {
                    'response': linear_response,
                    'analysis': model.analyze_system()
                }
        except Exception as e:
            print(f"运行线性分析失败: {e}")
            traceback.print_exc()

    # 生成分析报告
    if args.save_figs:
        try:
            generate_report(args, results)
        except Exception as e:
            print(f"生成分析报告失败: {e}")
            traceback.print_exc()

    # 显示程序运行时间
    end_time = time.time()
    print(f"Program finished, elapsed time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program execution error: {e}")
        traceback.print_exc()
