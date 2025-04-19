import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入模块
from simulation_v1 import PneumaticBallSystem
from control_system import PneumaticControlSystem
from state_space_model import PneumaticStateSpaceModel



def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='气动系统仿真程序')

    # 主要模式选择
    parser.add_argument('--mode', type=str, choices=['basic', 'control', 'state_space', 'all'],
                       default='all', help='仿真模式: 基础物理, 控制系统, 状态空间分析, 或全部')

    # 系统参数
    parser.add_argument('--m_ball', type=float, default=0.05, help='小球质量 (kg)')
    parser.add_argument('--A', type=float, default=0.001, help='管道横截面积 (m²)')
    parser.add_argument('--f_v', type=float, default=0.1, help='粘性摩擦系数')
    parser.add_argument('--V_tube', type=float, default=0.01, help='管道总体积 (m³)')
    parser.add_argument('--L_tube', type=float, default=0.5, help='管道长度 (m)')
    parser.add_argument('--P_s', type=float, default=500000, help='供气压力 (Pa)')

    # 初始条件
    parser.add_argument('--x_init', type=float, default=0.1, help='小球初始位置 (m)')
    parser.add_argument('--v_init', type=float, default=0.0, help='小球初始速度 (m/s)')

    # 控制系统参数
    parser.add_argument('--control_type', type=str, default='position',
                       choices=['position', 'pressure'], help='控制类型: 位置或压力控制')
    parser.add_argument('--target_position', type=float, default=0.3, help='目标位置 (m)')
    parser.add_argument('--kp', type=float, default=5.0, help='PID比例增益')
    parser.add_argument('--ki', type=float, default=0.1, help='PID积分增益')
    parser.add_argument('--kd', type=float, default=2.0, help='PID微分增益')

    # 状态空间分析参数
    parser.add_argument('--x_eq', type=float, default=0.25, help='平衡位置 (m)')
    parser.add_argument('--P_l_eq', type=float, default=200000, help='左气室平衡压力 (Pa)')
    parser.add_argument('--P_r_eq', type=float, default=200000, help='右气室平衡压力 (Pa)')

    # 模拟参数
    parser.add_argument('--t_span', type=float, default=5.0, help='模拟时间 (s)')
    parser.add_argument('--save_animation', action='store_true', help='保存动画为GIF文件')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')

    return parser.parse_args()



def ensure_output_dir(dir_path):
    """确保输出目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建输出目录: {dir_path}")



def build_simulation_params(args):
    """根据命令行参数构建模拟参数字典"""
    params = {
        # 系统参数
        'm_ball': args.m_ball,
        'A': args.A,
        'f_v': args.f_v,
        'V_tube': args.V_tube,
        'L_tube': args.L_tube,
        'P_s': args.P_s,

        # 初始条件
        'x_s': args.x_init,
        'x_b': args.x_init,
        'v_b': args.v_init,
        'P_l': 101325,  # 初始为大气压
        'P_r': 101325,  # 初始为大气压

        # 控制参数
        'control_type': args.control_type,
        'target_position': args.target_position,
        'kp': args.kp,
        'ki': args.ki,
        'kd': args.kd,

        # 平衡点参数
        'x_eq': args.x_eq,
        'P_l_eq': args.P_l_eq,
        'P_r_eq': args.P_r_eq,

        # 模拟参数
        't_span': (0, args.t_span)
    }

    return params



def run_basic_simulation(params, save_animation=False, output_dir='results'):
    """运行基础物理模拟"""
    print("========== 运行基础物理模拟 ==========")

    # 设置阀门开度
    params['s_spool'] = 0.5  # 左进右出

    # 创建系统实例
    system = PneumaticBallSystem(params)

    # 运行模拟
    print("正在进行物理模拟...")
    start_time = time.time()
    result = system.simulate()
    sim_time = time.time() - start_time
    print(f"模拟完成，计算用时: {sim_time:.3f} 秒")

    # 绘制结果
    print("绘制结果图...")
    fig = system.plot_results()

    # 保存图表
    if output_dir:
        ensure_output_dir(output_dir)
        fig_path = os.path.join(output_dir, 'basic_simulation_results.png')
        fig.savefig(fig_path)
        print(f"结果图表已保存到: {fig_path}")

    # 创建动画
    print("生成系统动画...")
    anim, anim_fig = system.create_animation()

    # 保存动画
    if save_animation and output_dir:
        anim_path = os.path.join(output_dir, 'basic_simulation_animation.gif')
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=30)
        anim.save(anim_path, writer=writer)
        print(f"动画已保存到: {anim_path}")

    return system, fig, anim, anim_fig



def run_control_simulation(params, save_animation=False, output_dir='results'):
    """运行控制系统模拟"""
    print("========== 运行控制系统模拟 ==========")

    # 创建控制系统实例
    system = PneumaticControlSystem(params)

    # 运行模拟
    print("正在进行控制系统模拟...")
    start_time = time.time()
    system.simulate()
    sim_time = time.time() - start_time
    print(f"模拟完成，计算用时: {sim_time:.3f} 秒")

    # 创建可视化
    print("创建系统可视化...")
    fig, anim = system.create_visualization()

    # 保存动画
    if save_animation and output_dir:
        ensure_output_dir(output_dir)
        anim_path = os.path.join(output_dir, 'control_simulation_animation.gif')
        system.save_animation(anim_path)

    return system, fig, anim



def run_state_space_analysis(params, output_dir='results'):
    """运行状态空间分析"""
    print("========== 运行状态空间分析 ==========")

    # 创建状态空间模型
    model = PneumaticStateSpaceModel(params)

    # 线性化系统
    print("线性化系统...")
    model.linearize_system()

    # 分析系统
    print("分析系统特性...")
    analysis = model.analyze_system()

    # 设计状态反馈控制器
    print("设计状态反馈控制器...")
    desired_poles = [-3, -3.5, -4, -4.5]
    K, sys_cl = model.design_state_feedback(desired_poles)

    # 打印摘要
    print("系统分析摘要:")
    model.print_summary()

    # 模拟系统响应
    print("模拟系统响应...")
    x0 = np.array([0.1, 0, 0, 0])  # 初始位置偏离平衡点0.1m
    response = model.simulate_response(x0=x0, t_span=(0, params['t_span'][1]))

    # 绘制响应
    print("生成图表...")
    fig_response = model.plot_response(response)
    fig_state_space = model.plot_state_space(response)
    fig_root_locus = model.plot_root_locus()
    fig_bode = model.plot_bode()

    # 保存图表
    if output_dir:
        ensure_output_dir(output_dir)
        fig_response.savefig(os.path.join(output_dir, 'state_space_response.png'))
        fig_state_space.savefig(os.path.join(output_dir, 'state_space_trajectory.png'))
        fig_root_locus.savefig(os.path.join(output_dir, 'root_locus.png'))
        fig_bode.savefig(os.path.join(output_dir, 'bode_plot.png'))
        print(f"状态空间分析图表已保存到: {output_dir}")

    return model, response, fig_response, fig_state_space, fig_root_locus, fig_bode



def compare_controllers(params, controllers=None, output_dir='results'):
    """比较不同控制器参数的性能"""
    print("========== 比较不同控制器性能 ==========")

    if controllers is None:
        # 默认比较三组不同的PID参数
        controllers = [
            {'name': '控制器1 (低响应)', 'kp': 2.0, 'ki': 0.05, 'kd': 0.5},
            {'name': '控制器2 (中等响应)', 'kp': 5.0, 'ki': 0.1, 'kd': 2.0},
            {'name': '控制器3 (高响应)', 'kp': 10.0, 'ki': 0.2, 'kd': 5.0}
        ]

    # 存储结果
    position_data = []
    controller_names = []

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 对每个控制器进行模拟
    for controller in controllers:
        # 更新参数
        temp_params = params.copy()
        temp_params['kp'] = controller['kp']
        temp_params['ki'] = controller['ki']
        temp_params['kd'] = controller['kd']

        # 创建控制系统
        system = PneumaticControlSystem(temp_params)

        # 运行模拟
        print(f"模拟 {controller['name']}...")
        system.simulate()

        # 提取位置数据
        position = system.states[0] + system.x_s
        position_data.append(position)
        controller_names.append(controller['name'])

        # 绘制位置轨迹
        ax.plot(system.times, position, label=controller['name'])

    # 添加目标位置线
    ax.axhline(y=params['target_position'], color='r', linestyle='--', label='目标位置')

    # 设置图表
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置 (m)')
    ax.set_title('不同控制器参数下的位置响应比较')
    ax.grid(True)
    ax.legend()

    # 保存图表
    if output_dir:
        ensure_output_dir(output_dir)
        fig_path = os.path.join(output_dir, 'controller_comparison.png')
        fig.savefig(fig_path)
        print(f"控制器比较图表已保存到: {fig_path}")

    return fig, position_data, controller_names



def analyze_parameter_sensitivity(params, param_name, values, output_dir='results'):
    """分析系统对特定参数的敏感性"""
    print(f"========== 分析系统对{param_name}参数的敏感性 ==========")

    # 存储结果
    position_data = []
    value_labels = []

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 对每个参数值进行模拟
    for value in values:
        # 更新参数
        temp_params = params.copy()
        temp_params[param_name] = value

        # 创建控制系统
        system = PneumaticControlSystem(temp_params)

        # 运行模拟
        print(f"模拟 {param_name} = {value}...")
        system.simulate()

        # 提取位置数据
        position = system.states[0] + system.x_s
        position_data.append(position)
        value_labels.append(f"{param_name} = {value}")

        # 绘制位置轨迹
        ax.plot(system.times, position, label=f"{param_name} = {value}")

    # 添加目标位置线
    ax.axhline(y=params['target_position'], color='r', linestyle='--', label='目标位置')

    # 设置图表
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置 (m)')
    ax.set_title(f'系统对{param_name}参数的敏感性分析')
    ax.grid(True)
    ax.legend()

    # 保存图表
    if output_dir:
        ensure_output_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{param_name}_sensitivity.png')
        fig.savefig(fig_path)
        print(f"参数敏感性分析图表已保存到: {fig_path}")

    return fig, position_data, value_labels



def generate_report(results, args, output_dir='results'):
    """生成综合报告"""
    print("========== 生成综合报告 ==========")

    ensure_output_dir(output_dir)
    report_path = os.path.join(output_dir, 'simulation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=================================================\n")
        f.write("            气动系统仿真综合报告\n")
        f.write("=================================================\n\n")

        # 记录仿真参数
        f.write("仿真参数:\n")
        f.write("-------------------------------------------------\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # 记录系统性能指标
        if 'control_system' in results:
            system = results['control_system']
            # 计算性能指标
            position = system.states[0] + system.x_s
            target = system.target_position
            error = position - target

            # 计算上升时间
            rise_mask = (position >= 0.1 * target) & (position <= 0.9 * target)
            if np.any(rise_mask):
                rise_indices = np.where(rise_mask)[0]
                rise_time = system.times[rise_indices[-1]] - system.times[rise_indices[0]]
            else:
                rise_time = "N/A"

            # 计算稳态误差
            if len(position) > 0:
                steady_state_error = abs(position[-1] - target)
            else:
                steady_state_error = "N/A"

            # 计算超调量
            overshoot = max(0, max(position) - target) / target * 100 if target > 0 else "N/A"

            # 记录性能指标
            f.write("控制系统性能指标:\n")
            f.write("-------------------------------------------------\n")
            f.write(f"目标位置: {target} m\n")
            f.write(f"上升时间 (10% - 90%): {rise_time} s\n")
            f.write(f"超调量: {overshoot} %\n")
            f.write(f"稳态误差: {steady_state_error} m\n\n")

        # 记录状态空间分析结果
        if 'state_space_model' in results:
            model = results['state_space_model']

            # 获取系统分析
            analysis = model.analyze_system()

            f.write("状态空间分析结果:\n")
            f.write("-------------------------------------------------\n")
            f.write(f"系统稳定性: {'稳定' if analysis['is_stable'] else '不稳定'}\n")
            f.write(f"系统可控性: {'完全可控' if analysis['is_controllable'] else '不完全可控'}\n")
            f.write(f"系统可观性: {'完全可观' if analysis['is_observable'] else '不完全可观'}\n\n")

            f.write("系统特征值:\n")
            for i, eig in enumerate(analysis['eigenvalues']):
                f.write(f"  λ{i+1} = {eig}\n")
            f.write("\n")

            # 记录状态反馈控制器信息（如果有）
            if model.K is not None:
                f.write("状态反馈控制器:\n")
                f.write(f"K = {model.K}\n")
                f.write(f"期望极点: {model.poles}\n\n")

        # 记录所有生成的图表文件
        f.write("生成的图表文件:\n")
        f.write("-------------------------------------------------\n")
        for file in os.listdir(output_dir):
            if file.endswith('.png') or file.endswith('.gif'):
                f.write(f"- {file}\n")

    print(f"综合报告已生成: {report_path}")
    return report_path



def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 构建模拟参数
    params = build_simulation_params(args)

    # 存储结果
    results = {}

    # 根据模式运行相应的模拟
    if args.mode in ['basic', 'all']:
        system, fig, anim, anim_fig = run_basic_simulation(
            params, args.save_animation, args.output_dir)
        results['basic_simulation'] = system

    if args.mode in ['control', 'all']:
        system, fig, anim = run_control_simulation(
            params, args.save_animation, args.output_dir)
        results['control_system'] = system

    if args.mode in ['state_space', 'all']:
        model, response, fig_response, fig_state_space, fig_root_locus, fig_bode = run_state_space_analysis(
            params, args.output_dir)
        results['state_space_model'] = model

    # 如果是全模式，进行额外分析
    if args.mode == 'all':
        # 比较不同控制器
        fig_compare, position_data, controller_names = compare_controllers(params, output_dir=args.output_dir)

        # 分析系统对小球质量的敏感性
        mass_values = [0.02, 0.05, 0.1]
        fig_mass, mass_data, mass_labels = analyze_parameter_sensitivity(
            params, 'm_ball', mass_values, args.output_dir)

        # 分析系统对摩擦系数的敏感性
        friction_values = [0.05, 0.1, 0.2]
        fig_friction, friction_data, friction_labels = analyze_parameter_sensitivity(
            params, 'f_v', friction_values, args.output_dir)

        # 生成综合报告
        report_path = generate_report(results, args, args.output_dir)

    # 显示图形（除非只保存不显示）
    plt.show()


if __name__ == "__main__":
    # 开始计时
    start_time = time.time()

    # 运行主程序
    main()

    # 结束计时
    end_time = time.time()
    runtime = end_time - start_time

    print(f"程序执行完成，总运行时间: {runtime:.2f} 秒")
