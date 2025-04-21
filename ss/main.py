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
    from modules.visualization import (
        plot_response, 
        plot_state_space, 
        plot_root_locus, 
        plot_bode, 
        plot_pid_response, 
        plot_comparison
    )
    print("成功导入所需模块")
except ImportError as e:
    print(f"模块导入错误: {e}")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='气动系统状态空间分析程序')
    
    # 分析模式选择
    parser.add_argument('--mode', type=str, choices=['linear', 'nonlinear', 'compare', 'pid', 'all'],
                        default='all', help='分析模式: 线性分析, 非线性分析, 比较分析, PID模拟, 或全部')
    
    # 系统参数
    parser.add_argument('--m_ball', type=float, default=0.05, help='小球质量 (kg)')
    parser.add_argument('--A', type=float, default=0.001, help='管道横截面积 (m²)')
    parser.add_argument('--f_v', type=float, default=0.1, help='粘性摩擦系数')
    parser.add_argument('--V_tube', type=float, default=0.01, help='管道总体积 (m³)')
    parser.add_argument('--L_tube', type=float, default=0.5, help='管道长度 (m)')
    parser.add_argument('--P_s', type=float, default=500000, help='供气压力 (Pa)')
    parser.add_argument('--P_atm', type=float, default=101325, help='大气压力 (Pa)')
    
    # 平衡点参数
    parser.add_argument('--x_eq', type=float, default=0.25, help='平衡位置 (m)')
    parser.add_argument('--P_l_eq', type=float, default=200000, help='左气室平衡压力 (Pa)')
    parser.add_argument('--P_r_eq', type=float, default=200000, help='右气室平衡压力 (Pa)')
    
    # 控制器参数
    parser.add_argument('--desired_poles', type=str, default='-3,-3.5,-4,-4.5', 
                        help='期望极点 (逗号分隔)')
    parser.add_argument('--kp', type=float, default=0.3, help='PID控制器比例增益')
    parser.add_argument('--ki', type=float, default=0.0, help='PID控制器积分增益')
    parser.add_argument('--kd', type=float, default=20.0, help='PID控制器微分增益')
    
    # 模拟参数 - 减少模拟时间方便测试
    parser.add_argument('--t_span', type=float, default=2.0, help='模拟时间 (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='时间步长 (s)')
    parser.add_argument('--x0', type=str, default='0.26,0,200000,200000',
                        help='初始状态 (逗号分隔)')
    parser.add_argument('--pid_duration', type=float, default=5.0, 
                        help='PID控制持续时间 (s)')
    parser.add_argument('--use_pid_trajectory', action='store_true', 
                        help='在线性和非线性模型中使用与PID控制相同的轨迹函数')
    
    # 输出参数
    parser.add_argument('--save_figs', action='store_true', help='保存图表')
    parser.add_argument('--output_dir', type=str, default='results', help='图表保存目录')
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
        'x_eq': args.x_eq,
        'P_l_eq': args.P_l_eq,
        'P_r_eq': args.P_r_eq
    }
    
    return params

def ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

def save_figure(fig, filename, output_dir):
    """保存图表到指定目录"""
    if not os.path.exists(output_dir):
        ensure_output_dir(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {filepath}")
    except Exception as e:
        print(f"保存图表失败: {e}")

def run_linear_analysis(model, args):
    """运行线性系统分析"""
    print("========== 线性系统分析 ==========")
    
    # 线性化系统
    print("线性化系统...")
    try:
        model.linearize_system()
    except Exception as e:
        print(f"线性化系统出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 分析系统
    print("分析系统特性...")
    try:
        analysis = model.analyze_system()
    except Exception as e:
        print(f"分析系统特性出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 设计状态反馈控制器
    print("设计状态反馈控制器...")
    try:
        desired_poles = [float(p) for p in args.desired_poles.split(',')]
        K, sys_cl = model.design_state_feedback(desired_poles)
    except Exception as e:
        print(f"设计状态反馈控制器出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 打印摘要
    try:
        model.print_summary()
    except Exception as e:
        print(f"打印系统摘要出错: {e}")
    
    # 解析初始状态
    try:
        # 使用与PID控制相同的初始点
        if args.use_pid_trajectory:
            # 从0开始，与PID控制一致
            x0_linear = np.array([-model.x_eq, 0, 0, 0])
        else:
            # 对于线性模型，需要转换为偏差状态
            x0_values = [float(x) for x in args.x0.split(',')]
            x0_linear = np.array([
                x0_values[0] - model.x_eq,
                x0_values[1] - model.v_eq,
                x0_values[2] - model.P_l_eq,
                x0_values[3] - model.P_r_eq
            ])
    except Exception as e:
        print(f"解析初始状态出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 模拟线性系统响应
    print("模拟线性系统响应...")
    try:
        linear_response = model.simulate_response(
            x0=x0_linear, 
            t_span=(0, args.t_span),
            dt=args.dt,
            use_setpoint=args.use_pid_trajectory
        )
    except Exception as e:
        print(f"模拟线性系统响应出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # 绘制图表
    print("生成图表...")
    try:
        fig_response = plot_response(linear_response, title_prefix="Linear Model")
        fig_state_space = plot_state_space(linear_response)
        fig_root_locus = plot_root_locus(model.sys)
        fig_bode = plot_bode(model.sys)
    except Exception as e:
        print(f"绘制图表出错: {e}")
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
            print(f"保存图表出错: {e}")
    
    # 显示图表
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
        
    return linear_response, fig_response, fig_state_space, fig_root_locus, fig_bode

def run_nonlinear_analysis(model, args):
    """运行非线性系统分析"""
    print("========== 非线性系统分析 ==========")
    
    # 确保线性化已完成（为了状态反馈控制器）
    if model.A_matrix is None:
        try:
            model.linearize_system()
        except Exception as e:
            print(f"线性化系统出错: {e}")
            traceback.print_exc()
            return None, None, None
        
    # 设计状态反馈控制器（如果尚未设计）
    if model.K is None:
        try:
            desired_poles = [float(p) for p in args.desired_poles.split(',')]
            model.design_state_feedback(desired_poles)
        except Exception as e:
            print(f"设计状态反馈控制器出错: {e}")
            traceback.print_exc()
            return None, None, None
    
    # 解析初始状态
    try:
        if args.use_pid_trajectory:
            # 从0开始，与PID控制一致
            x0_values = [0.0, 0, model.P_l_eq, model.P_r_eq]
        else:
            x0_values = [float(x) for x in args.x0.split(',')]
    except Exception as e:
        print(f"解析初始状态出错: {e}")
        traceback.print_exc()
        return None, None, None
    
    # 模拟非线性系统响应
    print("模拟非线性系统响应...")
    try:
        nonlinear_response = model.simulate_nonlinear(
            x0=x0_values,
            t_span=(0, args.t_span),
            dt=args.dt,
            use_setpoint=args.use_pid_trajectory
        )
    except Exception as e:
        print(f"模拟非线性系统响应出错: {e}")
        traceback.print_exc()
        return None, None, None
    
    # 绘制图表
    print("生成图表...")
    try:
        fig_response = plot_response(nonlinear_response, title_prefix="Nonlinear Model")
        fig_state_space = plot_state_space(nonlinear_response)
    except Exception as e:
        print(f"绘制图表出错: {e}")
        traceback.print_exc()
        return nonlinear_response, None, None
    
    # 保存图表
    if args.save_figs:
        try:
            save_figure(fig_response, "nonlinear_response.png", args.output_dir)
            save_figure(fig_state_space, "nonlinear_state_space.png", args.output_dir)
        except Exception as e:
            print(f"保存图表出错: {e}")
    
    # 显示图表
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
        
    return nonlinear_response, fig_response, fig_state_space

def run_comparison_analysis(model, args):
    """运行线性与非线性系统比较分析"""
    print("========== 线性与非线性系统比较分析 ==========")
    
    # 确保线性化已完成
    if model.A_matrix is None:
        try:
            model.linearize_system()
        except Exception as e:
            print(f"线性化系统出错: {e}")
            traceback.print_exc()
            return None, None, None, None
    
    # 解析初始状态
    try:
        if args.use_pid_trajectory:
            # 从0开始，与PID控制一致
            x0_values = [0.0, 0, model.P_l_eq, model.P_r_eq]
        else:
            x0_values = [float(x) for x in args.x0.split(',')]
    except Exception as e:
        print(f"解析初始状态出错: {e}")
        traceback.print_exc()
        return None, None, None, None
    
    # 比较线性和非线性模型
    print("比较线性和非线性模型响应...")
    try:
        error_analysis, linear_response, nonlinear_response = model.compare_linear_nonlinear(
            x0=np.array(x0_values),
            t_span=(0, args.t_span),
            dt=args.dt,
            use_setpoint=args.use_pid_trajectory
        )
    except Exception as e:
        print(f"比较线性和非线性模型出错: {e}")
        traceback.print_exc()
        return None, None, None, None
    
    # 打印误差分析
    try:
        model.print_nonlinear_vs_linear()
    except Exception as e:
        print(f"打印误差分析出错: {e}")
    
    # 绘制比较图表
    print("生成比较图表...")
    try:
        fig_comparison = plot_comparison(
            linear_response, 
            nonlinear_response,
            {
                'x_eq': model.x_eq,
                'v_eq': model.v_eq,
                'P_l_eq': model.P_l_eq,
                'P_r_eq': model.P_r_eq
            }
        )
    except Exception as e:
        print(f"绘制比较图表出错: {e}")
        traceback.print_exc()
        return error_analysis, linear_response, nonlinear_response, None
    
    # 保存图表
    if args.save_figs:
        try:
            save_figure(fig_comparison, "linear_nonlinear_comparison.png", args.output_dir)
        except Exception as e:
            print(f"保存图表出错: {e}")
    
    # 显示图表
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
        
    return error_analysis, linear_response, nonlinear_response, fig_comparison

def run_pid_simulation(model, args):
    """运行PID控制模拟"""
    print("========== PID控制模拟 ==========")
    
    # 确保线性化已完成
    if model.A_matrix is None:
        try:
            model.linearize_system()
        except Exception as e:
            print(f"线性化系统出错: {e}")
            traceback.print_exc()
            return None, None
    
    # 模拟PID控制响应
    print(f"模拟PID控制，持续时间: {args.pid_duration}秒...")
    print(f"PID控制将从位置0开始，目标最大位置: 0.156m")
    try:
        pid_response = model.simulate_pid_control(
            t_span=(0, args.pid_duration),
            dt=args.dt,
            kp=args.kp,
            ki=args.ki,
            kd=args.kd
        )
    except Exception as e:
        print(f"模拟PID控制出错: {e}")
        traceback.print_exc()
        return None, None
    
    # 绘制PID控制响应图表
    print("生成PID控制图表...")
    try:
        fig_pid = plot_pid_response(pid_response)
    except Exception as e:
        print(f"绘制PID控制图表出错: {e}")
        traceback.print_exc()
        return pid_response, None
    
    # 保存图表
    if args.save_figs:
        try:
            save_figure(fig_pid, "pid_control_response.png", args.output_dir)
        except Exception as e:
            print(f"保存图表出错: {e}")
    
    # 显示图表
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
        
    return pid_response, fig_pid

def generate_report(args, results=None):
    """生成分析报告"""
    print("========== 生成分析报告 ==========")
    
    report_file = os.path.join(args.output_dir, "analysis_report.txt")
    ensure_output_dir(args.output_dir)
    
    try:
        with open(report_file, 'w') as f:
            f.write("=========================================\n")
            f.write("  气动系统状态空间分析报告\n")
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
            f.write(f"PID参数: Kp={args.kp}, Ki={args.ki}, Kd={args.kd}\n")
            f.write("\n")
            
            # 模拟参数
            f.write("模拟参数:\n")
            f.write("-----------------------------------------\n")
            f.write(f"模拟时间: {args.t_span} s\n")
            f.write(f"时间步长: {args.dt} s\n")
            f.write(f"初始状态: {args.x0}\n")
            f.write(f"PID控制持续时间: {args.pid_duration} s\n")
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
                
                if 'comparison' in results:
                    error_analysis = results['comparison'].get('error_analysis', {})
                    f.write("线性与非线性比较:\n")
                    if error_analysis:
                        f.write(f"  位置最大误差: {error_analysis.get('max_error_position', 0):.6f} m\n")
                        f.write(f"  位置平均误差: {error_analysis.get('mean_error_position', 0):.6f} m\n")
                    f.write("\n")
                    
                if 'pid_simulation' in results:
                    f.write("PID控制模拟:\n")
                    f.write("  已生成PID控制响应图表\n")
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
            
        print(f"分析报告已保存到: {report_file}")
        return report_file
    
    except Exception as e:
        print(f"生成报告失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()
    
    print("程序开始运行，开始解析命令行参数...")
    
    # 解析命令行参数
    args = parse_args()
    
    print(f"使用的运行模式: {args.mode}")
    
    # 构建系统参数
    params = build_params(args)
    
    # 创建状态空间模型
    try:
        model = PneumaticStateSpaceModel(params)
        print("成功创建状态空间模型")
    except Exception as e:
        print(f"创建状态空间模型失败: {e}")
        traceback.print_exc()
        return
    
    # 存储结果
    results = {}
    
    # 根据模式运行相应的分析
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
    
    if args.mode in ['nonlinear', 'all']:
        try:
            nonlinear_response, fig_response, fig_state_space = \
                run_nonlinear_analysis(model, args)
            if nonlinear_response is not None:
                results['nonlinear_analysis'] = {
                    'response': nonlinear_response
                }
        except Exception as e:
            print(f"运行非线性分析失败: {e}")
            traceback.print_exc()
    
    if args.mode in ['compare', 'all']:
        try:
            error_analysis, linear_response, nonlinear_response, fig_comparison = \
                run_comparison_analysis(model, args)
            if error_analysis is not None:
                results['comparison'] = {
                    'error_analysis': error_analysis,
                    'linear_response': linear_response,
                    'nonlinear_response': nonlinear_response
                }
        except Exception as e:
            print(f"运行比较分析失败: {e}")
            traceback.print_exc()
    
    if args.mode in ['pid', 'all']:
        print("开始运行PID控制模拟...")
        try:
            pid_response, fig_pid = run_pid_simulation(model, args)
            print("PID控制模拟完成")
            if pid_response is not None:
                results['pid_simulation'] = {
                    'response': pid_response
                }
                print("成功保存PID模拟结果")
            else:
                print("警告: PID模拟返回了空结果")
        except Exception as e:
            print(f"运行PID控制模拟失败: {e}")
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
    print(f"程序运行完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
