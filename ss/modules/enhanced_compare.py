#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版线性-非线性模型比较模块
===========================
提供更全面的线性模型与非线性模型比较分析，包括多点测试和可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt

def enhanced_compare_linear_nonlinear(model, test_points=None, t_span=(0, 5), dt=0.01, use_setpoint=False):
    """
    增强版线性-非线性模型比较函数
    
    参数:
        model: PneumaticStateSpaceModel实例
        test_points: 要测试的管道位置点列表，如果为None则使用默认点
        t_span: 模拟时间范围
        dt: 时间步长
        use_setpoint: 是否使用设定点轨迹
        
    返回:
        results: 包含所有测试点误差分析的字典
        comparison_data: 所有测试点的响应数据
    """
    # 如果没有指定测试点，使用均匀分布的点
    if test_points is None:
        # 在管道长度范围内均匀取5个点
        test_points = np.linspace(0.01, model.L_tube-0.01, 5)
    
    all_results = []
    comparison_data = []
    
    # 对每个测试点进行比较
    for position in test_points:
        # 设置初始状态
        x0 = np.array([position, 0, model.P_l_eq, model.P_r_eq])
        
        # 计算线性模型的偏差状态
        x0_linear = np.array([
            position - model.x_eq,
            0,
            0,
            0
        ])
        
        # 运行非线性模拟
        nonlinear_response = model.simulate_nonlinear(
            x0=x0, t_span=t_span, dt=dt, use_setpoint=use_setpoint
        )
        
        # 运行线性模拟
        linear_response = model.simulate_response(
            x0=x0_linear, t_span=t_span, dt=dt, use_setpoint=use_setpoint
        )
        
        # 确保时间向量长度相同
        min_length = min(len(linear_response['time']), len(nonlinear_response['time']))
        
        # 将线性模型响应转换为实际状态值
        adjusted_linear_states = np.zeros((4, min_length))
        adjusted_linear_states[0, :] = linear_response['states'][0, :min_length] + model.x_eq
        adjusted_linear_states[1, :] = linear_response['states'][1, :min_length] + model.v_eq
        adjusted_linear_states[2, :] = linear_response['states'][2, :min_length] + model.P_l_eq
        adjusted_linear_states[3, :] = linear_response['states'][3, :min_length] + model.P_r_eq
        
        # 计算误差
        error_position = adjusted_linear_states[0, :] - nonlinear_response['states'][0, :min_length]
        error_velocity = adjusted_linear_states[1, :] - nonlinear_response['states'][1, :min_length]
        error_P_l = adjusted_linear_states[2, :] - nonlinear_response['states'][2, :min_length]
        error_P_r = adjusted_linear_states[3, :] - nonlinear_response['states'][3, :min_length]
        
        # 计算误差统计
        result = {
            'test_point': position,
            'position_ratio': position / model.L_tube,  # 位置在管道中的比例
            'max_error_position': np.max(np.abs(error_position)),
            'max_error_velocity': np.max(np.abs(error_velocity)),
            'max_error_P_l': np.max(np.abs(error_P_l)),
            'max_error_P_r': np.max(np.abs(error_P_r)),
            'mean_error_position': np.mean(np.abs(error_position)),
            'mean_error_velocity': np.mean(np.abs(error_velocity)),
            'mean_error_P_l': np.mean(np.abs(error_P_l)),
            'mean_error_P_r': np.mean(np.abs(error_P_r)),
            'rms_error_position': np.sqrt(np.mean(np.square(error_position))),
            'normalized_error': np.mean(np.abs(error_position)) / model.L_tube * 100  # 相对管长的百分比误差
        }
        
        all_results.append(result)
        comparison_data.append({
            'test_point': position,
            'time': nonlinear_response['time'][:min_length],
            'nonlinear_position': nonlinear_response['states'][0, :min_length],
            'linear_position': adjusted_linear_states[0, :],
            'error_position': error_position
        })
    
    return all_results, comparison_data

def plot_comparison_results(results, comparison_data, model, save_path=None):
    """
    绘制比较结果的综合图表
    
    参数:
        results: 来自enhanced_compare_linear_nonlinear的误差分析结果
        comparison_data: 来自enhanced_compare_linear_nonlinear的比较数据
        model: PneumaticStateSpaceModel实例
        save_path: 如果指定，将图表保存到此路径
    """
    # 创建一个3x2的图表网格
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 绘制位置误差随测试点变化的曲线
    ax1 = plt.subplot(3, 2, 1)
    test_points = [r['test_point'] for r in results]
    max_errors = [r['max_error_position'] for r in results]
    mean_errors = [r['mean_error_position'] for r in results]
    
    ax1.plot(test_points, max_errors, 'r-o', label='最大位置误差')
    ax1.plot(test_points, mean_errors, 'b-s', label='平均位置误差')
    ax1.axvline(x=model.x_eq, color='g', linestyle='--', label='平衡点')
    ax1.set_xlabel('测试位置 (m)')
    ax1.set_ylabel('位置误差 (m)')
    ax1.set_title('线性模型位置误差随测试点变化')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 绘制规范化误差随位置比例的变化
    ax2 = plt.subplot(3, 2, 2)
    position_ratios = [r['position_ratio'] for r in results]
    normalized_errors = [r['normalized_error'] for r in results]
    
    ax2.plot(position_ratios, normalized_errors, 'k-o')
    ax2.axvline(x=model.x_eq/model.L_tube, color='g', linestyle='--', label='平衡点')
    ax2.set_xlabel('位置比例 (相对管长)')
    ax2.set_ylabel('归一化误差 (%)')
    ax2.set_title('归一化误差随位置比例变化')
    ax2.grid(True)
    
    # 3. 为几个代表性测试点绘制位置响应
    ax3 = plt.subplot(3, 2, 3)
    
    # 选择一些代表性测试点 (如果有很多测试点，选择部分显示)
    if len(comparison_data) > 3:
        indices = [0, len(comparison_data)//2, len(comparison_data)-1]  # 开始、中间、结尾
    else:
        indices = range(len(comparison_data))
    
    for i in indices:
        data = comparison_data[i]
        ax3.plot(data['time'], data['nonlinear_position'], 
                 label=f'非线性模型 ({data["test_point"]:.3f}m)')
        ax3.plot(data['time'], data['linear_position'], '--',
                 label=f'线性模型 ({data["test_point"]:.3f}m)')
    
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('位置 (m)')
    ax3.set_title('线性和非线性模型位置响应对比')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 位置误差随时间变化
    ax4 = plt.subplot(3, 2, 4)
    
    for i in indices:
        data = comparison_data[i]
        ax4.plot(data['time'], data['error_position'], 
                 label=f'误差 ({data["test_point"]:.3f}m)')
    
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('位置误差 (m)')
    ax4.set_title('位置误差随时间变化')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 绘制压力和速度误差随测试点变化
    ax5 = plt.subplot(3, 2, 5)
    max_P_l_errors = [r['max_error_P_l'] for r in results]
    max_P_r_errors = [r['max_error_P_r'] for r in results]
    
    ax5.plot(test_points, max_P_l_errors, 'm-o', label='左气室压力最大误差')
    ax5.plot(test_points, max_P_r_errors, 'c-s', label='右气室压力最大误差')
    ax5.axvline(x=model.x_eq, color='g', linestyle='--', label='平衡点')
    ax5.set_xlabel('测试位置 (m)')
    ax5.set_ylabel('压力误差 (Pa)')
    ax5.set_title('压力误差随测试点变化')
    ax5.legend()
    ax5.grid(True)
    
    # 6. 绘制速度误差随测试点变化
    ax6 = plt.subplot(3, 2, 6)
    max_v_errors = [r['max_error_velocity'] for r in results]
    mean_v_errors = [r['mean_error_velocity'] for r in results]
    
    ax6.plot(test_points, max_v_errors, 'y-o', label='最大速度误差')
    ax6.plot(test_points, mean_v_errors, 'g-s', label='平均速度误差')
    ax6.axvline(x=model.x_eq, color='g', linestyle='--', label='平衡点')
    ax6.set_xlabel('测试位置 (m)')
    ax6.set_ylabel('速度误差 (m/s)')
    ax6.set_title('速度误差随测试点变化')
    ax6.legend()
    ax6.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    return fig

def generate_comparison_report(model, test_points=None, t_span=(0, 5), dt=0.01, use_setpoint=False, save_path=None):
    """
    生成完整的线性-非线性模型比较报告
    
    参数:
        model: PneumaticStateSpaceModel实例
        test_points: 要测试的管道位置点列表
        t_span: 模拟时间范围
        dt: 时间步长
        use_setpoint: 是否使用设定点轨迹
        save_path: 图表保存路径
        
    返回:
        结果摘要和统计数据
    """
    # 执行比较分析
    results, comparison_data = enhanced_compare_linear_nonlinear(
        model, test_points, t_span, dt, use_setpoint
    )
    
    # 绘制比较结果图表
    fig = plot_comparison_results(results, comparison_data, model, save_path)
    
    # 计算汇总统计信息
    avg_max_position_error = np.mean([r['max_error_position'] for r in results])
    avg_mean_position_error = np.mean([r['mean_error_position'] for r in results])
    avg_normalized_error = np.mean([r['normalized_error'] for r in results])
    
    # 找出误差最小的测试点
    min_error_idx = np.argmin([r['mean_error_position'] for r in results])
    best_test_point = results[min_error_idx]['test_point']
    
    # 生成摘要报告
    summary = {
        'average_max_position_error': avg_max_position_error,
        'average_mean_position_error': avg_mean_position_error, 
        'average_normalized_error': avg_normalized_error,
        'best_linearization_point': best_test_point,
        'equilibrium_point': model.x_eq,
        'detailed_results': results
    }
    
    # 打印摘要
    print("\n===== 线性-非线性模型比较摘要报告 =====")
    print(f"平衡点: {model.x_eq:.4f} m")
    print(f"最佳线性化点: {best_test_point:.4f} m")
    print(f"平均最大位置误差: {avg_max_position_error:.6f} m")
    print(f"平均位置误差: {avg_mean_position_error:.6f} m")
    print(f"平均归一化误差: {avg_normalized_error:.2f}%")
    print("=======================================\n")
    
    return summary, fig
