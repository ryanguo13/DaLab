#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
=========
用于数据可视化、绘制系统响应、相空间轨迹和频域分析图表。

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import control as ctrl

def plot_response(response, title_prefix="系统"):
    """
    绘制系统响应的时间域图表
    
    参数:
    response: 含有时间、状态、输出和输入的响应字典
    title_prefix: 图表标题前缀
    
    返回:
    matplotlib figure对象
    """
    # 提取数据
    time = response['time']
    states = response['states']
    output = response['output']
    control_input = response['input']
    
    # 修正输出和状态的维度
    if output.ndim > 1:
        output = output.flatten()
        
    # 修正状态矩阵的维度
    # 确保状态矩阵是(4, n)形状
    if states.ndim == 1:
        # 如果状态是一维的，将其展开为二维
        states = states.reshape(1, -1)
    
    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{title_prefix}响应", fontsize=16)
    
    # 位置图 (输出)
    axs[0, 0].plot(time, output, 'b-', linewidth=2)
    axs[0, 0].set_title('位置响应')
    axs[0, 0].set_xlabel('时间 (s)')
    axs[0, 0].set_ylabel('位置 (m)')
    axs[0, 0].grid(True)
    
    # 速度图 (状态1)
    axs[0, 1].plot(time, states[1], 'r-', linewidth=2)
    axs[0, 1].set_title('速度响应')
    axs[0, 1].set_xlabel('时间 (s)')
    axs[0, 1].set_ylabel('速度 (m/s)')
    axs[0, 1].grid(True)
    
    # 左右气室压力图 (状态2,3)
    axs[1, 0].plot(time, states[2]/1000, 'g-', linewidth=2, label='左气室')
    axs[1, 0].plot(time, states[3]/1000, 'm-', linewidth=2, label='右气室')
    axs[1, 0].set_title('气室压力响应')
    axs[1, 0].set_xlabel('时间 (s)')
    axs[1, 0].set_ylabel('压力 (kPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 控制输入图
    # 修复维度不匹配问题
    if control_input.ndim > 1:
        control_input = control_input.flatten()
    axs[1, 1].plot(time, control_input, 'k-', linewidth=2)
    axs[1, 1].set_title('控制输入')
    axs[1, 1].set_xlabel('时间 (s)')
    axs[1, 1].set_ylabel('阀门位置')
    axs[1, 1].grid(True)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def plot_state_space(response):
    """
    绘制状态空间轨迹
    
    参数:
    response: 含有状态的响应字典
    
    返回:
    matplotlib figure对象
    """
    # 提取数据
    states = response['states']
    
    # 修正状态矩阵的维度
    if states.ndim == 1:
        states = states.reshape(1, -1)
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 检查状态数据是否有效
    if states.shape[1] > 0:  # 确保状态数据不为空
        # 绘制3D轨迹 (位置、速度、左气室压力)
        ax.plot3D(states[0], states[1], states[2]/1000, 'blue', linewidth=2)
        
        # 添加起点和终点标记
        ax.scatter(states[0, 0], states[1, 0], states[2, 0]/1000, color='green', s=100, label='起点')
        ax.scatter(states[0, -1], states[1, -1], states[2, -1]/1000, color='red', s=100, label='终点')
    else:
        # 数据为空时，不绘制轨迹，只显示提示信息
        ax.text(0, 0, 0, "无有效状态数据", fontsize=14)
    
    # 设置标签和标题
    ax.set_xlabel('位置 (m)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_zlabel('左气室压力 (kPa)')
    ax.set_title('状态空间轨迹')
    
    # 添加图例
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    return fig

def plot_root_locus(system):
    """
    绘制系统根轨迹
    
    参数:
    system: 控制系统对象
    
    返回:
    matplotlib figure对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        # 尝试使用root_locus函数绘制根轨迹
        # 在较新版本的control库中，不再支持Plot参数
        ctrl.root_locus(system, grid=True)
        plt.title('系统根轨迹')
        plt.grid(True)
    except Exception as e:
        # 如果出现异常，打印错误信息并创建一个简单的说明图
        print(f"绘制根轨迹时出错: {e}")
        plt.text(0.5, 0.5, '无法绘制根轨迹\n可能是control库版本问题',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontsize=14)
        plt.title('系统根轨迹 (失败)')
    
    return fig

def plot_bode(system):
    """
    绘制系统Bode图
    
    参数:
    system: 控制系统对象
    
    返回:
    matplotlib figure对象
    """
    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('系统Bode图', fontsize=16)
    
    try:
        # 直接使用control库的频率响应函数计算而不是绘图
        freq_resp = ctrl.frequency_response(system, np.logspace(-2, 3, 500))
        
        # 提取幅度和相位数据
        omega = freq_resp.omega
        mag = 20 * np.log10(np.abs(freq_resp.fresp[0][0]))
        phase = np.angle(freq_resp.fresp[0][0], deg=True)
        
        # 手动绘制Bode图
        # 幅度图
        axs[0].semilogx(omega, mag, 'b-')
        axs[0].set_title('幅度图')
        axs[0].set_ylabel('幅度 (dB)')
        axs[0].set_xlabel('频率 (rad/s)')
        axs[0].grid(True, which='both')
        
        # 相位图
        axs[1].semilogx(omega, phase, 'r-')
        axs[1].set_title('相位图')
        axs[1].set_ylabel('相位 (度)')
        axs[1].set_xlabel('频率 (rad/s)')
        axs[1].grid(True, which='both')
    except Exception as e:
        # 如果出现异常，打印错误信息并创建一个简单的说明图
        print(f"绘制Bode图时出错: {e}")
        axs[0].text(0.5, 0.5, '无法绘制Bode图\n可能是control库版本问题',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[0].transAxes, fontsize=14)
        axs[1].text(0.5, 0.5, '无法绘制Bode图\n可能是control库版本问题',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_pid_response(pid_response):
    """
    绘制PID控制响应图
    
    参数:
    pid_response: 含有PID控制响应的字典
    
    返回:
    matplotlib figure对象
    """
    # 提取数据
    time = pid_response['time']
    states = pid_response['states']
    output = pid_response['output']
    control_input = pid_response['input']
    setpoint = pid_response['setpoint']
    
    # 修正输出和状态的维度
    if output.ndim > 1:
        output = output.flatten()
        
    # 修正状态矩阵的维度
    # 确保状态矩阵是(4, n)形状
    if states.ndim == 1:
        # 如果状态是一维的，将其展开为二维
        states = states.reshape(1, -1)
    
    # 创建3x2的子图布局
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 位置跟踪图
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, setpoint, 'r--', linewidth=2, label='设定点')
    ax1.plot(time, output, 'b-', linewidth=2, label='实际位置')
    ax1.set_title('PID控制位置跟踪')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置 (m)')
    ax1.grid(True)
    ax1.legend()
    
    # 跟踪误差图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, setpoint - output, 'g-', linewidth=2)
    ax2.set_title('跟踪误差')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('误差 (m)')
    ax2.grid(True)
    
    # 控制输入图
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, control_input, 'k-', linewidth=2)
    ax3.set_title('控制输入')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('阀门位置')
    ax3.grid(True)
    
    # 气室压力图
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time, states[2]/1000, 'g-', linewidth=2, label='左气室')
    ax4.plot(time, states[3]/1000, 'm-', linewidth=2, label='右气室')
    ax4.set_title('气室压力')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('压力 (kPa)')
    ax4.grid(True)
    ax4.legend()
    
    # 速度图
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, states[1], 'r-', linewidth=2)
    ax5.set_title('速度')
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('速度 (m/s)')
    ax5.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_comparison(linear_response, nonlinear_response, eq_point=None):
    """
    绘制线性和非线性模型比较图
    
    参数:
    linear_response: 线性模型响应
    nonlinear_response: 非线性模型响应
    eq_point: 平衡点参数字典
    
    返回:
    matplotlib figure对象
    """
    # 确保两个响应的时间向量长度相同
    min_length = min(len(linear_response['time']), len(nonlinear_response['time']))
    
    # 提取数据
    time = linear_response['time'][:min_length]
    
    # 修正线性响应的维度
    linear_output = linear_response['output']
    if linear_output.ndim > 1:
        linear_output = linear_output.flatten()
        
    # 确保状态矩阵的维度是正确的
    if linear_response['states'].ndim == 1:
        linear_response['states'] = linear_response['states'].reshape(1, -1)
    if nonlinear_response['states'].ndim == 1:
        nonlinear_response['states'] = nonlinear_response['states'].reshape(1, -1)
    
    # 如果没有提供平衡点，假设为零
    if eq_point is None:
        eq_point = {
            'x_eq': 0,
            'v_eq': 0,
            'P_l_eq': 0,
            'P_r_eq': 0
        }
    
    x_eq = eq_point.get('x_eq', 0)
    v_eq = eq_point.get('v_eq', 0)
    P_l_eq = eq_point.get('P_l_eq', 0)
    P_r_eq = eq_point.get('P_r_eq', 0)
    
    # 调整线性响应，加上平衡点
    linear_position = linear_response['states'][0, :min_length] + x_eq
    linear_velocity = linear_response['states'][1, :min_length] + v_eq
    linear_P_l = linear_response['states'][2, :min_length] + P_l_eq
    linear_P_r = linear_response['states'][3, :min_length] + P_r_eq
    
    # 提取非线性响应
    nonlinear_position = nonlinear_response['states'][0, :min_length]
    nonlinear_velocity = nonlinear_response['states'][1, :min_length]
    nonlinear_P_l = nonlinear_response['states'][2, :min_length]
    nonlinear_P_r = nonlinear_response['states'][3, :min_length]
    
    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('线性与非线性模型比较', fontsize=16)
    
    # 位置对比图
    axs[0, 0].plot(time, linear_position, 'b-', linewidth=2, label='线性模型')
    axs[0, 0].plot(time, nonlinear_position, 'r--', linewidth=2, label='非线性模型')
    axs[0, 0].set_title('位置对比')
    axs[0, 0].set_xlabel('时间 (s)')
    axs[0, 0].set_ylabel('位置 (m)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 速度对比图
    axs[0, 1].plot(time, linear_velocity, 'b-', linewidth=2, label='线性模型')
    axs[0, 1].plot(time, nonlinear_velocity, 'r--', linewidth=2, label='非线性模型')
    axs[0, 1].set_title('速度对比')
    axs[0, 1].set_xlabel('时间 (s)')
    axs[0, 1].set_ylabel('速度 (m/s)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 左气室压力对比图
    axs[1, 0].plot(time, linear_P_l/1000, 'b-', linewidth=2, label='线性模型')
    axs[1, 0].plot(time, nonlinear_P_l/1000, 'r--', linewidth=2, label='非线性模型')
    axs[1, 0].set_title('左气室压力对比')
    axs[1, 0].set_xlabel('时间 (s)')
    axs[1, 0].set_ylabel('压力 (kPa)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # 右气室压力对比图
    axs[1, 1].plot(time, linear_P_r/1000, 'b-', linewidth=2, label='线性模型')
    axs[1, 1].plot(time, nonlinear_P_r/1000, 'r--', linewidth=2, label='非线性模型')
    axs[1, 1].set_title('右气室压力对比')
    axs[1, 1].set_xlabel('时间 (s)')
    axs[1, 1].set_ylabel('压力 (kPa)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig