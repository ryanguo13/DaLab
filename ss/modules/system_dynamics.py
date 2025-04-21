#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统动力学模块
===========
实现气动系统的物理模型和动力学方程。

"""

import numpy as np
from scipy.integrate import solve_ivp

def nonlinear_dynamics(t, x, u, params):
    """
    完整的非线性系统动力学

    参数:
    t: 时间点
    x: 状态向量 [position, velocity, P_l, P_r]
    u: 控制输入
    params: 系统参数字典

    返回:
    状态导数向量 [dx_dt, dv_dt, dP_l_dt, dP_r_dt]
    """
    # 解包参数
    m_ball = params.get('m_ball', 0.05)      # 小球质量 (kg)
    A = params.get('A', 0.001)               # 管道横截面积 (m²)
    f_v = params.get('f_v', 0.1)             # 粘性摩擦系数
    V_tube = params.get('V_tube', 0.01)      # 管道总体积 (m³)
    P_s = params.get('P_s', 500000)          # 供气压力 (Pa)
    P_atm = params.get('P_atm', 101325)      # 大气压力 (Pa)
    C_max = params.get('C_max', 1e-8)        # 最大音速导纳 (m³/(s·Pa))

    # 解包状态变量
    position, velocity, P_l, P_r = x
    
    # 计算当前的气室体积
    V_l = A * position
    V_r = V_tube - A * position
    
    # 防止体积为零或负值（物理上不可能）
    V_l = max(V_l, 1e-6)
    V_r = max(V_r, 1e-6)
    
    # 气体参数
    gamma = 1.4  # 绝热指数
    
    # 计算系统动力学
    
    # 位置导数 = 速度
    dx_dt = velocity
    
    # 速度导数 = 力/质量
    F_pressure = (P_l - P_r) * A  # 压力产生的力
    F_friction = -f_v * velocity  # 摩擦力
    F_total = F_pressure + F_friction
    dv_dt = F_total / m_ball
    
    # 气室压力变化（考虑控制输入和活塞运动）
    
    # 活塞运动引起的压力变化
    dP_l_dt_piston = -gamma * P_l * A * velocity / V_l
    dP_r_dt_piston = gamma * P_r * A * velocity / V_r
    
    # 控制阀引起的压力变化（简化模型）
    valve_position = u if u is not None else 0
    
    # 进气阀流量（当阀门开度为正时）
    if valve_position > 0:
        # 进气到左气室
        dP_l_dt_valve = gamma * valve_position * C_max * (P_s - P_l) / V_l
        # 右气室排气
        dP_r_dt_valve = -gamma * valve_position * C_max * (P_r - P_atm) / V_r
    else:
        # 左气室排气
        dP_l_dt_valve = gamma * valve_position * C_max * (P_l - P_atm) / V_l
        # 进气到右气室
        dP_r_dt_valve = -gamma * valve_position * C_max * (P_s - P_r) / V_r
        
    # 总压力变化率
    dP_l_dt = dP_l_dt_piston + dP_l_dt_valve
    dP_r_dt = dP_r_dt_piston + dP_r_dt_valve
    
    # 返回状态导数向量
    dxdt = [dx_dt, dv_dt, dP_l_dt, dP_r_dt]
    return dxdt

def simulate_nonlinear_system(x0, t_span, dt, input_func, params):
    """
    模拟非线性系统响应

    参数:
    x0: 初始状态
    t_span: 时间范围 (t_start, t_end)
    dt: 时间步长
    input_func: 控制输入函数，形式为 input_func(t, x)
    params: 系统参数

    返回:
    模拟结果字典，包含时间、状态、输出和输入
    """
    print(f"开始非线性系统模拟：初始位置={x0[0]}, 初始速度={x0[1]}")
    # 定义包含输入的系统动力学
    def dynamics_with_input(t, x):
        u = input_func(t, x)
        return nonlinear_dynamics(t, x, u, params)
    
    # 求解微分方程
    try:
        # 确保初始状态是数组而不是列表
        x0_array = np.array(x0, dtype=float)
        
        # 创建时间向量
        t_eval = np.arange(t_span[0], t_span[1], dt)
        print(f"求解微分方程，时间范围：{t_span}，时间步长：{dt}，时间点数：{len(t_eval)}")
        
        solution = solve_ivp(
            dynamics_with_input,
            t_span,
            x0_array,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        print(f"微分方程求解完成，成功计算了 {len(solution.t)} 个时间点")
        
        # 提取结果
        t = solution.t
        states = solution.y
        
        # 计算控制输入
        input_values = np.zeros_like(t)
        for i, (ti, xi) in enumerate(zip(t, states.T)):
            try:
                input_values[i] = float(input_func(ti, xi))
            except:
                # 处理返回非标量值的情况
                input_values[i] = 0.0
                print(f"警告: 时间 {ti} 的控制输入无法转换为标量值，已设为0")
        
        # 提取输出（位置）
        output = states[0, :]  # 位置是第一个状态变量
        
        # 返回结果
        response = {
            'time': t,
            'states': states,
            'output': output,
            'input': input_values
        }
        
        return response
        
    except Exception as e:
        print(f"模拟非线性系统出错: {e}")
        # 返回一个空的响应结构，以避免程序崩溃
        return {
            'time': np.array([]),
            'states': np.zeros((4, 0)),
            'output': np.array([]),
            'input': np.array([])
        }

def linearize_at_point(x_eq, params):
    """
    在指定平衡点处线性化系统

    参数:
    x_eq: 平衡状态 [position, velocity, P_l, P_r]
    params: 系统参数

    返回:
    A, B, C, D 系统矩阵
    """
    # 解包平衡点
    position_eq, velocity_eq, P_l_eq, P_r_eq = x_eq
    
    # 解包参数
    m_ball = params.get('m_ball', 0.05)      # 小球质量 (kg)
    A = params.get('A', 0.001)               # 管道横截面积 (m²)
    f_v = params.get('f_v', 0.1)             # 粘性摩擦系数
    V_tube = params.get('V_tube', 0.01)      # 管道总体积 (m³)
    P_s = params.get('P_s', 500000)          # 供气压力 (Pa)
    P_atm = params.get('P_atm', 101325)      # 大气压力 (Pa)
    C_max = params.get('C_max', 1e-8)        # 最大音速导纳 (m³/(s·Pa))
    
    # 计算平衡点的体积
    V_l_eq = A * position_eq
    V_r_eq = V_tube - A * position_eq
    
    # 气体参数
    gamma = 1.4  # 绝热指数
    
    # 压力-体积变化率常数
    k_l = gamma * P_l_eq / V_l_eq
    k_r = gamma * P_r_eq / V_r_eq
    
    # A矩阵元素计算（完整的雅可比矩阵）
    
    # 位置对各状态变量的偏导数
    a11 = 0  # ∂(dx/dt)/∂x
    a12 = 1  # ∂(dx/dt)/∂v
    a13 = 0  # ∂(dx/dt)/∂P_l
    a14 = 0  # ∂(dx/dt)/∂P_r
    
    # 速度对各状态变量的偏导数
    a21 = 0  # ∂(dv/dt)/∂x
    a22 = -f_v / m_ball  # ∂(dv/dt)/∂v
    a23 = A / m_ball  # ∂(dv/dt)/∂P_l
    a24 = -A / m_ball  # ∂(dv/dt)/∂P_r
    
    # 左气室压力对各状态变量的偏导数
    a31 = -P_l_eq * A / V_l_eq  # ∂(dP_l/dt)/∂x
    a32 = -k_l * A  # ∂(dP_l/dt)/∂v
    a33 = 0  # ∂(dP_l/dt)/∂P_l (简化，实际应考虑泄漏)
    a34 = 0  # ∂(dP_l/dt)/∂P_r
    
    # 右气室压力对各状态变量的偏导数
    a41 = P_r_eq * A / V_r_eq  # ∂(dP_r/dt)/∂x
    a42 = k_r * A  # ∂(dP_r/dt)/∂v
    a43 = 0  # ∂(dP_r/dt)/∂P_l
    a44 = 0  # ∂(dP_r/dt)/∂P_r (简化，实际应考虑泄漏)
    
    # B矩阵元素计算
    # 控制输入对各状态变量导数的影响
    b1 = 0  # ∂(dx/dt)/∂s
    b2 = 0  # ∂(dv/dt)/∂s
    
    # 阀门开度对左气室压力变化率的影响
    b3 = k_l * C_max * (P_s - P_l_eq)  # ∂(dP_l/dt)/∂s
    
    # 阀门开度对右气室压力变化率的影响
    b4 = -k_r * C_max * (P_r_eq - P_atm)  # ∂(dP_r/dt)/∂s
    
    # 构建系统矩阵
    A_matrix = np.array([
        [a11, a12, a13, a14],
        [a21, a22, a23, a24],
        [a31, a32, a33, a34],
        [a41, a42, a43, a44]
    ])
    
    B_matrix = np.array([[b1], [b2], [b3], [b4]])
    
    # 输出矩阵 (我们只关心位置)
    C_matrix = np.array([[1, 0, 0, 0]])
    
    # 直接传递矩阵 (无直接传递)
    D_matrix = np.array([[0]])
    
    return A_matrix, B_matrix, C_matrix, D_matrix