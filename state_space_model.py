#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气动系统状态空间模型及分析
=====================
这个模块从现代控制理论的角度分析系统，进行线性化并设计状态反馈控制器。
提供了系统分析工具和可视化功能。

作者: Claude AI
日期: 2025-04-19
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ctrl
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class PneumaticStateSpaceModel:
    """
    基于状态空间的气动系统分析类
    实现线性化、状态反馈控制和系统特性分析
    """
    
    def __init__(self, params=None):
        """初始化系统参数"""
        # 如果没有提供参数，使用默认参数
        if params is None:
            params = {}
            
        # 系统参数
        self.m_ball = params.get('m_ball', 0.05)      # 小球质量 (kg)
        self.A = params.get('A', 0.001)               # 管道横截面积 (m²)
        self.f_v = params.get('f_v', 0.1)             # 粘性摩擦系数
        self.V_tube = params.get('V_tube', 0.01)      # 管道总体积 (m³)
        self.L_tube = params.get('L_tube', 0.5)       # 管道长度 (m)
        
        # 气动参数
        self.P_0 = params.get('P_0', 101325)          # 参考压力 (Pa)
        self.rho_0 = params.get('rho_0', 1.225)       # 参考密度 (kg/m³)
        self.P_s = params.get('P_s', 500000)          # 供气压力 (Pa)
        self.P_atm = params.get('P_atm', 101325)      # 大气压力 (Pa)
        
        # 阀门参数
        self.C_max = params.get('C_max', 1e-8)        # 最大音速导纳 (m³/(s·Pa))
        
        # 平衡点 (线性化位置)
        self.x_eq = params.get('x_eq', 0.25)           # 平衡位置 (m)
        self.v_eq = 0                                  # 平衡速度 (m/s)
        self.P_l_eq = params.get('P_l_eq', 200000)     # 左气室平衡压力 (Pa)
        self.P_r_eq = params.get('P_r_eq', 200000)     # 右气室平衡压力 (Pa)
        self.s_eq = 0                                  # 阀门平衡位置
        
        # 线性化模型矩阵
        self.A_matrix = None
        self.B_matrix = None
        self.C_matrix = None
        self.D_matrix = None
        self.sys = None
        
        # 控制设计参数
        self.K = None  # 状态反馈增益矩阵
        self.poles = None  # 期望极点
        
    def linearize_system(self):
        """在平衡点处线性化系统并创建状态空间模型"""
        # 状态变量：x = [position, velocity, P_l, P_r]
        # 输入变量：u = [spool_position]
        # 输出变量：y = [position]
        
        # 计算平衡点的体积
        V_l_eq = self.A * self.x_eq
        V_r_eq = self.V_tube - self.A * self.x_eq
        
        # 基于气体状态方程和质量流量关系的常数
        gamma = 1.4  # 气体绝热指数
        R = 287.05   # 气体常数 (J/(kg·K))
        T = 293.15   # 温度 (K)
        
        # 压力-体积变化率常数
        k_l = gamma * self.P_l_eq / V_l_eq
        k_r = gamma * self.P_r_eq / V_r_eq
        
        # A矩阵元素计算（完整的雅可比矩阵）
        
        # 位置对各状态变量的偏导数
        a11 = 0  # ∂(dx/dt)/∂x
        a12 = 1  # ∂(dx/dt)/∂v
        a13 = 0  # ∂(dx/dt)/∂P_l
        a14 = 0  # ∂(dx/dt)/∂P_r
        
        # 速度对各状态变量的偏导数
        a21 = 0  # ∂(dv/dt)/∂x - 在平衡点，压力对位置的影响抵消
        a22 = -self.f_v / self.m_ball  # ∂(dv/dt)/∂v
        a23 = self.A / self.m_ball  # ∂(dv/dt)/∂P_l
        a24 = -self.A / self.m_ball  # ∂(dv/dt)/∂P_r
        
        # 左气室压力对各状态变量的偏导数
        a31 = -k_l * self.A / V_l_eq  # ∂(dP_l/dt)/∂x
        a32 = -k_l * self.A / V_l_eq  # ∂(dP_l/dt)/∂v
        # 考虑气流通过阀门时的压力依赖性
        a33 = -self.C_max * np.sqrt(gamma/(R*T)) * np.sqrt(self.P_l_eq) * (self.P_l_eq > self.P_atm)  # ∂(dP_l/dt)/∂P_l
        a34 = 0  # ∂(dP_l/dt)/∂P_r
        
        # 右气室压力对各状态变量的偏导数
        a41 = k_r * self.A / V_r_eq  # ∂(dP_r/dt)/∂x
        a42 = k_r * self.A / V_r_eq  # ∂(dP_r/dt)/∂v
        a43 = 0  # ∂(dP_r/dt)/∂P_l
        # 考虑气流通过阀门时的压力依赖性
        a44 = -self.C_max * np.sqrt(gamma/(R*T)) * np.sqrt(self.P_r_eq) * (self.P_r_eq > self.P_atm)  # ∂(dP_r/dt)/∂P_r
        
        # 构建A矩阵
        self.A_matrix = np.array([
            [a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44]
        ])
        
        # 构建B矩阵（控制输入矩阵）
        # 阀门位置对各状态变量的影响
        b1 = 0  # 阀门位置不直接影响位置
        b2 = 0  # 阀门位置不直接影响速度
        
        # 阀门位置对压力的影响（基于流量方程）
        # 考虑临界流动和亚临界流动条件
        critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))  # 临界压力比
        
        # 进气时流量增益
        if self.P_l_eq/self.P_s > critical_ratio:  # 亚临界流动
            flow_gain_l = self.C_max * self.P_s * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                         ((self.P_l_eq/self.P_s)**(2/gamma) - (self.P_l_eq/self.P_s)**((gamma+1)/gamma)))
        else:  # 临界流动
            flow_gain_l = self.C_max * self.P_s * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
        
        # 排气时流量增益
        if self.P_r_eq/self.P_s > critical_ratio:  # 亚临界流动
            flow_gain_r = self.C_max * self.P_s * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                         ((self.P_r_eq/self.P_s)**(2/gamma) - (self.P_r_eq/self.P_s)**((gamma+1)/gamma)))
        else:  # 临界流动
            flow_gain_r = self.C_max * self.P_s * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
        
        b3 = flow_gain_l  # 阀门位置对左压力的影响
        b4 = -flow_gain_r  # 阀门位置对右压力的影响（注意符号）
        
        self.B_matrix = np.array([[b1], [b2], [b3], [b4]])
        
        # 构建C矩阵（输出矩阵）
        # 我们只关心小球位置作为输出
        self.C_matrix = np.array([[1, 0, 0, 0]])
        
        # 构建D矩阵（直通矩阵）
        self.D_matrix = np.array([[0]])
        
        # 创建状态空间模型
        self.sys = ctrl.StateSpace(self.A_matrix, self.B_matrix, self.C_matrix, self.D_matrix)
        
        return self.sys
    
    def analyze_system(self):
        """分析系统特性，包括稳定性、可控性和可观性"""
        if self.sys is None:
            self.linearize_system()
        
        # 计算特征值（极点）
        eigenvalues = np.linalg.eigvals(self.A_matrix)
        
        # 检查系统稳定性
        is_stable = np.all(np.real(eigenvalues) < 0)
        
        # 计算可控性矩阵
        C_matrix = ctrl.ctrb(self.A_matrix, self.B_matrix)
        rank_C = np.linalg.matrix_rank(C_matrix)
        is_controllable = (rank_C == self.A_matrix.shape[0])
        
        # 计算可观性矩阵
        O_matrix = ctrl.obsv(self.A_matrix, self.C_matrix)
        rank_O = np.linalg.matrix_rank(O_matrix)
        is_observable = (rank_O == self.A_matrix.shape[0])
        
        # 返回分析结果
        analysis_result = {
            'eigenvalues': eigenvalues,
            'is_stable': is_stable,
            'controllability_matrix': C_matrix,
            'is_controllable': is_controllable,
            'observability_matrix': O_matrix,
            'is_observable': is_observable
        }
        
        return analysis_result
    
    def design_state_feedback(self, desired_poles=None):
        """设计状态反馈控制器"""
        if self.sys is None:
            self.linearize_system()
        
        # 默认期望极点
        if desired_poles is None:
            # 设置期望极点，使系统更稳定和响应更快
            desired_poles = [-2, -2.5, -3, -3.5]
        
        self.poles = desired_poles
        
        # 使用极点配置方法计算状态反馈增益矩阵
        self.K = ctrl.place(self.A_matrix, self.B_matrix, desired_poles)
        
        # 创建闭环系统
        A_cl = self.A_matrix - self.B_matrix @ self.K
        B_cl = self.B_matrix
        C_cl = self.C_matrix
        D_cl = self.D_matrix
        
        # 创建闭环状态空间模型
        self.sys_cl = ctrl.StateSpace(A_cl, B_cl, C_cl, D_cl)
        
        return self.K, self.sys_cl
    
    def simulate_response(self, x0=None, t_span=(0, 5), input_signal=None):
        """模拟系统响应"""
        if self.sys is None:
            self.linearize_system()
        
        # 设置默认初始状态（相对于平衡点的偏移）
        if x0 is None:
            x0 = np.array([0.1, 0, 0, 0])  # 位置偏移0.1m，其他保持在平衡点
        
        # 时间向量
        t = np.linspace(t_span[0], t_span[1], 1000)
        
        # 创建输入信号（如果没有提供）
        if input_signal is None:
            # 默认使用阶跃输入信号
            u = np.ones((1, len(t)))
        else:
            u = input_signal(t)
        
        # 开环系统响应
        if self.K is None:
            # 模拟开环系统
            result = ctrl.forced_response(self.sys, T=t, U=u, X0=x0)
            # 兼容不同版本的python-control库
            if isinstance(result, tuple):
                if len(result) == 3:
                    t_out, y_out, x_out = result
                elif len(result) == 2:  # 较新版本可能只返回两个值
                    response_obj, _ = result
                    t_out = response_obj.time
                    y_out = response_obj.outputs
                    x_out = response_obj.states
            else:  # 最新版本可能返回一个对象
                t_out = result.time
                y_out = result.outputs
                x_out = result.states
        else:
            # 模拟闭环系统（状态反馈）
            # 参考输入
            r = np.zeros_like(u)
            
            # 状态反馈控制系统
            result = ctrl.forced_response(self.sys_cl, T=t, U=r, X0=x0)
            # 兼容不同版本的python-control库
            if isinstance(result, tuple):
                if len(result) == 3:
                    t_out, y_out, x_out = result
                elif len(result) == 2:  # 较新版本可能只返回两个值
                    response_obj, _ = result
                    t_out = response_obj.time
                    y_out = response_obj.outputs
                    x_out = response_obj.states
            else:  # 最新版本可能返回一个对象
                t_out = result.time
                y_out = result.outputs
                x_out = result.states
        
        # 返回仿真结果
        return {
            'time': t_out,
            'states': x_out,
            'output': y_out
        }
    
    def simulate_nonlinear_system(self, x0=None, t_span=(0, 5), input_signal=None):
        """模拟非线性系统响应"""
        if x0 is None:
            x0 = np.array([0.1, 0, 0, 0])  # 位置偏移0.1m，其他保持在平衡点
        
        # 绝对状态值（相对于平衡点）
        x0_abs = np.array([
            x0[0] + self.x_eq,  # 绝对位置
            x0[1] + self.v_eq,  # 绝对速度
            x0[2] + self.P_l_eq,  # 绝对左压力
            x0[3] + self.P_r_eq   # 绝对右压力
        ])
        
        # 时间向量
        t = np.linspace(t_span[0], t_span[1], 1000)
        
        # 创建输入信号（如果没有提供）
        if input_signal is None:
            def step_input(t):
                return np.ones_like(t) * 0.1  # 默认10%阀门开度
            input_signal = step_input
        
        # 非线性系统动力学方程
        def system_dynamics(t, x, u):
            # 状态变量
            position = x[0]
            velocity = x[1]
            P_l = x[2]
            P_r = x[3]
            
            # 控制输入
            spool_position = u(t)
            
            # 体积计算
            V_l = self.A * position
            V_r = self.V_tube - self.A * position
            
            # 限制体积，避免数值问题
            V_l = max(V_l, 1e-6)
            V_r = max(V_r, 1e-6)
            
            # 气动参数
            gamma = 1.4  # 气体绝热指数
            R = 287.05   # 气体常数 (J/(kg·K))
            T = 293.15   # 温度 (K)
            
            # 临界压力比
            critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))
            
            # 计算流量（根据阀门位置和压力）
            # 进气流量（左气室）
            if spool_position > 0:
                if P_l/self.P_s > critical_ratio:  # 亚临界流动
                    q_in_l = spool_position * self.C_max * self.P_s * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                            ((P_l/self.P_s)**(2/gamma) - (P_l/self.P_s)**((gamma+1)/gamma)))
                else:  # 临界流动
                    q_in_l = spool_position * self.C_max * self.P_s * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            else:
                q_in_l = 0
            
            # 排气流量（左气室）
            if spool_position < 0 and P_l > self.P_atm:
                if self.P_atm/P_l > critical_ratio:  # 亚临界流动
                    q_out_l = -spool_position * self.C_max * P_l * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                             ((self.P_atm/P_l)**(2/gamma) - (self.P_atm/P_l)**((gamma+1)/gamma)))
                else:  # 临界流动
                    q_out_l = -spool_position * self.C_max * P_l * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            else:
                q_out_l = 0
            
            # 进气流量（右气室）
            if spool_position < 0:
                if P_r/self.P_s > critical_ratio:  # 亚临界流动
                    q_in_r = -spool_position * self.C_max * self.P_s * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                            ((P_r/self.P_s)**(2/gamma) - (P_r/self.P_s)**((gamma+1)/gamma)))
                else:  # 临界流动
                    q_in_r = -spool_position * self.C_max * self.P_s * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            else:
                q_in_r = 0
            
            # 排气流量（右气室）
            if spool_position > 0 and P_r > self.P_atm:
                if self.P_atm/P_r > critical_ratio:  # 亚临界流动
                    q_out_r = spool_position * self.C_max * P_r * np.sqrt(2*gamma/(R*T*(gamma-1)) * 
                             ((self.P_atm/P_r)**(2/gamma) - (self.P_atm/P_r)**((gamma+1)/gamma)))
                else:  # 临界流动
                    q_out_r = spool_position * self.C_max * P_r * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            else:
                q_out_r = 0
            
            # 状态导数计算
            # 位置导数 = 速度
            dx1_dt = velocity
            
            # 速度导数 = 加速度 = 力/质量
            F_pressure = P_l * self.A - P_r * self.A  # 压力力
            F_friction = -self.f_v * velocity  # 摩擦力
            dx2_dt = (F_pressure + F_friction) / self.m_ball
            
            # 压力导数（基于气体状态方程和质量流量）
            # 左气室压力导数
            mdot_l = q_in_l - q_out_l  # 质量流量净值
            dx3_dt = (gamma * P_l / V_l) * (-self.A * velocity) + (gamma * R * T / V_l) * mdot_l
            
            # 右气室压力导数
            mdot_r = q_in_r - q_out_r  # 质量流量净值
            dx4_dt = (gamma * P_r / V_r) * (self.A * velocity) + (gamma * R * T / V_r) * mdot_r
            
            return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])
        
        # 求解微分方程
        result = solve_ivp(
            lambda t, x: system_dynamics(t, x, input_signal),
            t_span,
            x0_abs,
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # 转换为相对于平衡点的状态
        states_rel = np.zeros_like(result.y)
        states_rel[0] = result.y[0] - self.x_eq
        states_rel[1] = result.y[1] - self.v_eq
        states_rel[2] = result.y[2] - self.P_l_eq
        states_rel[3] = result.y[3] - self.P_r_eq
        
        # 输出 (位置)
        output = result.y[0]
        
        return {
            'time': result.t,
            'states': states_rel,
            'states_abs': result.y,
            'output': output
        }
    
    def plot_response(self, response, title_prefix="线性模型"):
        """绘制系统响应"""
        # 获取响应数据
        t = response['time']
        states = response['states']
        output = response['output']
        
        # 创建图形
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 绘制位置和速度
        axs[0].plot(t, states[0] + self.x_eq, 'b-', label='位置')
        axs[0].axhline(y=self.x_eq, color='r', linestyle='--', label='平衡位置')
        axs[0].set_xlabel('时间 (s)')
        axs[0].set_ylabel('位置 (m)')
        axs[0].set_title(f'{title_prefix}: Ball Position Response')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(t, states[1], 'g-', label='速度')
        axs[1].axhline(y=0, color='r', linestyle='--', label='零速度')
        axs[1].set_xlabel('时间 (s)')
        axs[1].set_ylabel('速度 (m/s)')
        axs[1].set_title(f'{title_prefix}：小球速度响应')
        axs[1].grid(True)
        axs[1].legend()
        
        # 绘制压力
        axs[2].plot(t, states[2] + self.P_l_eq, 'r-', label='左气室压力')
        axs[2].plot(t, states[3] + self.P_r_eq, 'm-', label='右气室压力')
        axs[2].set_xlabel('时间 (s)')
        axs[2].set_ylabel('压力 (Pa)')
        axs[2].set_title(f'{title_prefix}：气室压力响应')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        return fig
    
    def compare_linear_nonlinear(self, x0=None, t_span=(0, 5), input_signal=None):
        """比较线性和非线性模型响应"""
        # 确保系统已线性化
        if self.sys is None:
            self.linearize_system()
        
        # 模拟线性系统响应
        linear_response = self.simulate_response(x0=x0, t_span=t_span, input_signal=input_signal)
        
        # 模拟非线性系统响应
        nonlinear_response = self.simulate_nonlinear_system(x0=x0, t_span=t_span, input_signal=input_signal)
        
        # 创建图形
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        
        # 绘制位置对比
        axs[0].plot(linear_response['time'], linear_response['states'][0] + self.x_eq, 'b-', label='线性模型')
        axs[0].plot(nonlinear_response['time'], nonlinear_response['states_abs'][0], 'r--', label='非线性模型')
        axs[0].axhline(y=self.x_eq, color='k', linestyle=':', label='平衡位置')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('线性与非线性模型位置响应对比')
        axs[0].grid(True)
        axs[0].legend()
        
        # 绘制速度对比
        axs[1].plot(linear_response['time'], linear_response['states'][1], 'b-', label='线性模型')
        axs[1].plot(nonlinear_response['time'], nonlinear_response['states_abs'][1], 'r--', label='非线性模型')
        axs[1].axhline(y=0, color='k', linestyle=':', label='零速度')
        axs[1].set_xlabel('时间 (s)')
        axs[1].set_ylabel('速度 (m/s)')
        axs[1].set_title('线性与非线性模型速度响应对比')
        axs[1].grid(True)
        axs[1].legend()
        
        # 绘制左气室压力对比
        axs[2].plot(linear_response['time'], linear_response['states'][2] + self.P_l_eq, 'b-', label='线性模型')
        axs[2].plot(nonlinear_response['time'], nonlinear_response['states_abs'][2], 'r--', label='非线性模型')
        axs[2].axhline(y=self.P_l_eq, color='k', linestyle=':', label='平衡压力')
        axs[2].set_xlabel('时间 (s)')
        axs[2].set_ylabel('左气室压力 (Pa)')
        axs[2].set_title('线性与非线性模型左气室压力响应对比')
        axs[2].grid(True)
        axs[2].legend()
        
        # 绘制右气室压力对比
        axs[3].plot(linear_response['time'], linear_response['states'][3] + self.P_r_eq, 'b-', label='线性模型')
        axs[3].plot(nonlinear_response['time'], nonlinear_response['states_abs'][3], 'r--', label='非线性模型')
        axs[3].axhline(y=self.P_r_eq, color='k', linestyle=':', label='平衡压力')
        axs[3].set_xlabel('时间 (s)')
        axs[3].set_ylabel('右气室压力 (Pa)')
        axs[3].set_title('线性与非线性模型右气室压力响应对比')
        axs[3].grid(True)
        axs[3].legend()
        
        plt.tight_layout()
        
        # 计算并返回误差分析
        error_position = np.abs(linear_response['states'][0] - nonlinear_response['states'][0])
        error_velocity = np.abs(linear_response['states'][1] - nonlinear_response['states'][1])
        error_P_l = np.abs(linear_response['states'][2] - nonlinear_response['states'][2])
        error_P_r = np.abs(linear_response['states'][3] - nonlinear_response['states'][3])
        
        error_analysis = {
            'max_error_position': np.max(error_position),
            'max_error_velocity': np.max(error_velocity),
            'max_error_P_l': np.max(error_P_l),
            'max_error_P_r': np.max(error_P_r),
            'mean_error_position': np.mean(error_position),
            'mean_error_velocity': np.mean(error_velocity),
            'mean_error_P_l': np.mean(error_P_l),
            'mean_error_P_r': np.mean(error_P_r)
        }
        
        return fig, error_analysis, linear_response, nonlinear_response
    
    def plot_state_space(self, response=None):
        """绘制状态空间轨迹（3D）"""
        # 如果没有提供响应，模拟系统以获取数据
        if response is None:
            x0 = np.array([0.1, 0, 0, 0])  # 默认初始状态
            response = self.simulate_response(x0=x0)
        
        # 获取响应数据
        states = response['states']
        
        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制状态空间轨迹
        ax.plot(states[0] + self.x_eq, states[1], states[2] + self.P_l_eq, 'b-', linewidth=2)
        
        # 标记起始点和终止点
        ax.scatter(states[0, 0] + self.x_eq, states[1, 0], states[2, 0] + self.P_l_eq, 
                  color='green', s=100, label='Initial State')
        ax.scatter(states[0, -1] + self.x_eq, states[1, -1], states[2, -1] + self.P_l_eq, 
                  color='red', s=100, label='Final State')
        
        # 标记平衡点
        ax.scatter(self.x_eq, 0, self.P_l_eq, color='black', s=100, label='Equilibrium Point')
        
        # 设置标签
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_zlabel('Left Chamber Pressure (Pa)')
        ax.set_title('Pneumatic System State Space Trajectory')
        
        # 添加图例
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_root_locus(self):
        """绘制根轨迹图"""
        if self.sys is None:
            self.linearize_system()
        
        # 创建单入单出传递函数
        sys_tf = ctrl.ss2tf(self.sys)
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # 手动计算根轨迹点
        # 获取传递函数的分子分母多项式
        num, den = sys_tf.num[0][0], sys_tf.den[0][0]
        
        # 计算极点和零点
        poles = np.roots(den)
        zeros = np.roots(num)
        
        # 绘制极点和零点
        ax.plot(np.real(poles), np.imag(poles), 'bx', markersize=10, label='极点')
        if len(zeros) > 0:
            ax.plot(np.real(zeros), np.imag(zeros), 'go', markersize=10, label='零点')
        
        # 绘制虚轴
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 设置标题和标签
        ax.set_title('气动系统根轨迹图')
        ax.set_xlabel('实部')
        ax.set_ylabel('虚部')
        ax.grid(True)
        ax.legend()
        
        # 添加稳定性区域指示
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='稳定性边界')
        
        # 设置合适的视图范围
        max_abs_real = max(abs(np.real(poles).min()), abs(np.real(poles).max())) * 1.2
        max_abs_imag = max(abs(np.imag(poles).min()), abs(np.imag(poles).max())) * 1.2
        ax.set_xlim([-max_abs_real, max_abs_real])
        ax.set_ylim([-max_abs_imag, max_abs_imag])
        
        plt.tight_layout()
        return fig
    
    def plot_bode(self):
        """绘制Bode图"""
        if self.sys is None:
            self.linearize_system()
        
        # 创建单入单出传递函数
        sys_tf = ctrl.ss2tf(self.sys)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 手动计算频率点
        omega = np.logspace(-2, 3, 500)
        
        # 手动计算传递函数的频率响应
        num, den = sys_tf.num[0][0], sys_tf.den[0][0]
        s = 1j * omega
        
        # 计算每个频率点的复数传递函数值
        H = np.polyval(num, s) / np.polyval(den, s)
        
        # 计算幅值（dB）和相位（度）
        mag = 20 * np.log10(np.abs(H))
        phase = np.angle(H, deg=True)
        
        # 手动绘制Bode图
        ax1.semilogx(omega, mag)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Bode Magnitude Plot')
        ax1.grid(True, which="both")
        
        ax2.semilogx(omega, phase)
        ax2.set_xlabel('Frequency (rad/s)')
        ax2.set_ylabel('Phase (deg)')
        ax2.set_title('Bode Phase Plot')
        ax2.grid(True, which="both")
        
        plt.suptitle('气动系统 Bode 图')
        plt.tight_layout()
        return fig
        
    def control_design_summary(self):
        """生成控制设计的摘要信息"""
        if self.sys is None:
            self.linearize_system()
        
        # 分析系统
        analysis = self.analyze_system()
        
        # 设计状态反馈（如果尚未设计）
        if self.K is None:
            self.design_state_feedback()
        
        # 创建摘要信息
        summary = {
            'system_matrix_A': self.A_matrix,
            'control_matrix_B': self.B_matrix,
            'output_matrix_C': self.C_matrix,
            'feedthrough_matrix_D': self.D_matrix,
            'eigenvalues': analysis['eigenvalues'],
            'is_stable': analysis['is_stable'],
            'is_controllable': analysis['is_controllable'],
            'is_observable': analysis['is_observable'],
            'state_feedback_gain': self.K,
            'desired_poles': self.poles
        }
        
        return summary
    
    def print_summary(self, summary=None):
        """打印系统分析和控制设计的摘要"""
        if summary is None:
            summary = self.control_design_summary()
        
        print("======== 气动系统状态空间模型分析 ========")
        print("\n系统矩阵 A:")
        print(summary['system_matrix_A'])
        
        print("\n控制矩阵 B:")
        print(summary['control_matrix_B'])
        
        print("\n输出矩阵 C:")
        print(summary['output_matrix_C'])
        
        print("\n直通矩阵 D:")
        print(summary['feedthrough_matrix_D'])
        
        print("\n系统特征值:")
        for i, eig in enumerate(summary['eigenvalues']):
            print(f"  λ{i+1} = {eig}")
        
        print(f"\n系统稳定性: {'稳定' if summary['is_stable'] else '不稳定'}")
        print(f"系统可控性: {'完全可控' if summary['is_controllable'] else '不完全可控'}")
        print(f"系统可观性: {'完全可观' if summary['is_observable'] else '不完全可观'}")
        
        print("\n状态反馈增益矩阵 K:")
        print(summary['state_feedback_gain'])
        
        print("\n期望极点位置:")
        for i, pole in enumerate(summary['desired_poles']):
            print(f"  p{i+1} = {pole}")
        
        # 计算闭环特征值
        A_cl = self.A_matrix - self.B_matrix @ self.K
        eig_cl = np.linalg.eigvals(A_cl)
        
        print("\n闭环系统特征值:")
        for i, eig in enumerate(eig_cl):
            print(f"  λ{i+1} = {eig}")
        
        print("\n====================================")
    
    def print_nonlinear_vs_linear(self):
        """打印非线性与线性模型的详细对比"""
        print("\n======== 非线性与线性模型详细对比 ========")
        
        print("\n【非线性模型数学表达】")
        print("1. 状态方程：")
        print("   dx₁/dt = x₂")
        print("   dx₂/dt = (P_l·A - P_r·A - f_v·x₂) / m")
        print("   dx₃/dt = γ·P_l/V_l·(-A·x₂) + γ·R·T/V_l·ṁ_l")
        print("   dx₄/dt = γ·P_r/V_r·(A·x₂) + γ·R·T/V_r·ṁ_r")
        print("   其中 V_l = A·x₁, V_r = V_tube - A·x₁")
        
        print("\n2. 质量流量：")
        print("   临界压力比: r_cr = (2/(γ+1))^(γ/(γ-1))")
        print("   进气流量（亚临界）: ṁ = C·P_in·√(2γ/(R·T·(γ-1))·((P_out/P_in)^(2/γ) - (P_out/P_in)^((γ+1)/γ)))")
        print("   进气流量（临界）: ṁ = C·P_in·√(γ/(R·T))·(2/(γ+1))^((γ+1)/(2(γ-1)))")
        
        print("\n【线性模型数学表达】")
        print("1. 状态空间表达式：")
        print("   ẋ = Ax + Bu")
        print("   y = Cx + Du")
        
        print("\n2. 线性化雅可比矩阵 A：")
        print("   a₁₁ = 0,           a₁₂ = 1,           a₁₃ = 0,             a₁₄ = 0")
        print("   a₂₁ = 0,           a₂₂ = -f_v/m,      a₂₃ = A/m,           a₂₄ = -A/m")
        print("   a₃₁ = -k_l·A/V_l,  a₃₂ = -k_l·A/V_l,  a₃₃ = -C·√(γ/RT)·√P_l, a₃₄ = 0")
        print("   a₄₁ = k_r·A/V_r,   a₄₂ = k_r·A/V_r,   a₄₃ = 0,             a₄₄ = -C·√(γ/RT)·√P_r")
        
        print("\n3. 控制矩阵 B：")
        print("   b₁ = 0")
        print("   b₂ = 0")
        print("   b₃ = flow_gain_l  (基于流量方程在平衡点处线性化)")
        print("   b₄ = -flow_gain_r (基于流量方程在平衡点处线性化)")
        
        print("\n【主要差异】")
        print("1. 非线性效应：")
        print("   - 非线性模型: 考虑气体的流动状态（临界/亚临界）、体积变化的精确影响")
        print("   - 线性模型: 在平衡点附近线性化，舍弃高阶项")
        
        print("\n2. 有效范围：")
        print("   - 非线性模型: 适用于全工作范围，可精确描述大信号响应")
        print("   - 线性模型: 仅在平衡点附近有效，偏离平衡点越远越不准确")
        
        print("\n3. 分析便利性：")
        print("   - 非线性模型: 难以使用频域分析、难以设计控制器")
        print("   - 线性模型: 可应用成熟的线性系统理论和分析工具")
        
        print("\n4. 计算效率：")
        print("   - 非线性模型: 计算复杂度高，需要数值求解")
        print("   - 线性模型: 线性代数运算，效率高，适合实时控制")
        
        print("\n====================================")


# 使用示例
def run_state_space_demo():
    """运行状态空间分析演示"""
    # 系统参数
    params = {
        'm_ball': 0.05,        # 小球质量 (kg)
        'A': 0.001,            # 管道横截面积 (m²)
        'f_v': 0.1,            # 粘性摩擦系数
        'V_tube': 0.01,        # 管道总体积 (m³)
        'L_tube': 0.5,         # 管道长度 (m)
        'P_s': 500000,         # 供气压力 (Pa)
        'P_atm': 101325,       # 大气压力 (Pa)
        'x_eq': 0.25,          # 平衡位置 (m)
        'P_l_eq': 200000,      # 左气室平衡压力 (Pa)
        'P_r_eq': 200000       # 右气室平衡压力 (Pa)
    }
    
    # 创建状态空间模型
    model = PneumaticStateSpaceModel(params)
    
    # 线性化系统
    model.linearize_system()
    
    # 分析系统
    analysis = model.analyze_system()
    
    # 设计状态反馈控制器
    desired_poles = [-3, -3.5, -4, -4.5]
    K, sys_cl = model.design_state_feedback(desired_poles)
    
    # 打印摘要
    model.print_summary()
    
    # 定义初始条件和时间跨度
    x0 = np.array([0.1, 0, 0, 0])  # 初始位置偏离平衡点0.1m
    t_span = (0, 2)
    
    try:
        # 比较线性和非线性模型
        print("\n===== 线性与非线性模型对比分析 =====")
        fig_comparison, error_analysis, linear_response, nonlinear_response = model.compare_linear_nonlinear(
            x0=x0, t_span=t_span
        )
        
        # 打印误差分析
        print("\n线性与非线性模型误差分析:")
        print(f"  位置最大误差: {error_analysis['max_error_position']:.6f} m")
        print(f"  速度最大误差: {error_analysis['max_error_velocity']:.6f} m/s")
        print(f"  左气室压力最大误差: {error_analysis['max_error_P_l']:.2f} Pa")
        print(f"  右气室压力最大误差: {error_analysis['max_error_P_r']:.2f} Pa")
        print(f"  位置平均误差: {error_analysis['mean_error_position']:.6f} m")
        print(f"  速度平均误差: {error_analysis['mean_error_velocity']:.6f} m/s")
        print(f"  左气室压力平均误差: {error_analysis['mean_error_P_l']:.2f} Pa")
        print(f"  右气室压力平均误差: {error_analysis['mean_error_P_r']:.2f} Pa")
        
        # 绘制线性模型响应
        fig_linear = model.plot_response(linear_response, title_prefix="线性模型")
        
        # 尝试绘制非线性模型响应
        nonlinear_response_for_plot = {
            'time': nonlinear_response['time'],
            'states': nonlinear_response['states'],
            'output': nonlinear_response['output']
        }
        fig_nonlinear = model.plot_response(nonlinear_response_for_plot, title_prefix="非线性模型")
        
        # 绘制状态空间轨迹
        fig_state_space = model.plot_state_space(linear_response)
        
        # 绘制根轨迹
        fig_root_locus = model.plot_root_locus()
        
        # 绘制Bode图
        fig_bode = model.plot_bode()
        
        # 打印更详细的非线性与线性模型对比
        model.print_nonlinear_vs_linear()
        
        plt.show()
        
        return model, linear_response, nonlinear_response, fig_comparison, fig_linear, fig_nonlinear, fig_state_space, fig_root_locus, fig_bode
        
    except Exception as e:
        print(f"运行对比分析时出错: {e}")
        
        # 如果对比失败，至少尝试运行线性模型部分
        print("仅运行线性模型分析...")
        response = model.simulate_response(x0=x0, t_span=t_span)
        fig_response = model.plot_response(response, title_prefix="线性模型")
        fig_state_space = model.plot_state_space(response)
        fig_root_locus = model.plot_root_locus()
        fig_bode = model.plot_bode()
        
        plt.show()
        
        return model, response, fig_response, fig_state_space, fig_root_locus, fig_bode


if __name__ == "__main__":
    model, response, fig_response, fig_state_space, fig_root_locus, fig_bode = run_state_space_demo()
