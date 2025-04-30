#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态空间模型模块
==============
实现气动系统的状态空间模型，包括线性化、可控性分析和状态反馈控制器设计。

"""

import numpy as np
import control as ctrl
from .system_dynamics import nonlinear_dynamics, simulate_nonlinear_system

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
            
        # 系统参数 - 修改默认值以匹配15cm管道和9mm球体
        self.m_ball = params.get('m_ball', 0.003)     # 小球质量 (kg) - 9mm球的估计质量
        self.A = params.get('A', 0.0001)              # 管道横截面积 (m²) - 适合9mm球
        self.f_v = params.get('f_v', 0.05)            # 粘性摩擦系数 - 针对短管径小球优化
        self.V_tube = params.get('V_tube', 0.0015)    # 管道总体积 (m³) - 15cm管道
        self.L_tube = params.get('L_tube', 0.15)      # 管道长度 (m) - 15cm
        self.r_ball = np.sqrt(self.A / np.pi)         # 计算小球半径 (m)
        
        # 气动参数
        self.P_0 = params.get('P_0', 101325)          # 参考压力 (Pa)
        self.rho_0 = params.get('rho_0', 1.225)       # 参考密度 (kg/m³)
        self.P_s = params.get('P_s', 500000)          # 供气压力 (Pa)
        self.P_atm = params.get('P_atm', 101325)      # 大气压力 (Pa)
        
        # 阀门参数
        self.C_max = params.get('C_max', 1e-8)        # 最大音速导纳 (m³/(s·Pa))
        
        # 平衡点 (线性化位置) - 设为管道中点
        self.x_eq = params.get('x_eq', 0.075)         # 平衡位置 (m) - 设为15cm管道的中点
        self.v_eq = 0                                 # 平衡速度 (m/s)
        self.P_l_eq = params.get('P_l_eq', 180000)    # 左气室平衡压力 (Pa) - 针对短管道优化
        self.P_r_eq = params.get('P_r_eq', 180000)    # 右气室平衡压力 (Pa) - 针对短管道优化
        self.s_eq = 0                                 # 阀门平衡位置
        
        # 线性化模型矩阵
        self.A_matrix = None
        self.B_matrix = None
        self.C_matrix = None
        self.D_matrix = None
        self.sys = None
        
        # 控制设计参数
        self.K = None  # 状态反馈增益矩阵
        self.poles = None  # 期望极点
        
        # 保存所有系统参数，供非线性模拟使用
        self.params = {
            'm_ball': self.m_ball,
            'A': self.A,
            'f_v': self.f_v,
            'V_tube': self.V_tube,
            'L_tube': self.L_tube,
            'P_0': self.P_0,
            'rho_0': self.rho_0,
            'P_s': self.P_s,
            'P_atm': self.P_atm,
            'C_max': self.C_max
        }
        
    def linearize_system(self, leakage_coeff=0.015, ball_clearance=0.00005):
        """增强版线性化：在平衡点处线性化系统并创建状态空间模型，包含泄漏效应和球体间隙
        
        参数:
            leakage_coeff: 泄漏系数，影响压力衰减率 (优化为0.015)
            ball_clearance: 球体与管壁间隙(m)，影响气室间的气体交换 (优化为0.00005)
        """
        # 状态变量：x = [position, velocity, P_l, P_r]
        # 输入变量：u = [spool_position]
        # 输出变量：y = [position]
        
        # 适应15cm管道长度的几何计算
        ball_diameter = 0.009  # 球直径9mm
        
        # 计算有效截面积（考虑球体直径）
        effective_area = self.A * (1.0 - (ball_diameter/2)**2 / (self.A/np.pi))
        area_ratio = effective_area / self.A
        
        # 计算平衡点的体积 - 考虑球体体积
        ball_volume = (4/3) * np.pi * (ball_diameter/2)**3
        V_l_eq = self.A * self.x_eq - ball_volume/2
        V_r_eq = self.A * (self.L_tube - self.x_eq) - ball_volume/2
        
        # 基于气体状态方程和质量流量关系的常数
        gamma = 1.4  # 气体绝热指数
        R = 287.05   # 气体常数 (J/(kg·K))
        T = 293.15   # 温度 (K)
        
        # 压力-体积变化率常数
        k_l = gamma * self.P_l_eq / V_l_eq
        k_r = gamma * self.P_r_eq / V_r_eq
        
        # A矩阵元素计算（增强版雅可比矩阵）
        
        # 位置对各状态变量的偏导数
        a11 = 0  # ∂(dx/dt)/∂x
        a12 = 1  # ∂(dx/dt)/∂v
        a13 = 0  # ∂(dx/dt)/∂P_l
        a14 = 0  # ∂(dx/dt)/∂P_r
        
        # 速度对各状态变量的偏导数 - 使用有效面积
        a21 = 0  # ∂(dv/dt)/∂x
        a22 = -self.f_v / self.m_ball  # ∂(dv/dt)/∂v
        
        # 使用有效面积计算压力对速度的影响
        a23 = effective_area / self.m_ball  # ∂(dv/dt)/∂P_l
        a24 = -effective_area / self.m_ball  # ∂(dv/dt)/∂P_r
        
        # 左气室压力对各状态变量的偏导数
        a31 = -self.P_l_eq * self.A / V_l_eq  # ∂(dP_l/dt)/∂x
        a32 = -k_l * self.A  # ∂(dP_l/dt)/∂v
        
        # 添加气体泄漏效应
        a33 = -leakage_coeff * self.P_l_eq / self.P_0  # 左气室压力泄漏
        
        # 气室间气体交换（如果球体不完全密封）
        coupling_factor = ball_clearance * self.A * np.sqrt(k_l * k_r)
        a34 = coupling_factor * (self.P_r_eq - self.P_l_eq) / self.P_0  # 右气室压力影响左气室
        
        # 右气室压力对各状态变量的偏导数
        a41 = self.P_r_eq * self.A / V_r_eq  # ∂(dP_r/dt)/∂x
        a42 = k_r * self.A  # ∂(dP_r/dt)/∂v
        
        # 气室间气体交换
        a43 = coupling_factor * (self.P_l_eq - self.P_r_eq) / self.P_0  # 左气室压力影响右气室
        a44 = -leakage_coeff * self.P_r_eq / self.P_0  # 右气室压力泄漏
        
        # B矩阵元素计算 - 增强阀门模型
        # 控制输入对各状态变量导数的影响
        b1 = 0  # ∂(dx/dt)/∂s
        b2 = 0  # ∂(dv/dt)/∂s
        
        # 增强阀门模型：使用非线性效应的线性近似
        valve_nonlinearity = 1.2  # 阀门非线性修正因子 
        
        # 阀门开度对左气室压力变化率的影响
        b3 = valve_nonlinearity * k_l * self.C_max * (self.P_s - self.P_l_eq)  # ∂(dP_l/dt)/∂s
        
        # 阀门开度对右气室压力变化率的影响
        b4 = -valve_nonlinearity * k_r * self.C_max * (self.P_r_eq - self.P_atm)  # ∂(dP_r/dt)/∂s
        
        # 构建状态空间矩阵
        self.A_matrix = np.array([
            [a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44]
        ])
        
        self.B_matrix = np.array([[b1], [b2], [b3], [b4]])
        
        # 输出矩阵 (我们只关心位置)
        self.C_matrix = np.array([[1, 0, 0, 0]])
        
        # 直接传递矩阵 (无直接传递)
        self.D_matrix = np.array([[0]])
        
        # 创建线性状态空间模型
        self.sys = ctrl.ss(self.A_matrix, self.B_matrix, self.C_matrix, self.D_matrix)
        
        return self.A_matrix, self.B_matrix, self.C_matrix, self.D_matrix
    
    def analyze_system(self):
        """分析系统特性：可控性、可观性、稳定性和极点"""
        if self.sys is None:
            self.linearize_system()
            
        # 计算特征值（极点）
        eigenvalues, _ = np.linalg.eig(self.A_matrix)
        
        # 检查可控性和可观性
        ctrb_matrix = ctrl.ctrb(self.A_matrix, self.B_matrix)
        obsv_matrix = ctrl.obsv(self.A_matrix, self.C_matrix)
        
        ctrb_rank = np.linalg.matrix_rank(ctrb_matrix)
        obsv_rank = np.linalg.matrix_rank(obsv_matrix)
        
        # 判断系统是否可控和可观
        is_controllable = (ctrb_rank == self.A_matrix.shape[0])
        is_observable = (obsv_rank == self.A_matrix.shape[0])
        
        # 计算系统稳定性
        is_stable = all(np.real(eigenvalues) < 0)
        
        # 计算传递函数和系统极零点
        tf_sys = ctrl.ss2tf(self.sys)
        
        # 在新版本的control库中，使用zeros和poles
        try:
            # 尝试使用zeros函数获取零点
            zeros = ctrl.zeros(tf_sys)
        except AttributeError:
            # 如果zeros函数不存在，使用旧版本的zero函数
            try:
                zeros = ctrl.zero(tf_sys)
            except AttributeError:
                # 如果两个函数都不存在，就设置为空数组
                zeros = np.array([])
                print("警告: 无法计算系统零点，control库可能缺少相应函数")
        
        # 获取极点 - 使用poles而不是pole
        try:
            poles = ctrl.poles(tf_sys)
        except AttributeError:
            try:
                poles = ctrl.pole(tf_sys)
            except AttributeError:
                # 如果两个函数都不存在，使用特征值
                poles = eigenvalues
                print("警告: 无法计算系统极点，使用特征值代替")
        
        # 计算DC增益
        dc_gain = ctrl.dcgain(tf_sys)
        
        # 返回分析结果
        results = {
            'eigenvalues': eigenvalues,
            'is_controllable': is_controllable,
            'is_observable': is_observable,
            'is_stable': is_stable,
            'zeros': zeros,
            'poles': poles,
            'dc_gain': dc_gain,
            'ctrb_rank': ctrb_rank,
            'obsv_rank': obsv_rank,
            'system_order': self.A_matrix.shape[0]
        }
        
        return results
    
    def design_state_feedback(self, desired_poles=None):
        """设计状态反馈控制器"""
        if self.sys is None:
            self.linearize_system()
            
        # 如果没有指定期望极点，使用默认值
        if desired_poles is None:
            # 默认设置为A矩阵特征值实部的2倍（更快的响应）
            eigenvalues = np.linalg.eigvals(self.A_matrix)
            # 对于不稳定的极点，将其设为稳定
            desired_poles = [min(2 * np.real(ev), -1) + 1j * np.imag(ev) for ev in eigenvalues]
        
        # 保存期望极点
        self.poles = desired_poles
        
        # 计算反馈增益矩阵
        self.K = ctrl.place(self.A_matrix, self.B_matrix, desired_poles)
        
        # 创建闭环系统
        A_cl = self.A_matrix - self.B_matrix @ self.K
        sys_cl = ctrl.ss(A_cl, self.B_matrix, self.C_matrix, self.D_matrix)
        
        return self.K, sys_cl
    
    def simulate_response(self, x0=None, t_span=(0, 10), dt=0.01, input_signal=None):
        """模拟线性系统响应
        
        参数:
            x0: 初始状态 (偏差状态)
            t_span: 模拟时间范围 (s)
            dt: 时间步长 (s)
            input_signal: 输入信号数组 (与时间向量长度相同)
        """
        
        if self.sys is None:
            self.linearize_system()
            
        # 如果没有指定初始状态，则使用零偏移
        if x0 is None:
            x0 = np.zeros(self.A_matrix.shape[0])
            
        # 生成时间向量
        t = np.arange(t_span[0], t_span[1], dt)
        
        # 如果没有指定输入信号，使用零输入
        if input_signal is None:
             U = np.zeros_like(t)
        else:
             U = input_signal
        
        # 如果已设计状态反馈控制器，使用闭环系统
        if self.K is not None:
            # 计算闭环系统矩阵
            A_cl = self.A_matrix - self.B_matrix @ self.K
            sys_cl = ctrl.ss(A_cl, self.B_matrix, self.C_matrix, self.D_matrix)
            
            try:
                # 尝试使用新版本的forced_response函数（返回2个值）
                response = ctrl.forced_response(sys_cl, T=t, U=U, X0=x0)
                # 新版本的control库，response是一个对象，包含time, outputs, states等属性
                y_cl = response.y
                x_cl = response.x
            except ValueError:
                try:
                    # 尝试使用旧版本的forced_response函数（返回3个值）
                    _, y_cl, x_cl = ctrl.forced_response(sys_cl, T=t, U=U, X0=x0)
                except Exception as e:
                    print(f"模拟系统响应出错: {e}")
                    # 返回空的响应结构
                    return {
                        'time': t,
                        'states': np.zeros((4, len(t))),
                        'output': np.zeros(len(t)),
                        'input': np.zeros(len(t))
                    }
            
            # 计算控制输入
            u_cl = -np.dot(self.K, x_cl)
            
            response = {
                'time': t,
                'states': x_cl,
                'output': y_cl,
                'input': u_cl,
                'setpoint': None # Remove setpoint
            }
        else:
            # 模拟开环系统响应
            try:
                # 尝试使用新版本的forced_response函数
                response = ctrl.forced_response(self.sys, T=t, U=U, X0=x0)
                # 新版本的control库，response是一个对象
                y = response.y
                x = response.x
            except ValueError:
                try:
                    # 尝试使用旧版本的forced_response函数
                    _, y, x = ctrl.forced_response(self.sys, T=t, U=U, X0=x)
                except Exception as e:
                    print(f"模拟系统响应出错: {e}")
                    # 返回空的响应结构
                    return {
                        'time': t,
                        'states': np.zeros((4, len(t))),
                        'output': np.zeros(len(t)),
                        'input': np.zeros(len(t))
                    }
            
            response = {
                'time': t,
                'states': x,
                'output': y,
                'input': U,
                'setpoint': None # Remove setpoint
            }
            
        return response
    
    def simulate_nonlinear(self, x0=None, t_span=(0, 10), dt=0.005, input_func=None):
        """模拟非线性系统响应
        优化参数:
            dt: 时间步长从0.01优化为0.005以提高数值稳定性
        """
        
        # 如果没有指定初始状态，则始终从平衡点开始
        if x0 is None:
            x0 = [self.x_eq, 0, self.P_l_eq, self.P_r_eq]
            
        # 如果没有指定输入函数，则使用零输入
        if input_func is None:
            def input_func(t, x):
                return 0.0
        
        # 使用通用非线性模拟函数
        response = simulate_nonlinear_system(x0, t_span, dt, input_func, self.params)
        
        # Remove setpoint logic
        # if use_setpoint:
        #     # 计算并添加设定点到响应中
        #     response['setpoint'] = np.array([generate_setpoint_curve(t, x_min=0.0, x_max=0.156) for t in response['time']])
        
        return response
    
    def compare_linear_nonlinear(self, x0=None, t_span=(0, 10), dt=0.01):
        """比较线性和非线性模型的响应"""
        # 如果没有指定初始状态，则从平衡点附近开始
        if x0 is None:
            x0 = np.array([self.x_eq + 0.01, 0, self.P_l_eq, self.P_r_eq])
            x0_linear = np.array([0.01, 0, 0, 0])  # 线性模型使用偏差状态
        else:
            # 为线性模型计算偏差状态
            x0_linear = np.array([
                x0[0] - self.x_eq,
                x0[1] - self.v_eq,
                x0[2] - self.P_l_eq,
                x0[3] - self.P_r_eq
            ])
        
        # 模拟线性和非线性系统响应，使用零输入
        linear_response = self.simulate_response(x0=x0_linear, t_span=t_span, dt=dt, input_signal=np.zeros_like(np.arange(t_span[0], t_span[1], dt)))
        nonlinear_response = self.simulate_nonlinear(x0=x0, t_span=t_span, dt=dt, input_func=lambda t, x: 0.0)
        
        # 确保两个响应的时间向量长度相同
        min_length = min(len(linear_response['time']), len(nonlinear_response['time']))
        
        # 调整线性响应，将偏差状态转换回实际值
        adjusted_linear_states = np.zeros((4, min_length))
        adjusted_linear_states[0, :] = linear_response['states'][0, :min_length] + self.x_eq
        adjusted_linear_states[1, :] = linear_response['states'][1, :min_length] + self.v_eq
        adjusted_linear_states[2, :] = linear_response['states'][2, :min_length] + self.P_l_eq
        adjusted_linear_states[3, :] = linear_response['states'][3, :min_length] + self.P_r_eq
        
        adjusted_linear_output = linear_response['output'][:min_length] + self.x_eq
        
        # 检查是否有足够的数据进行比较
        if min_length > 0:
            # 计算误差
            error_position = adjusted_linear_states[0, :] - nonlinear_response['states'][0, :min_length]
            error_velocity = adjusted_linear_states[1, :] - nonlinear_response['states'][1, :min_length]
            error_P_l = adjusted_linear_states[2, :] - nonlinear_response['states'][2, :min_length]
            error_P_r = adjusted_linear_states[3, :] - nonlinear_response['states'][3, :min_length]
            
            # 计算误差统计
            error_analysis = {
                'max_error_position': np.max(np.abs(error_position)),
                'max_error_velocity': np.max(np.abs(error_velocity)),
                'max_error_P_l': np.max(np.abs(error_P_l)),
                'max_error_P_r': np.max(np.abs(error_P_r)),
                'mean_error_position': np.mean(np.abs(error_position)),
                'mean_error_velocity': np.mean(np.abs(error_velocity)),
                'mean_error_P_l': np.mean(np.abs(error_P_l)),
                'mean_error_P_r': np.mean(np.abs(error_P_r)),
            }
        else:
            # 如果没有数据，返回空的错误分析
            error_analysis = {
                'max_error_position': 0.0,
                'max_error_velocity': 0.0,
                'max_error_P_l': 0.0,
                'max_error_P_r': 0.0,
                'mean_error_position': 0.0,
                'mean_error_velocity': 0.0,
                'mean_error_P_l': 0.0,
                'mean_error_P_r': 0.0,
            }
        
        # 返回结果
        return error_analysis, linear_response, nonlinear_response
    
    def print_nonlinear_vs_linear(self):
        """打印线性化模型和非线性模型比较"""
        # 进行比较分析
        x0 = np.array([self.x_eq + 0.01, 0, self.P_l_eq, self.P_r_eq])
        error_analysis, _, _ = self.compare_linear_nonlinear(x0=x0, t_span=(0, 2))

        print("\n====== 线性化模型与非线性模型比较 ======")
        print(f"初始状态偏差: 位置偏离平衡点 0.01m")
        print(f"速度最大误差: {error_analysis['max_error_velocity']:.6f} m/s")
        print(f"左气室压力最大误差: {error_analysis['max_error_P_l']:.2f} Pa")
        print(f"右气室压力最大误差: {error_analysis['max_error_P_r']:.2f} Pa")
        print(f"位置平均误差: {error_analysis['mean_error_position']:.6f} m")
        print(f"速度平均误差: {error_analysis['mean_error_velocity']:.6f} m/s")
        print(f"左气室压力平均误差: {error_analysis['mean_error_P_l']:.2f} Pa")
        print(f"右气室压力平均误差: {error_analysis['mean_error_P_r']:.2f} Pa") # Corrected key
        print("===========================================\n")
        
    def print_summary(self):
        """打印系统分析摘要"""
        if self.A_matrix is None:
            self.linearize_system()
            
        analysis = self.analyze_system()
        
        print("\n====== 气动系统状态空间分析摘要 ======")
        print(f"系统阶数: {analysis['system_order']}")
        print(f"系统稳定性: {'稳定' if analysis['is_stable'] else '不稳定'}")
        print(f"系统可控性: {'完全可控 (rank=' + str(analysis['ctrb_rank']) + ')' if analysis['is_controllable'] else '不完全可控'}")
        print(f"系统可观性: {'完全可观 (rank=' + str(analysis['obsv_rank']) + ')' if analysis['is_observable'] else '不完全可观'}")
        
        print("\n系统矩阵:")
        print("A = ")
        print(np.array2string(self.A_matrix, precision=4))
        print("\nB = ")
        print(np.array2string(self.B_matrix, precision=4))
        print("\nC = ")
        print(np.array2string(self.C_matrix, precision=4))
        
        print("\n系统特征值:")
        for i, ev in enumerate(analysis['eigenvalues']):
            print(f"  λ{i+1} = {ev:.4f}")
            
        if self.K is not None:
            print("\n状态反馈控制器:")
            print(f"K = {np.array2string(self.K, precision=4)}")
            print("\n期望极点:")
            for i, p in enumerate(self.poles):
                print(f"  p{i+1} = {p:.4f}")
                
        print("\nDC增益:")
        print(f"  {analysis['dc_gain']:.4f}")
        
        print("======================================\n")
