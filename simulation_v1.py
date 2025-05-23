#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气动系统中小球运动的数值模拟
===========================
这个模块实现了气动系统的基础物理模型，包括牛顿运动方程、气体动力学计算
以及使用数值积分方法求解微分方程。

作者: Claude AI
日期: 2025-04-29
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os
import sys

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class PWMController:
    """PWM控制器类
    
    控制阀门开度，接受输入电压(0-10V)，转换为占空比，然后产生相应的阀门位置信号
    - 0V对应最大左向气流（负占空比-100%）
    - 5V对应无气流（0%占空比）
    - 10V对应最大右向气流（正占空比100%）
    """
    
    def __init__(self):
        """初始化PWM控制器"""
        self.voltage = 5.0  # 初始电压 (V)
        self.duty_cycle = 0  # 初始占空比 (%)
        self.s_spool = 0  # 阀门位置 (-1到1之间)
        self.history = {
            'time': [],
            'voltage': [],
            'duty_cycle': [],
            's_spool': []
        }
    
    def set_voltage(self, voltage, time=0):
        """设置控制电压
        
        参数:
            voltage: 控制电压 (0-10V)
            time: 当前时间点
        """
        # 确保电压在有效范围内
        self.voltage = np.clip(voltage, 0, 10)
        
        # 计算占空比: 0V -> -100%, 5V -> 0%, 10V -> 100%
        self.duty_cycle = (self.voltage - 5.0) * 20  # 转换为占空比 (-100% 到 100%)
        
        # 计算阀门位置: -100% -> -1, 0% -> 0, 100% -> 1
        self.s_spool = self.duty_cycle / 100.0
        
        # 记录历史值
        self.history['time'].append(time)
        self.history['voltage'].append(self.voltage)
        self.history['duty_cycle'].append(self.duty_cycle)
        self.history['s_spool'].append(self.s_spool)
    
    def set_duty_cycle(self, duty_cycle, time=0):
        """直接设置占空比
        
        参数:
            duty_cycle: 占空比 (-100% 到 100%)
            time: 当前时间点
        """
        # 确保占空比在有效范围内
        self.duty_cycle = np.clip(duty_cycle, -100, 100)
        
        # 计算对应的电压
        self.voltage = (self.duty_cycle / 20.0) + 5.0
        
        # 计算阀门位置
        self.s_spool = self.duty_cycle / 100.0
        
        # 记录历史值
        self.history['time'].append(time)
        self.history['voltage'].append(self.voltage)
        self.history['duty_cycle'].append(self.duty_cycle)
        self.history['s_spool'].append(self.s_spool)
    
    def get_s_spool(self):
        """获取当前阀门位置"""
        return self.s_spool
    
    def plot_history(self):
        """绘制控制历史"""
        if not self.history['time']:
            raise ValueError("没有控制历史数据")
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        
        # 电压历史
        axs[0].step(self.history['time'], self.history['voltage'], 'b-', where='post')
        axs[0].set_ylabel('Voltage (V)')
        axs[0].set_title('Control Input History')
        axs[0].grid(True)
        
        # 占空比历史
        axs[1].step(self.history['time'], self.history['duty_cycle'], 'r-', where='post')
        axs[1].set_ylabel('Duty Cycle (%)')
        axs[1].grid(True)
        
        # 阀门位置历史
        axs[2].step(self.history['time'], self.history['s_spool'], 'g-', where='post')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Spool Position (-1 to 1)')
        axs[2].grid(True)
        
        plt.tight_layout()
        return fig


class PneumaticBallSystem:
    """
    气动系统中小球运动的数值模型
    描述一个水平管道中由左右气室压力差驱动的小球运动系统
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
        self.L_tube = params.get('L_tube', 0.156)     # 管道长度 (m)，按照要求为156mm
        self.V_tube = params.get('V_tube', self.A * self.L_tube)  # 管道总体积 (m³)
        self.r_ball = np.sqrt(self.A / np.pi)         # 计算小球半径 (m)
        
        # 气动参数
        self.P_0 = params.get('P_0', 101325)          # 参考压力 (Pa)
        self.rho_0 = params.get('rho_0', 1.225)       # 参考密度 (kg/m³)
        self.T_0 = params.get('T_0', 293.15)          # 参考温度 (K)
        self.T_1 = params.get('T_1', 293.15)          # 上游温度 (K)
        self.P_s = params.get('P_s', 500000)          # 供气压力 (Pa)
        self.P_atm = params.get('P_atm', 101325)      # 大气压力 (Pa)
        
        # 阀门参数
        self.C_max = params.get('C_max', 1e-8)        # 最大音速导纳 (m³/(s·Pa))
        self.b_cr = params.get('b_cr', 0.5)           # 临界压力比
        
        # 初始状态
        # x_0表示相对于管道中心的位置，范围为[-L_tube/2, L_tube/2]
        self.x_0 = params.get('x_0', 0.0)             # 小球初始位置 (m)，默认在中心
        self.v_0 = params.get('v_0', 0)               # 小球初始速度 (m/s)
        self.P_l = params.get('P_l', self.P_atm)      # 左气室初始压力 (Pa)
        self.P_r = params.get('P_r', self.P_atm)      # 右气室初始压力 (Pa)
        
        # 初始化PWM控制器
        self.pwm_controller = PWMController()
        
        # 模拟参数
        self.t_span = params.get('t_span', (0, 10))   # 模拟时间范围 (s)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 1000)  # 评估时间点
        
        # 状态变量存储
        self.times = None
        self.states = None
        self.info = None
    
    def C_spool(self, s_spool):
        """计算给定阀门位置的音速导纳"""
        # 简单线性模型：阀门位置和音速导纳的关系
        return self.C_max * abs(s_spool)
    
    def b_spool(self, s_spool):
        """计算给定阀门位置的临界压力比"""
        # 这里我们假设临界压力比是常数
        return self.b_cr
    
    def calc_mass_flow_rate(self, P_up, P_down, s_spool):
        """计算质量流量
        
        参数:
            P_up: 上游压力 (Pa)
            P_down: 下游压力 (Pa)
            s_spool: 阀门位置
            
        返回:
            m_dot: 质量流量 (kg/s)
        """
        # 计算压力比
        P_ratio = P_down / P_up
        
        # 阀门特性
        C = self.C_spool(s_spool)
        b = self.b_spool(s_spool)
        
        # 临界流动条件
        if P_ratio <= b:
            # 临界流动（声速流动）
            m_dot = P_up * C * np.sqrt(self.T_0 / self.T_1)
        else:
            # 亚临界流动
            m_dot = P_up * C * np.sqrt(self.T_0 / self.T_1) * \
                   np.sqrt(1 - ((P_ratio - b) / (1 - b))**2)
        
        return m_dot
    
    def system_dynamics(self, t, y):
        """系统动力学方程
        
        参数:
            t: 时间
            y: 状态向量 [x_0, v_0, P_l, P_r]
            
        返回:
            dydt: 状态导数 [v_0, a_0, dP_l/dt, dP_r/dt]
        """
        # 解包状态变量 - x_0是相对于管道中心的位置
        x_0, v_0, P_l, P_r = y
        
        # 根据当前时间更新控制输入
        self.update_control(t)
        s_spool = self.pwm_controller.get_s_spool()
        
        # 计算左右气室长度
        # 将相对于中心的坐标转换为从左端点开始的坐标
        x_abs = x_0 + self.L_tube/2
        
        # 计算体积
        V_l = self.A * x_abs
        V_r = self.A * (self.L_tube - x_abs)
        
        # 根据阀门位置确定流动方向和上下游压力
        if s_spool > 0:  # 正占空比：右向气流
            P_1l = self.P_atm
            P_1r = self.P_s
            m_dot_l = self.calc_mass_flow_rate(P_l, P_1l, s_spool)
            m_dot_r = self.calc_mass_flow_rate(P_1r, P_r, s_spool)
        elif s_spool < 0:  # 负占空比：左向气流
            P_1l = self.P_s
            P_1r = self.P_atm
            m_dot_l = self.calc_mass_flow_rate(P_1l, P_l, -s_spool)
            m_dot_r = self.calc_mass_flow_rate(P_r, P_1r, -s_spool)
        else:  # 零占空比：阀门关闭
            m_dot_l = 0
            m_dot_r = 0
        
        # 计算压力变化率
        dP_l_dt = (m_dot_l / V_l) * (self.P_0 / self.rho_0) - P_l * (v_0 * self.A / V_l)
        dP_r_dt = (m_dot_r / V_r) * (self.P_0 / self.rho_0) + P_r * (v_0 * self.A / V_r)
        
        # 计算加速度
        a_0 = (P_l * self.A - P_r * self.A - self.f_v * v_0) / self.m_ball
        
        return [v_0, a_0, dP_l_dt, dP_r_dt]
    
    def update_control(self, t):
        """根据时间更新控制输入
        
        这个函数可以被重写，以实现不同的控制策略
        """
        # 默认行为 - 时间依赖的电压设置
        # 这是一个示例控制序列，可以根据需要修改
        if t <= 2.00:
            self.pwm_controller.set_duty_cycle(100, t)    # 100% 右向气流
        elif t <= 8.00:
            self.pwm_controller.set_duty_cycle(50, t)     # 50% 右向气流
        elif t <= 8.10:
            self.pwm_controller.set_duty_cycle(0, t)      # 0% 气流 (阀门关闭)
        else:
            self.pwm_controller.set_duty_cycle(50, t)     # 50% 右向气流
    
    def simulate(self):
        """模拟系统动力学"""
        # 初始状态 [x_0, v_0, P_l, P_r]
        y0 = [self.x_0, self.v_0, self.P_l, self.P_r]
        
        # 微分方程求解
        result = solve_ivp(
            self.system_dynamics,
            self.t_span,
            y0,
            method='RK45',
            t_eval=self.t_eval,
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )
        
        self.times = result.t
        self.states = result.y
        self.info = result
        
        return result
    
    def plot_results(self):
        """绘制模拟结果"""
        if self.times is None or self.states is None:
            raise ValueError("请先运行simulate()方法")
        
        # 解包状态
        x_0 = self.states[0]  # 相对于中心的位置
        v_0 = self.states[1]  # 速度
        P_l = self.states[2]  # 左气室压力
        P_r = self.states[3]  # 右气室压力
        
        # 创建一个2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 位置时间图 - 将位置转换为mm显示
        axes[0, 0].plot(self.times, x_0 * 1000, 'b-', label='Ball Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (mm)')
        axes[0, 0].set_title('Ball Position vs Time')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # 速度时间图
        axes[0, 1].plot(self.times, v_0, 'r-', label='Ball Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Ball Velocity vs Time')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 压力时间图
        axes[1, 0].plot(self.times, P_l / 1000, 'g-', label='Left Chamber Pressure')
        axes[1, 0].plot(self.times, P_r / 1000, 'm-', label='Right Chamber Pressure')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Pressure (kPa)')
        axes[1, 0].set_title('Chamber Pressures vs Time')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # 压力差图
        pressure_diff = P_l - P_r
        axes[1, 1].plot(self.times, pressure_diff / 1000, 'c-', label='Pressure Difference')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Pressure Diff (kPa)')
        axes[1, 1].set_title('Pressure Difference vs Time')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_control_input(self):
        """绘制控制输入"""
        return self.pwm_controller.plot_history()
    
    def create_animation(self):
        """创建系统动画"""
        if self.times is None or self.states is None:
            raise ValueError("请先运行simulate()方法")
        
        # 解包状态
        x_0 = self.states[0]  # 相对于中心的位置
        P_l = self.states[2]
        P_r = self.states[3]
        
        # 设置图形
        fig, ax = plt.subplots(figsize=(10, 4))
        tube_length = self.L_tube
        
        # 设置坐标轴 - 使用相对于中心的坐标系
        ax.set_xlim(-tube_length/2, tube_length/2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        ax.set_title('Pneumatic Ball System Animation')
        
        # 创建管道
        tube = plt.Rectangle((-tube_length/2, -0.3), tube_length, 0.6, fill=False, color='black')
        ax.add_patch(tube)
        
        # 创建小球
        ball = plt.Circle((x_0[0], 0), self.r_ball, color='blue')
        ax.add_patch(ball)
        
        # 创建文本标签
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        pl_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
        pr_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        control_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        
        # 创建左右箭头表示压力
        left_arrow = ax.arrow(-tube_length/4, 0, 0.1, 0, head_width=0.05, head_length=0.02, fc='green', ec='green')
        right_arrow = ax.arrow(tube_length/4, 0, -0.1, 0, head_width=0.05, head_length=0.02, fc='red', ec='red')

        def animate(i):
            nonlocal left_arrow, right_arrow
            # 更新小球位置
            ball.center = (x_0[i], 0)
            
            # 找到最接近当前时间的控制值
            time_idx = np.abs(np.array(self.pwm_controller.history['time']) - self.times[i]).argmin()
            current_s_spool = self.pwm_controller.history['s_spool'][time_idx]
            current_duty = self.pwm_controller.history['duty_cycle'][time_idx]
            
            # 更新文本信息
            time_text.set_text(f'Time: {self.times[i]:.2f} s')
            pl_text.set_text(f'Left P: {P_l[i]/1000:.2f} kPa')
            pr_text.set_text(f'Right P: {P_r[i]/1000:.2f} kPa')
            control_text.set_text(f'Duty: {current_duty:.1f}% (Spool: {current_s_spool:.2f})')
            
            # 更新箭头长度（表示压力大小）
            left_scale = min(max(P_l[i] / self.P_s, 0.01), 0.3)
            right_scale = min(max(P_r[i] / self.P_s, 0.01), 0.3)
            
            # 移除旧箭头
            left_arrow.remove()
            right_arrow.remove()

            # 添加新箭头
            left_arrow = ax.arrow(-tube_length/4, 0, left_scale, 0, head_width=0.05, head_length=0.02, fc='green', ec='green')
            right_arrow = ax.arrow(tube_length/4, 0, -right_scale, 0, head_width=0.05, head_length=0.02, fc='red', ec='red')
            
            return ball, time_text, pl_text, pr_text, control_text, left_arrow, right_arrow
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(self.times), interval=20, blit=True)
        
        return anim, fig


# 使用示例
def run_demo():
    """运行演示"""
    # 系统参数
    params = {
        'm_ball': 0.05,         # 小球质量 (kg)
        'A': 0.001,             # 管道横截面积 (m²)
        'f_v': 0.1,             # 粘性摩擦系数
        'L_tube': 0.156,        # 管道长度 (m)，按照要求为156mm
        'P_s': 500000,          # 供气压力 (Pa)
        'P_atm': 101325,        # 大气压力 (Pa)
        'x_0': 0.0,             # 小球初始位置 (m)，默认在中心
        'v_0': 0,               # 小球初始速度 (m/s)
        'P_l': 101325,          # 左气室初始压力 (Pa)
        'P_r': 101325,          # 右气室初始压力 (Pa)
        't_span': (0, 10)       # 模拟时间范围 (s)
    }
    
    # 创建系统实例
    system = PneumaticBallSystem(params)
    
    # 运行模拟
    result = system.simulate()
    
    # 绘制结果
    fig = system.plot_results()
    
    # 绘制控制输入
    control_fig = system.plot_control_input()
    
    # 创建动画
    anim, anim_fig = system.create_animation()
    
    plt.show()
    
    return system, fig, control_fig, anim, anim_fig


if __name__ == "__main__":
    system, fig, control_fig, anim, anim_fig = run_demo()
