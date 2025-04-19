#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气动系统中小球运动的数值模拟
===========================
这个模块实现了气动系统的基础物理模型，包括牛顿运动方程、气体动力学计算
以及使用数值积分方法求解微分方程。

作者: Claude AI
日期: 2025-04-19
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
        self.V_tube = params.get('V_tube', 0.01)      # 管道总体积 (m³)
        self.L_tube = params.get('L_tube', 0.5)       # 管道长度 (m)
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
        self.x_s = params.get('x_s', 0.1)             # 小球初始位置 (m)
        self.x_b = params.get('x_b', 0.1)             # 小球当前位置 (m)
        self.v_b = params.get('v_b', 0)               # 小球初始速度 (m/s)
        self.P_l = params.get('P_l', self.P_atm)      # 左气室初始压力 (Pa)
        self.P_r = params.get('P_r', self.P_atm)      # 右气室初始压力 (Pa)
        
        # 阀门位置控制
        self.s_spool = params.get('s_spool', 0)       # 阀门位置 (-1到1之间)
        
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
        # 解包状态变量
        x_0, v_0, P_l, P_r = y
        
        # 计算实际位置
        x_b = x_0 + self.x_s
        
        # 计算体积
        V_l = self.A * x_b
        V_r = self.V_tube - self.A * x_b
        
        # 根据阀门位置确定流动方向和上下游压力
        if self.s_spool > 0:  # 左进右出
            P_1l = self.P_s
            P_1r = self.P_atm
            m_dot_l = self.calc_mass_flow_rate(P_1l, P_l, self.s_spool)
            m_dot_r = self.calc_mass_flow_rate(P_r, P_1r, self.s_spool)
        elif self.s_spool < 0:  # 左出右进
            P_1l = self.P_atm
            P_1r = self.P_s
            m_dot_l = self.calc_mass_flow_rate(P_l, P_1l, -self.s_spool)
            m_dot_r = self.calc_mass_flow_rate(P_1r, P_r, -self.s_spool)
        else:  # 阀门关闭
            m_dot_l = 0
            m_dot_r = 0
        
        # 计算压力变化率
        dP_l_dt = (m_dot_l / V_l) * (self.P_0 / self.rho_0) - P_l * (v_0 / x_b)
        dP_r_dt = (m_dot_r / V_r) * (self.P_0 / self.rho_0) + P_r * (self.A * v_0 / V_r)
        
        # 计算加速度
        a_0 = (P_l * self.A - P_r * self.A - self.f_v * v_0) / self.m_ball
        
        return [v_0, a_0, dP_l_dt, dP_r_dt]
    
    def simulate(self):
        """模拟系统动力学"""
        # 初始状态 [x_0, v_0, P_l, P_r]
        y0 = [self.x_b - self.x_s, self.v_b, self.P_l, self.P_r]
        
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
        x_0 = self.states[0]
        v_0 = self.states[1]
        P_l = self.states[2]
        P_r = self.states[3]
        
        # 计算实际位置
        x_b = x_0 + self.x_s
        
        # 创建一个2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 位置时间图
        axes[0, 0].plot(self.times, x_b, 'b-', label='Ball Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
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
    
    def create_animation(self):
        """创建系统动画"""
        if self.times is None or self.states is None:
            raise ValueError("请先运行simulate()方法")
        
        # 解包状态
        x_0 = self.states[0]
        P_l = self.states[2]
        P_r = self.states[3]
        
        # 计算实际位置
        x_b = x_0 + self.x_s
        
        # 设置图形
        fig, ax = plt.subplots(figsize=(10, 4))
        tube_length = self.L_tube
        
        # 设置坐标轴
        ax.set_xlim(0, tube_length)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        ax.set_title('Pneumatic Ball System Animation')
        
        # 创建管道
        tube = plt.Rectangle((0, -0.3), tube_length, 0.6, fill=False, color='black')
        ax.add_patch(tube)
        
        # 创建小球
        ball = plt.Circle((x_b[0], 0), self.r_ball, color='blue')
        ax.add_patch(ball)
        
        # 创建文本标签
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        pl_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
        pr_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        
        # 创建左右箭头表示压力
        left_arrow = ax.arrow(0.1, 0, 0.1, 0, head_width=0.05, head_length=0.02, fc='green', ec='green')
        right_arrow = ax.arrow(tube_length-0.1, 0, -0.1, 0, head_width=0.05, head_length=0.02, fc='red', ec='red')

        def animate(i):
            nonlocal left_arrow, right_arrow
            # 更新小球位置
            ball.center = (x_b[i], 0)
            
            # 更新文本信息
            time_text.set_text(f'Time: {self.times[i]:.2f} s')
            pl_text.set_text(f'Left P: {P_l[i]/1000:.2f} kPa')
            pr_text.set_text(f'Right P: {P_r[i]/1000:.2f} kPa')
            
            # 更新箭头长度（表示压力大小）
            left_scale = min(max(P_l[i] / self.P_s, 0.01), 0.3)
            right_scale = min(max(P_r[i] / self.P_s, 0.01), 0.3)
            
            # 移除旧箭头
            left_arrow.remove()
            right_arrow.remove()

            # 添加新箭头
            left_arrow = ax.arrow(0.05, 0, left_scale, 0, head_width=0.05, head_length=0.02, fc='green', ec='green')
            right_arrow = ax.arrow(tube_length-0.05, 0, -right_scale, 0, head_width=0.05, head_length=0.02, fc='red', ec='red')
            
            return ball, time_text, pl_text, pr_text, left_arrow, right_arrow
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(self.times), interval=20, blit=True)
        
        return anim, fig


# 使用示例
def run_demo():
    """运行演示"""
    # 系统参数
    params = {
        'm_ball': 0.05,        # 小球质量 (kg)
        'A': 0.001,            # 管道横截面积 (m²)
        'f_v': 0.1,            # 粘性摩擦系数
        'V_tube': 0.01,        # 管道总体积 (m³)
        'L_tube': 0.5,         # 管道长度 (m)
        'P_s': 500000,         # 供气压力 (Pa)
        'P_atm': 101325,       # 大气压力 (Pa)
        'x_s': 0.1,            # 小球初始位置 (m)
        'x_b': 0.1,            # 小球当前位置 (m)
        'v_b': 0,              # 小球初始速度 (m/s)
        'P_l': 101325,         # 左气室初始压力 (Pa)
        'P_r': 101325,         # 右气室初始压力 (Pa)
        's_spool': 0.5,        # 阀门位置 (正：左进右出，负：左出右进)
        't_span': (0, 5)       # 模拟时间范围 (s)
    }
    
    # 创建系统实例
    system = PneumaticBallSystem(params)
    
    # 运行模拟
    result = system.simulate()
    
    # 绘制结果
    fig = system.plot_results()
    
    # 创建动画
    anim, anim_fig = system.create_animation()
    
    plt.show()
    
    return system, fig, anim, anim_fig


if __name__ == "__main__":
    system, fig, anim, anim_fig = run_demo()
