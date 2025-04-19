#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气动系统控制仿真模块
==================
这个模块在气动系统基础模型之上添加了控制功能，实现了PID控制器用于位置控制，
并提供了更丰富的动画和可视化分析工具。

作者: Claude AI
日期: 2025-04-19
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import os
import sys

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class PneumaticControlSystem:
    """
    可视化和控制的气动系统仿真类
    包含了图形界面、动画、状态反馈和控制策略
    """
    
    def __init__(self, params=None):
        """初始化系统参数和控制参数"""
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
        
        # 阀门控制参数
        self.control_type = params.get('control_type', 'position')  # 'position'或'pressure'
        self.target_position = params.get('target_position', 0.25)  # 目标位置 (m)
        self.kp = params.get('kp', 2.0)               # 比例增益
        self.ki = params.get('ki', 0.1)               # 积分增益
        self.kd = params.get('kd', 0.5)               # 微分增益
        self.control_interval = params.get('control_interval', 0.01)  # 控制间隔 (s)
        
        # PID控制器状态
        self.integral_error = 0
        self.prev_error = 0
        self.last_control_time = 0
        
        # 阀门位置初始化
        self.s_spool = 0  # 阀门位置 (-1到1之间)
        
        # 模拟参数
        self.t_span = params.get('t_span', (0, 5))   # 模拟时间范围 (s)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 1000)  # 评估时间点
        
        # 图形和动画参数
        self.fig = None
        self.anim = None
        
        # 状态变量存储
        self.times = None
        self.states = None
        self.spool_history = []
        self.control_times = []
        
    def C_spool(self, s_spool):
        """计算给定阀门位置的音速导纳"""
        return self.C_max * abs(s_spool)
    
    def b_spool(self, s_spool):
        """计算给定阀门位置的临界压力比"""
        return self.b_cr
    
    def calc_mass_flow_rate(self, P_up, P_down, s_spool):
        """计算质量流量"""
        # 防止除零错误
        if P_up <= 0:
            return 0
            
        # 计算压力比
        P_ratio = max(min(P_down / P_up, 1), 0)
        
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
        """系统动力学方程"""
        # 解包状态变量
        x_0, v_0, P_l, P_r = y
        
        # 计算实际位置
        x_b = x_0 + self.x_s
        
        # 执行控制操作
        if t >= self.last_control_time + self.control_interval:
            self.update_control(t, x_b, v_0, P_l, P_r)
            self.last_control_time = t
        
        # 计算体积
        V_l = self.A * x_b
        V_r = self.V_tube - self.A * x_b
        
        # 防止体积为零
        V_l = max(V_l, 1e-10)
        V_r = max(V_r, 1e-10)
        
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
        dP_l_dt = (m_dot_l / V_l) * (self.P_0 / self.rho_0) - P_l * (v_0 / x_b) if x_b > 0 else 0
        dP_r_dt = (m_dot_r / V_r) * (self.P_0 / self.rho_0) + P_r * (self.A * v_0 / V_r)
        
        # 计算加速度
        a_0 = (P_l * self.A - P_r * self.A - self.f_v * v_0) / self.m_ball
        
        # 保存控制输入历史
        if len(self.control_times) == 0 or t > self.control_times[-1]:
            self.control_times.append(t)
            self.spool_history.append(self.s_spool)
        
        return [v_0, a_0, dP_l_dt, dP_r_dt]
    
    def update_control(self, t, x_b, v_b, P_l, P_r):
        """更新控制输入（实现PID控制器）"""
        if self.control_type == 'position':
            # 位置控制
            error = self.target_position - x_b
            # 积分项
            self.integral_error += error * self.control_interval
            # 微分项
            derivative_error = (error - self.prev_error) / self.control_interval
            # PID控制器输出
            control_signal = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
            # 限制阀门开度在[-1, 1]范围内
            self.s_spool = max(min(control_signal, 1), -1)
            # 更新上一次误差
            self.prev_error = error
        elif self.control_type == 'pressure':
            # 压力差控制（简化实现）
            pressure_diff = P_l - P_r
            target_diff = (self.target_position - x_b) * 10000  # 简化的压力差目标
            error = target_diff - pressure_diff
            # PID控制器输出
            control_signal = self.kp * error
            # 限制阀门开度
            self.s_spool = max(min(control_signal / 100000, 1), -1)
    
    def simulate(self):
        """模拟系统动力学"""
        # 重置控制器状态
        self.integral_error = 0
        self.prev_error = 0
        self.last_control_time = 0
        self.spool_history = []
        self.control_times = []
        
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
        
        return result
    
    def create_visualization(self):
        """创建完整的可视化界面（包括状态图和系统动画）"""
        if self.times is None or self.states is None:
            raise ValueError("请先运行simulate()方法")
        
        # 解包状态
        x_0 = self.states[0]
        v_0 = self.states[1]
        P_l = self.states[2]
        P_r = self.states[3]
        
        # 计算实际位置
        x_b = x_0 + self.x_s
        
        # 创建复杂的图形布局
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3)
        
        # 系统动画显示区域（占据左侧两列）
        ax_system = self.fig.add_subplot(gs[:2, :2])
        ax_system.set_xlim(0, self.L_tube)
        ax_system.set_ylim(-0.3, 0.3)
        ax_system.set_aspect('equal')
        ax_system.set_xlabel('Position (m)')
        ax_system.set_title('Pneumatic System Simulation')
        
        # 状态图表
        ax_pos = self.fig.add_subplot(gs[0, 2])
        ax_vel = self.fig.add_subplot(gs[1, 2])
        ax_pressure = self.fig.add_subplot(gs[2, 0])
        ax_control = self.fig.add_subplot(gs[2, 1])
        ax_phase = self.fig.add_subplot(gs[2, 2])
        
        # 位置时间图
        line_pos, = ax_pos.plot([], [], 'b-', label='Ball Position')
        line_target = ax_pos.plot([self.times[0], self.times[-1]], 
                                 [self.target_position, self.target_position], 
                                 'r--', label='Target Position')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.set_title('Position Tracking')
        ax_pos.set_xlim(self.times[0], self.times[-1])
        ax_pos.set_ylim(0, self.L_tube)
        ax_pos.grid(True)
        ax_pos.legend()
        
        # 速度时间图
        line_vel, = ax_vel.plot([], [], 'r-', label='Ball Velocity')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.set_title('Velocity Profile')
        ax_vel.set_xlim(self.times[0], self.times[-1])
        max_vel = max(abs(np.max(v_0)), abs(np.min(v_0))) * 1.2
        ax_vel.set_ylim(-max_vel, max_vel)
        ax_vel.grid(True)
        ax_vel.legend()
        
        # 压力时间图
        line_pl, = ax_pressure.plot([], [], 'g-', label='Left Pressure')
        line_pr, = ax_pressure.plot([], [], 'm-', label='Right Pressure')
        ax_pressure.set_xlabel('Time (s)')
        ax_pressure.set_ylabel('Pressure (kPa)')
        ax_pressure.set_title('Chamber Pressures')
        ax_pressure.set_xlim(self.times[0], self.times[-1])
        max_pressure = max(np.max(P_l), np.max(P_r)) / 1000 * 1.2
        ax_pressure.set_ylim(0, max_pressure)
        ax_pressure.grid(True)
        ax_pressure.legend()
        
        # 控制输入图
        line_control, = ax_control.plot([], [], 'c-', label='Valve Position')
        ax_control.set_xlabel('Time (s)')
        ax_control.set_ylabel('Valve Position')
        ax_control.set_title('Control Input')
        ax_control.set_xlim(self.times[0], self.times[-1])
        ax_control.set_ylim(-1.2, 1.2)
        ax_control.grid(True)
        ax_control.legend()
        
        # 相空间图 (位置-速度)
        line_phase, = ax_phase.plot([], [], 'k-', label='Phase Trajectory')
        mark_phase, = ax_phase.plot([], [], 'ro', markersize=5)
        ax_phase.set_xlabel('Position (m)')
        ax_phase.set_ylabel('Velocity (m/s)')
        ax_phase.set_title('Phase Space Trajectory')
        ax_phase.set_xlim(0, self.L_tube)
        ax_phase.set_ylim(-max_vel, max_vel)
        ax_phase.grid(True)
        ax_phase.legend()
        
        # 创建管道和小球
        tube = Rectangle((0, -0.1), self.L_tube, 0.2, fill=False, color='black', linewidth=2)
        ax_system.add_patch(tube)
        
        ball = Circle((x_b[0], 0), self.r_ball, color='blue', zorder=2)
        ax_system.add_patch(ball)
        
        # 创建文本标签
        time_text = ax_system.text(0.02, 0.95, '', transform=ax_system.transAxes)
        pl_text = ax_system.text(0.02, 0.9, '', transform=ax_system.transAxes)
        pr_text = ax_system.text(0.02, 0.85, '', transform=ax_system.transAxes)
        control_text = ax_system.text(0.02, 0.8, '', transform=ax_system.transAxes)
        
        # 创建左右压力指示器
        left_pressure_bar = Rectangle((0.05, -0.2), 0.1, 0.05, color='green', alpha=0.7)
        right_pressure_bar = Rectangle((self.L_tube-0.15, -0.2), 0.1, 0.05, color='magenta', alpha=0.7)
        ax_system.add_patch(left_pressure_bar)
        ax_system.add_patch(right_pressure_bar)
        
        # 创建阀门指示器
        valve_indicator = Rectangle((self.L_tube/2-0.05, -0.25), 0.1, 0.05, color='cyan', alpha=0.7)
        ax_system.add_patch(valve_indicator)
        
        # 管中间的标记线
        mid_line = plt.Line2D([self.L_tube/2, self.L_tube/2], [-0.1, 0.1], 
                              color='gray', linestyle='--', alpha=0.5)
        ax_system.add_line(mid_line)
        
        # 目标位置标记线
        target_line = plt.Line2D([self.target_position, self.target_position], [-0.1, 0.1], 
                                color='red', linestyle='--', alpha=0.7)
        ax_system.add_line(target_line)
        
        # 微调布局
        plt.tight_layout()
        
        # 创建动画更新函数
        def animate(i):
            # 更新小球位置
            ball.center = (x_b[i], 0)
            
            # 更新位置图
            line_pos.set_data(self.times[:i+1], x_b[:i+1])
            
            # 更新速度图
            line_vel.set_data(self.times[:i+1], v_0[:i+1])
            
            # 更新压力图
            line_pl.set_data(self.times[:i+1], P_l[:i+1]/1000)
            line_pr.set_data(self.times[:i+1], P_r[:i+1]/1000)
            
            # 更新控制输入图
            current_idx = np.searchsorted(self.control_times, self.times[i])
            if current_idx > 0:
                control_times = np.array(self.control_times[:current_idx])
                spool_values = np.array(self.spool_history[:current_idx])
                line_control.set_data(control_times, spool_values)
            
            # 更新相空间图
            line_phase.set_data(x_b[:i+1], v_0[:i+1])
            mark_phase.set_data([x_b[i]], [v_0[i]])
            
            # 更新文本信息
            time_text.set_text(f'Time: {self.times[i]:.2f} s')
            pl_text.set_text(f'Left P: {P_l[i]/1000:.2f} kPa')
            pr_text.set_text(f'Right P: {P_r[i]/1000:.2f} kPa')
            
            # 找到当前的阀门位置
            if i < len(self.times):
                current_time = self.times[i]
                idx = np.searchsorted(self.control_times, current_time)
                if idx > 0 and idx <= len(self.spool_history):
                    current_spool = self.spool_history[idx-1]
                    control_text.set_text(f'Valve Pos: {current_spool:.2f}')
                    # 更新阀门指示器
                    valve_indicator.set_x(self.L_tube/2 - 0.05 + current_spool * 0.05)
                    valve_indicator.set_color('red' if current_spool > 0 else 'blue' if current_spool < 0 else 'gray')
            
            # 更新压力指示器
            left_scale = P_l[i] / self.P_s
            right_scale = P_r[i] / self.P_s
            left_pressure_bar.set_height(0.15 * left_scale)
            right_pressure_bar.set_height(0.15 * right_scale)
            
            return (ball, time_text, pl_text, pr_text, control_text, left_pressure_bar, 
                    right_pressure_bar, valve_indicator, line_pos, line_vel, line_pl, 
                    line_pr, line_control, line_phase, mark_phase)
        
        # 创建动画
        self.anim = FuncAnimation(self.fig, animate, frames=len(self.times),
                                 interval=20, blit=True)
        
        return self.fig, self.anim
    
    def save_animation(self, filename='pneumatic_control_animation.gif'):
        """保存动画为GIF文件"""
        if self.anim is None:
            raise ValueError("请先创建动画")
        
        # 保存为GIF
        writer = PillowWriter(fps=30)
        self.anim.save(filename, writer=writer)
        print(f"动画已保存为 {filename}")


# 使用示例
def run_control_demo():
    """运行控制系统演示"""
    # 系统参数
    params = {
        'm_ball': 0.05,                # 小球质量 (kg)
        'A': 0.001,                    # 管道横截面积 (m²)
        'f_v': 0.1,                    # 粘性摩擦系数
        'V_tube': 0.01,                # 管道总体积 (m³)
        'L_tube': 0.5,                 # 管道长度 (m)
        'P_s': 500000,                 # 供气压力 (Pa)
        'x_s': 0.1,                    # 小球初始位置 (m)
        'x_b': 0.1,                    # 小球当前位置 (m)
        'control_type': 'position',    # 控制类型
        'target_position': 0.3,        # 目标位置 (m)
        'kp': 5.0,                     # 比例增益
        'ki': 0.1,                     # 积分增益
        'kd': 2.0,                     # 微分增益
        'control_interval': 0.01,      # 控制间隔 (s)
        't_span': (0, 3)               # 模拟时间范围 (s)
    }
    
    # 创建控制系统实例
    system = PneumaticControlSystem(params)
    
    # 运行模拟
    system.simulate()
    
    # 创建可视化
    fig, anim = system.create_visualization()
    
    # 显示图形
    plt.show()
    
    return system, fig, anim


if __name__ == "__main__":
    system, fig, anim = run_control_demo()
