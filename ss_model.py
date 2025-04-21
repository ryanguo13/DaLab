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
import inspect

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 从camera_pid导入需要的组件
from camera_pid import generate_setpoint_curve, PIDController, FuzzyPID


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
            
        # 系统参数 - 与camera_pid.py兼容
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
        
        # PID控制器参数 - 与camera_pid.py兼容
        self.kp = params.get('kp', 0.3)              # 比例系数
        self.ki = params.get('ki', 0.0)              # 积分系数
        self.kd = params.get('kd', 20.0)             # 微分系数
        self.pwm_freq = params.get('pwm_freq', 3000)  # PWM频率Hz
        
        # 平衡点 (线性化位置)
        self.x_eq = params.get('x_eq', 0.25)           # 平衡位置 (m)
        self.v_eq = 0                                  # 平衡速度 (m/s)
        self.P_l_eq = params.get('P_l_eq', 200000)     # 左气室平衡压力 (Pa)
        self.P_r_eq = params.get('P_r_eq', 200000)     # 右气室平衡压力 (Pa)
        self.s_eq = 0                                  # 阀门平衡位置
        
        # 轨迹参数 - 与camera_pid.py兼容
        self.cycle_time = params.get('cycle_time', 28.0)  # 轨迹周期时间(秒)
        
        # 线性化模型矩阵
        self.A_matrix = None
        self.B_matrix = None
        self.C_matrix = None
        self.D_matrix = None
        self.sys = None
        
        # 控制设计参数
        self.K = None  # 状态反馈增益矩阵
        self.poles = None  # 期望极点