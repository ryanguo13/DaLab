#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID控制器模块
=============
提供各种PID控制器实现，包括基础PID、模糊PID等。

"""

import time
import math

class PIDController:
    """基础PID控制器类"""
    def __init__(self, kp=1.0, ki=0.0, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, setpoint, measured_value):
        """计算PID控制输出"""
        current_time = time.time()
        dt = current_time - self.last_time

        # 防止dt过小导致计算异常
        if dt < 0.001:
            dt = 0.001

        error = setpoint - measured_value

        # 积分项
        self.integral += error * dt
        # 微分项
        derivative = (error - self.previous_error) / dt

        # 计算PID控制输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 保存当前误差和时间供下次计算使用
        self.previous_error = error
        self.last_time = current_time

        return output, error

    def reset(self):
        """重置控制器状态"""
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

class FuzzyPID:
    """模糊PID控制器，根据误差和误差变化率动态调整PID参数"""
    def __init__(self, kp_base=0.6, ki_base=0.0, kd_base=7.0):
        self.kp_base = kp_base
        self.ki_base = ki_base
        self.kd_base = kd_base

        # 基础PID控制器
        self.pid = PIDController(kp=kp_base, ki=ki_base, kd=kd_base)

        # 用于计算变化率的前一个值
        self.prev_error = 0
        self.prev_time = time.time()
        self.current_phase = "init"  # 添加阶段标记

    def fuzzy_inference(self, error, error_rate):
        """简单模糊推理，根据误差和误差变化率调整PID参数"""
        # 将误差和误差变化率归一化到[-1, 1]范围（假设最大误差约为50mm或0.05m）
        norm_error = max(-1, min(1, error / 0.05))  # 使用公尺单位
        norm_error_rate = max(-1, min(1, error_rate / 0.2))  # 假设最大速率为0.2 m/s

        # 模糊规则（简化版）
        kp_factor = 1.0 + 0.5 * abs(norm_error) - 0.2 * abs(norm_error_rate)
        ki_factor = 1.0 + 0.5 * (1 - abs(norm_error)) * (1 - abs(norm_error_rate))
        kd_factor = 1.0 + 0.3 * abs(norm_error_rate) + 0.2 * (1 - abs(norm_error))

        # 应用限制以防止极端调整
        kp_factor = max(0.5, min(1.5, kp_factor))
        ki_factor = max(0.5, min(1.5, ki_factor))  # 允许Ki调整
        kd_factor = max(0.5, min(1.5, kd_factor))

        # 更新PID参数
        self.pid.kp = self.kp_base * kp_factor
        self.pid.ki = self.ki_base * ki_factor  # 允许Ki调整
        self.pid.kd = self.kd_base * kd_factor

        return self.pid.kp, self.pid.ki, self.pid.kd

    def compute(self, setpoint, measured_value):
        """计算控制输出，同时更新PID参数"""
        current_time = time.time()
        dt = current_time - self.prev_time
        dt = max(0.001, dt)  # 防止除零

        error = setpoint - measured_value
        error_rate = (error - self.prev_error) / dt

        # 确定当前阶段
        if error > 0.01:  # 假设误差大于1cm视为上升或下降
            self.current_phase = "moving"
        else:
            self.current_phase = "holding"

        # 应用模糊推理
        kp, ki, kd = self.fuzzy_inference(error, error_rate)

        # 使用基础PID的compute方法计算控制输出
        output, error = self.pid.compute(setpoint, measured_value)

        # 更新前一个值
        self.prev_error = error
        self.prev_time = current_time

        return output, error, kp, ki, kd, self.current_phase

def generate_setpoint_curve(t, x_min=0.0, x_max=0.156, cycle_time=28.0):
    """
    生成理想位移曲线，支持无限循环
    t: 当前时间(秒)
    x_min: 最小位移(m)
    x_max: 最大位移(m)
    cycle_time: 周期时间(秒)
    返回: 期望位置(m)
    """
    # 计算在周期内的时间点
    t_cycle = t % cycle_time

    # 定义周期内各阶段时间
    delay_start = 2.0    # 初始保持阶段
    rise_time = 10.0     # 上升阶段
    hold_time = 4.0      # 高位保持阶段
    fall_time = 10.0     # 下降阶段
    delay_end = 2.0      # 末尾保持阶段

    # 确保所有阶段时间之和等于周期时间
    assert abs(delay_start + rise_time + hold_time + fall_time + delay_end - cycle_time) < 1e-6, "阶段时间总和应等于周期时间"

    # 目标位移值
    X_max = x_max  # 使用函数参数

    # 根据周期内时间点确定输出值
    if t_cycle < delay_start:
        return x_min  # 初始延迟
    elif t_cycle < delay_start + rise_time:
        # 上升阶段，使用三次多项式平滑过渡
        progress = (t_cycle - delay_start) / rise_time
        smooth_progress = 3 * progress**2 - 2 * progress**3
        return x_min + (X_max - x_min) * smooth_progress
    elif t_cycle < delay_start + rise_time + hold_time:
        return X_max  # 高位保持
    elif t_cycle < delay_start + rise_time + hold_time + fall_time:
        # 下降阶段，使用三次多项式平滑过渡
        progress = (t_cycle - (delay_start + rise_time + hold_time)) / fall_time
        smooth_progress = 3 * progress**2 - 2 * progress**3
        return X_max - (X_max - x_min) * smooth_progress
    else:
        return x_min  # 末尾延迟
