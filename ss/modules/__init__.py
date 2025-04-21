#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统模块包
==========
提供各模块的导入支持，便于从外部引入。
"""

# 从各模块导入关键组件
try:
    from .pid_controller import PIDController, FuzzyPID, generate_setpoint_curve
except ImportError as e:
    print(f"导入pid_controller模块失败: {e}")

try:
    from .system_dynamics import nonlinear_dynamics, simulate_nonlinear_system
except ImportError as e:
    print(f"导入system_dynamics模块失败: {e}")

try:
    from .state_space_model import PneumaticStateSpaceModel
except ImportError as e:
    print(f"导入state_space_model模块失败: {e}")

try:
    from .visualization import (
        plot_response, 
        plot_state_space, 
        plot_root_locus, 
        plot_bode, 
        plot_pid_response, 
        plot_comparison
    )
except ImportError as e:
    print(f"导入visualization模块失败: {e}")
