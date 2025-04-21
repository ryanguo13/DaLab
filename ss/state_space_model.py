#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ctrl
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time
import math

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 從 camera_pid.py 導入的程式碼開始 ---

# PID 控制器基類
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, setpoint, measured_value):
        current_time = time.time()
        dt = current_time - self.last_time

        # 防