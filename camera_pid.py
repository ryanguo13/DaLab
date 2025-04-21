# coding=utf-8
import mvsdk
import numpy as np  
import cv2
import time
import math
import signal
import sys
import ctypes
import gc  # 用于禁用垃圾回收
import os
import psutil  # 用于设置进程优先级
from collections import deque
from threading import Thread, Lock
from rpi_hardware_pwm import HardwarePWM
import matplotlib.pyplot as plt
import atexit
import io
from datetime import datetime

# 高精度定时器类定义
class Timer:
    """使用Linux内核时钟的精确定时器类"""
    CLOCK_MONOTONIC = 1
    CLOCK_MONOTONIC_RAW = 4

    class Timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', ctypes.c_long),
            ('tv_nsec', ctypes.c_long)
        ]

    def __init__(self):
        # 尝试加载不同的库名以提高兼容性
        try:
            self.librt = ctypes.CDLL('librt.so.1', use_errno=True)
        except OSError:
            try:
                self.librt = ctypes.CDLL('librt.so', use_errno=True)
            except OSError:
                self.librt = ctypes.CDLL('libc.so.6', use_errno=True)
                print("Using libc.so.6 for timer functions")
        
        # 设置函数参数和返回类型
        self.librt.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(self.Timespec)]
        self.librt.clock_gettime.restype = ctypes.c_int
        
        # 初始化timespec结构体
        self.ts = self.Timespec()
        self._start_time = self.get_raw_time()
        
    def get_raw_time(self):
        """获取原始时间"""
        if self.librt.clock_gettime(self.CLOCK_MONOTONIC_RAW, ctypes.byref(self.ts)) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return self.ts.tv_sec + self.ts.tv_nsec * 1e-9
        
    def get_time(self):
        """获取相对于开始时间的时间差（秒）"""
        return self.get_raw_time() - self._start_time
        
    def busy_wait(self, target_time):
        """忙等待直到目标时间"""
        target_absolute = self._start_time + target_time
        while self.get_raw_time() < target_absolute:
            pass

# 全局变量
global_ball_position = 0.0  # 当前球的位置，单位mm
position_lock = Lock()  # 用于保护球位置变量的锁
cycle_time = 28.0  # 将周期时间定义为全局变量

# 数据记录用变量
data_lock = Lock()
time_data = []  # 记录时间
setpoint_data = []  # 记录设定值
position_data = []  # 记录实际位置
error_data = []  # 记录误差
output_data = []  # 记录控制输出
jitter_data = []  # 记录时间抖动 - 新增

# PID参数记录 - 新增
kp_data = []  # 记录Kp参数变化
ki_data = []  # 记录Ki参数变化
kd_data = []  # 记录Kd参数变化
phase_data = []  # 记录控制阶段变化

# Add this global variable at the top of your file
data_already_saved = False
experiment_running = False  # 新增: 标记实验是否已经开始

# 全局变量，用于存储终端输出
original_stdout = sys.stdout

# 创建输出重定向类
class OutputCollector(io.TextIOBase):
    def __init__(self):
        self.buffer = []
    
    def write(self, text):
        # 同时写到原始标准输出和我们的缓冲区
        original_stdout.write(text)
        self.buffer.append(text)
        return len(text)
    
    def flush(self):
        original_stdout.flush()
    
    def get_output(self):
        return ''.join(self.buffer)

# 初始化全局输出收集器实例
output_collector = OutputCollector()

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
        
        # 防止dt过小导致计算异常
        if dt < 0.001:
            dt = 0.001
            
        error = setpoint - measured_value
        
        # 计算P控制输出 (由于ki和kd为0，实际只使用P控制)
        output = self.kp * error
        
        # 保存当前误差和时间供下次计算使用
        self.previous_error = error
        self.last_time = current_time
        
        return output, error
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

class GainSchedulingPID:
    def __init__(self):
        # Define parameter sets for different regions
        self.rising_params = {'kp': 0.3, 'ki': 0.0, 'kd': 15}  # More aggressive for rising
        self.holding_params = {'kp': 0.1, 'ki': 0.0, 'kd': 20}  # More stable for holding
        self.falling_params = {'kp': 0.4, 'ki': 0.0, 'kd': 15}  # Less aggressive for falling
        
        # Create PID controller with initial parameters
        self.pid = PIDController(kp=self.rising_params['kp'], 
                                ki=self.rising_params['ki'], 
                                kd=self.rising_params['kd'])
        
        self.previous_setpoint = 0
        self.current_phase = "init"  # 新增：记录当前控制阶段
        
    def compute(self, setpoint, measured_value):
        # Determine phase based on setpoint change
        if setpoint > self.previous_setpoint + 1:  # Rising (with 1mm threshold)
            self.pid.kp = self.rising_params['kp']
            self.pid.ki = self.rising_params['ki']
            self.pid.kd = self.rising_params['kd']
            self.current_phase = "rising"  # 记录当前为上升阶段
        elif setpoint < self.previous_setpoint - 1:  # Falling
            self.pid.kp = self.falling_params['kp']
            self.pid.ki = self.falling_params['ki']
            self.pid.kd = self.falling_params['kd']
            self.current_phase = "falling"  # 记录当前为下降阶段
        else:  # Holding
            self.pid.kp = self.holding_params['kp']
            self.pid.ki = self.holding_params['ki']
            self.pid.kd = self.holding_params['kd']
            self.current_phase = "holding"  # 记录当前为保持阶段
            
        self.previous_setpoint = setpoint
        
        # Use regular PID computation
        output, error = self.pid.compute(setpoint, measured_value)
        
        # 返回控制输出、误差和当前PID参数
        return output, error, self.pid.kp, self.pid.ki, self.pid.kd, self.current_phase

class FuzzyPID:
    def __init__(self, kp_base=0.6, ki_base=0.0, kd_base=7.0):
        self.kp_base = kp_base
        self.ki_base = ki_base
        self.kd_base = kd_base
        
        # Base PID controller
        self.pid = PIDController(kp=kp_base, ki=ki_base, kd=kd_base)
        
        # Previous values for calculating rate of change
        self.prev_error = 0
        self.prev_time = time.time()
        self.current_phase = "init"  # 添加阶段标记，保持与GainSchedulingPID接口一致
        
    def fuzzy_inference(self, error, error_rate):
        """Simple fuzzy inference to adjust PID parameters"""
        # Normalize error and error_rate to [-1, 1] range (assuming max error is around 50mm)
        norm_error = max(-1, min(1, error / 50.0))
        norm_error_rate = max(-1, min(1, error_rate / 20.0))  # Assuming 20mm/s max rate
        
        # Fuzzy rules (simplified)
        # 1. Large error -> increase Kp
        # 2. Small error -> decrease Kp, increase Kd
        # 3. Large error rate -> increase Kd
        # 4. Small error and small error rate -> increase Ki
        
        # Calculate adjustment factors
        kp_factor = 1.0 + 0.5 * abs(norm_error) - 0.2 * abs(norm_error_rate)
        ki_factor = 1.0 + 0.5 * (1 - abs(norm_error)) * (1 - abs(norm_error_rate))
        kd_factor = 1.0 + 0.3 * abs(norm_error_rate) + 0.2 * (1 - abs(norm_error))
        
        # Apply limits to prevent extreme adjustments
        kp_factor = max(0.5, min(1.5, kp_factor))
        ki_factor = max(0.5, min(1.5, ki_factor))
        kd_factor = max(0.5, min(1.5, kd_factor))
        
        # Update PID parameters
        self.pid.kp = self.kp_base * kp_factor
        self.pid.ki = self.ki_base * ki_factor
        self.pid.kd = self.kd_base * kd_factor
        
        return self.pid.kp, self.pid.ki, self.pid.kd
        
    def compute(self, setpoint, measured_value):
        current_time = time.time()
        dt = current_time - self.prev_time
        dt = max(0.001, dt)  # Prevent division by zero
        
        error = setpoint - measured_value
        error_rate = (error - self.prev_error) / dt
        
        # 确定当前阶段，保持与GainSchedulingPID的接口一致
        if setpoint > self.prev_error + 1:
            self.current_phase = "rising"
        elif setpoint < self.prev_error - 1:
            self.current_phase = "falling"
        else:
            self.current_phase = "holding"
        
        # Apply fuzzy inference
        kp, ki, kd = self.fuzzy_inference(error, error_rate)
        
        # Compute control output
        output, error = self.pid.compute(setpoint, measured_value)
        
        # Update previous values
        self.prev_error = error
        self.prev_time = current_time
        
        # 返回与GainSchedulingPID相同格式的输出
        return output, error, kp, ki, kd, self.current_phase

class ADRC:
    def __init__(self, b0=1.0, w0=10.0, wc=5.0):
        """
        b0: Approximate system gain
        w0: Observer bandwidth
        wc: Controller bandwidth
        """
        self.b0 = b0          # Approximate model parameter
        self.w0 = w0          # Observer bandwidth
        self.wc = wc          # Controller bandwidth
        
        # ESO (Extended State Observer) parameters
        self.beta1 = 3.0 * w0
        self.beta2 = 3.0 * w0**2
        self.beta3 = w0**3
        
        # ESO states
        self.z1 = 0.0  # Estimated position
        self.z2 = 0.0  # Estimated velocity
        self.z3 = 0.0  # Estimated total disturbance
        
        # Controller parameter
        self.kp = wc**2
        self.kd = 2 * wc
        
        # Previous values
        self.u_prev = 0.0
        self.last_time = time.time()
        
    def update_eso(self, y, u, dt):
        """Update the extended state observer"""
        # Prediction error
        e = y - self.z1
        
        # Update ESO states
        self.z1 += dt * (self.z2 + self.beta1 * e)
        self.z2 += dt * (self.z3 + self.beta2 * e + self.b0 * u)
        self.z3 += dt * (self.beta3 * e)
        
        return self.z1, self.z2, self.z3
        
    def compute(self, setpoint, measured_value):
        current_time = time.time()
        dt = current_time - self.last_time
        dt = max(0.001, dt)  # Prevent division by zero
        
        # Update ESO
        _, z2, z3 = self.update_eso(measured_value, self.u_prev, dt)
        
        # Calculate error
        error = setpoint - measured_value
        
        # ADRC control law (linear state error feedback + disturbance compensation)
        u0 = self.kp * error - self.kd * z2  # State feedback
        u = (u0 - z3) / self.b0               # Disturbance compensation
        
        # Store for next iteration
        self.u_prev = u
        self.last_time = current_time
        
        return u, error

def generate_setpoint_curve(t, x_min=0.0, x_max=156.0):
    """
    生成理想位移曲线，支持无限循环
    t: 当前时间(秒)
    x_min: 最小位移(mm)
    x_max: 最大位移(mm)
    返回: 期望位置(mm)
    """
   
    global cycle_time  # 使用全局变量
    cycle_time=28.0  # 周期时间(秒)
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
    
class FrameProcessor:
    def __init__(self):
        self.frame_queue = deque(maxlen=1)  # 只保留最新的一帧
        self.lock = Lock()
        self.fps_lock = Lock()  # 用于FPS数据的锁
        self.running = True
        self.sdk_fps = 0.0  # 存储SDK计算的FPS
        
    def add_frame(self, frame):
        with self.lock:
            if len(self.frame_queue) > 0:
                self.frame_queue.pop()  # 移除旧帧
            self.frame_queue.append(frame)

    def get_frame(self):
        with self.lock:
            return self.frame_queue.pop() if self.frame_queue else None
            
    def set_sdk_fps(self, fps):
        with self.fps_lock:
            self.sdk_fps = fps
            
    def get_sdk_fps(self):
        with self.fps_lock:
            return self.sdk_fps

def pixel_to_physical(pixel_x, zero_pixel_x, scale_factor):
    """
    将像素坐标转换为物理坐标
    pixel_x: 像素坐标_320*240
    zero_pixel_x: 0点对应的像素坐标_320*240
    scale_factor: 比例尺系数 (mm/pixel)
    """
    return (pixel_x - zero_pixel_x) * scale_factor

def capture_frames(hCamera, pFrameBuffer, processor):
    """相机采集线程"""
    global global_ball_position
    
    # 初始化帧统计变量
    last_time = time.time()
    last_frame_stat = mvsdk.CameraGetFrameStatistic(hCamera)
    last_capture_count = last_frame_stat.iCapture
    fps_update_interval = 1.0  # 每1秒更新一次帧率
    
    # ROI参数
    roi_x = 7
    roi_y = 220
    roi_width = 570
    roi_height = 43
    zero_pixel_x = 18.8  # 修改为X方向的零点
    scale_factor = 165/572  # 24mm/167pixel
    
    while processor.running:
        try:
            # 计算SDK帧率
            current_time = time.time()
            if current_time - last_time >= fps_update_interval:
                # 获取当前帧统计信息
                current_frame_stat = mvsdk.CameraGetFrameStatistic(hCamera)
                current_capture_count = current_frame_stat.iCapture
                
                # 计算区间内的帧数和经过的时间
                frame_count_interval = current_capture_count - last_capture_count
                time_interval = current_time - last_time
                
                # 计算实时帧率
                if time_interval > 0:
                    sdk_fps = frame_count_interval / time_interval
                    processor.set_sdk_fps(sdk_fps)
                    print(f"SDK帧率: {sdk_fps:.1f}")
                
                # 更新上一次的值
                last_time = current_time
                last_capture_count = current_capture_count
            
            # 执行软触发采集一帧图像
            mvsdk.CameraClearBuffer(hCamera)  # 清空相机内已缓存的所有帧
            mvsdk.CameraSoftTrigger(hCamera)  # 执行一次软触发
            
            # 取一帧图像，添加200ms超时
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            
            # 将图像数据转换成OpenCV格式
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth)
            
            # ROI裁剪
            roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            # 反二值化
            _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
            
            # 移除小于500像素的连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # 处理连通区域
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 500:
                    # 获取轮廓mask
                    contour_mask = (labels == i).astype(np.uint8) * 255
        
                    # 寻找轮廓并计算最小包围圆
                    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # 计算最小包围圆
                        (x, y), radius = cv2.minEnclosingCircle(contours[0])
            
                        # 计算圆度过滤非圆形物体
                        perimeter = cv2.arcLength(contours[0], True)
                        if perimeter > 0:
                            circularity = 4 * math.pi * stats[i, cv2.CC_STAT_AREA] / (perimeter ** 2)
                            if 0.85 < circularity < 1.15:  # 圆度阈值范围
                                # 找到小球，计算质心并更新位置
                                centroid_x = centroids[i][0]
                                pixel_x = centroid_x + roi_x  # 转换为原图坐标系
                                physical_x = pixel_to_physical(pixel_x, zero_pixel_x, scale_factor)
                                
                                # 更新全局球位置
                                with position_lock:
                                    global_ball_position = physical_x
            
            # 释放图像缓冲区
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            
            # 将帧添加到队列
            frame_with_roi = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(frame_with_roi, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)
            processor.add_frame(frame_with_roi)
            
            # 控制采集速率
            time.sleep(0.002)  # 限制最大帧率为500fps
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:  # 忽略超时错误
                print("Camera error({}): {}".format(e.error_code, e.message))
            time.sleep(0.01)  # 出错时短暂等待
        except Exception as e:
            print(f"Error in capture loop: {e}")

def set_realtime_priority():
    """使用纯Python设置实时优先级"""
    try:
        # 设置CPU亲和性到核心3
        current_process = psutil.Process()
        current_process.cpu_affinity([3])
        print("Process pinned to CPU core 3")
        
        # 尝试设置实时调度策略
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
            print("Process set to FIFO scheduling with priority 99")
        except Exception as e:
            print(f"Failed to set FIFO scheduling: {e}")
        
        # 尝试内存锁定功能
        try:
            import resource
            if hasattr(resource, 'mlockall'):
                resource.mlockall(resource.MCL_CURRENT | resource.MCL_FUTURE)
                print("Memory locked to prevent swapping")
            else:
                print("Memory locking not supported on this platform")
        except Exception as e:
            print("Memory locking not supported on this platform")
            
    except Exception as e:
        print(f"Failed to set real-time priority: {e}")

def run_control_experiment():
    """进行控制实验的主要函数"""
    global time_data, setpoint_data, position_data, error_data, output_data, jitter_data, global_ball_position, experiment_running
    global pwm_freq, DT, Kp, cycle_time
    global kp_data, ki_data, kd_data, phase_data  # 新增：PID参数记录变量
    
    # 标记实验已开始
    experiment_running = True
    
    # 清理上次实验数据
    with data_lock:
        time_data.clear()
        setpoint_data.clear()
        position_data.clear()
        error_data.clear()
        output_data.clear()
        jitter_data.clear()
        # 清理PID参数数据
        kp_data.clear()
        ki_data.clear()
        kd_data.clear()
        phase_data.clear()
    
    # 禁用垃圾回收
    gc.disable()
    
    # 设置优先级
    set_realtime_priority()
    
    # 创建高精度定时器
    timer = Timer()
    
    # 控制实验参数
    DT = 0.01  # 固定的时间步长，10ms
    Kp = 0.6  # 比例系数 - 根据2_1_P_controller.py中的值
    ki=0.0
    kd=7.0
    free_roll_time = 0.0  # 不使用自由滚动时间
    experiment_duration = 3 * cycle_time  # 实验时长（秒）
    expected_points = int(experiment_duration / DT)  # 预期数据点数
    
    # 预分配数据数组
    with data_lock:
        time_data = np.zeros(expected_points)
        setpoint_data = np.zeros(expected_points)
        position_data = np.zeros(expected_points)
        error_data = np.zeros(expected_points)
        output_data = np.zeros(expected_points)
        jitter_data = np.zeros(expected_points)
        # 预分配PID参数数组
        kp_data = np.zeros(expected_points)
        ki_data = np.zeros(expected_points)
        kd_data = np.zeros(expected_points)
        phase_data = np.empty(expected_points, dtype='U10')  # 使用字符串存储阶段名称
    
    # PWM设置
    pwm_channel = 2  # 从2_1_P_controller.py采用
    pwm_freq = 3000  # PWM频率Hz
    pwm = HardwarePWM(pwm_channel=pwm_channel, hz=pwm_freq, chip=2)
    pwm.start(0)  # 启动PWM，初始占空比为0
    print(f"PWM started on channel {pwm_channel}")
    
    # 创建实时动态PID控制器 - 使用FuzzyPID替代原来的GainSchedulingPID
    pid_controller = FuzzyPID(kp_base=0.3, ki_base=0.0, kd_base=20.0)
    
    try:
        # 实验开始时间
        print(f"Starting experiment with duration {experiment_duration}s...")
        start_time = timer.get_time()
        prev_elapsed_time = 0
        
        # 设置轨迹的最小值和最大值
        x_min = 0.0  # 轨迹的最小位置，单位mm
        x_max = 156.0  # 轨迹的最大位置，单位mm
        
        # 控制循环
        for i in range(expected_points):
            # 计算下一个采样时间点 - 这是关键部分，确保固定时间步长
            next_sample_time = start_time + (i * DT)
            
            # 以高精度等待到下一个采样时间
            timer.busy_wait(next_sample_time)
            
            # 获取当前时间
            current_time = timer.get_time()
            elapsed_time = current_time - start_time
            
            # 计算定时抖动
            actual_dt = elapsed_time - prev_elapsed_time if i > 0 else DT
            jitter_ms = (actual_dt - DT) * 1000  # 转换为毫秒
            prev_elapsed_time = elapsed_time
            
            # 获取当前小球位置
            with position_lock:
                current_position = global_ball_position
                
            # 计算设定点 - 使用全局cycle_time变量
            setpoint = generate_setpoint_curve(elapsed_time, x_min=x_min, x_max=x_max)
            
            # 计算控制输出 - 使用FuzzyPID控制器，获取当前PID参数
            output, error, current_kp, current_ki, current_kd, current_phase = pid_controller.compute(setpoint, current_position)
            
            # 限制输出电压在0-3.3V范围内
            output = max(0, min(output, 3.3))
                
            # 将电压转换为PWM占空比 (0-3.3V -> 0-100%)
            duty_cycle = (output / 3.3) * 100
            
            # 设置PWM占空比
            pwm.change_duty_cycle(duty_cycle)
            
            # 记录数据
            time_data[i] = elapsed_time
            setpoint_data[i] = setpoint
            position_data[i] = current_position
            error_data[i] = error
            output_data[i] = output
            jitter_data[i] = jitter_ms
            
            # 记录PID参数
            kp_data[i] = current_kp
            ki_data[i] = current_ki
            kd_data[i] = current_kd
            phase_data[i] = current_phase
            
            # 打印控制信息
            if i % 10 == 0:  # 只打印每10个点，减少输出量
                print(f"Time: {elapsed_time:.2f}s, Phase: {current_phase}, Setpoint: {setpoint:.2f}mm, Position: {current_position:.2f}mm, "
                      f"Error: {error:.2f}mm, Kp: {current_kp:.2f}, Kd: {current_kd:.2f}, Output: {output:.2f}V")
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted!")
    finally:
        # 停止PWM
        if 'pwm' in locals():
            pwm.stop()
            print("PWM stopped")
        
        # 重新启用垃圾回收
        gc.enable()
        
        # 保存实验数据
        save_data_and_plot()

def save_data_and_plot():
    """保存记录的数据到CSV文件并生成曲线图"""
    global time_data, setpoint_data, position_data, error_data, output_data, jitter_data, data_already_saved, experiment_running
    global position_plot_filename, error_plot_filename, output_plot_filename, jitter_plot_filename
    global kp_data, ki_data, kd_data, phase_data
    global pid_params_plot_filename  # 新增PID参数图表文件名
    
    # 如果数据已经保存过了，或实验未开始，则不再重复保存
    if data_already_saved or not experiment_running:
        print("No data to save or data already saved")
        return
    
    # 指定保存路径
    save_path = "/home/pi/Downloads/Xiaohui/Test_py/Xiaohui_camera_test/407_Zhenyuan_data/data"
    
    try:
        # 确保目录存在
        import os
        os.makedirs(save_path, exist_ok=True)
        
        with data_lock:
            if len(time_data) == 0 or isinstance(time_data, list) and not time_data:
                print("No data to save")
                return
                
            # 裁剪数据到有效长度 (如果数组是预分配的)
            if isinstance(time_data, np.ndarray) and np.all(time_data[-1] == 0):
                # 找到最后一个非零时间索引
                valid_indices = np.where(time_data > 0)[0]
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1] + 1
                    time_data = time_data[:last_valid_idx]
                    setpoint_data = setpoint_data[:last_valid_idx]
                    position_data = position_data[:last_valid_idx]
                    error_data = error_data[:last_valid_idx]
                    output_data = output_data[:last_valid_idx]
                    jitter_data = jitter_data[:last_valid_idx]
                    
                    # 裁剪PID参数数据
                    kp_data = kp_data[:last_valid_idx]
                    ki_data = ki_data[:last_valid_idx]
                    kd_data = kd_data[:last_valid_idx]
                    phase_data = phase_data[:last_valid_idx]
                
            # 创建数据数组 (不包含phase_data因为它是字符串)
            data = np.column_stack((
                time_data,
                setpoint_data,
                position_data,
                error_data,
                output_data,
                jitter_data,
                kp_data,
                ki_data,
                kd_data
            ))
            
            # 保存到CSV文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"control_data_{timestamp}.csv")
            
            # 添加表头并保存
            header = "Time(s),Setpoint(mm),Position(mm),Error(mm),Output(V),Jitter(ms),Kp,Ki,Kd"
            np.savetxt(filename, data, delimiter=',', header=header, comments='')
            
            # 单独保存phase数据（因为它是字符串）
            phase_filename = os.path.join(save_path, f"phase_data_{timestamp}.txt")
            with open(phase_filename, 'w') as f:
                for phase in phase_data:
                    f.write(f"{phase}\n")
            
            print(f"Data saved to {filename}")
            print(f"Phase data saved to {phase_filename}")
            
            # 图1: 绘制位置控制曲线
            plt.figure(figsize=(12, 6))
            plt.plot(time_data, setpoint_data, 'r-', linewidth=2, label='Setpoint')
            plt.plot(time_data, position_data, 'b-', linewidth=1.5, label='Actual Position')
            plt.title('Ball Position Control')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (mm)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            position_plot_filename = os.path.join(save_path, f"position_curves_{timestamp}.png")
            plt.savefig(position_plot_filename)
            plt.close()  # 关闭当前图表
            
            # 图2: 绘制误差
            plt.figure(figsize=(12, 6))
            plt.plot(time_data, error_data, 'g-', linewidth=1.5, label='Error')
            plt.title('Control Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Error (mm)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            error_plot_filename = os.path.join(save_path, f"error_output_curves_{timestamp}.png")
            plt.savefig(error_plot_filename)
            plt.close()
            
            # 图3：绘制控制输出
            plt.figure(figsize=(12, 6))
            plt.plot(time_data, output_data, 'k-', linewidth=1, label='Control Output')
            plt.title('Control Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output (V)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            output_plot_filename = os.path.join(save_path, f"output_curves_{timestamp}.png")
            plt.savefig(output_plot_filename)
            plt.close()
            
            # 图4: 绘制时间抖动
            plt.figure(figsize=(12, 6))
            plt.plot(time_data, jitter_data, 'k-', linewidth=1, label='Timing Jitter')
            if len(jitter_data) > 0:
                max_jitter = np.max(np.abs(jitter_data))
                avg_jitter = np.mean(np.abs(jitter_data))
                plt.title(f'Timing Jitter - Max: {max_jitter:.3f} ms, Avg: {avg_jitter:.3f} ms')
            else:
                plt.title('Timing Jitter')
            plt.xlabel('Time (s)')
            plt.ylabel('Jitter (ms)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            jitter_plot_filename = os.path.join(save_path, f"timing_jitter_{timestamp}.png")
            plt.savefig(jitter_plot_filename)
            plt.close()
            
            # 图5: 改进PID参数变化图 - 使用更好的可视化方式
            plt.figure(figsize=(12, 10))
            
            # 子图1: 设定值、位置和误差
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(time_data, setpoint_data, 'r-', linewidth=1.5, label='Setpoint')
            plt.plot(time_data, position_data, 'b-', linewidth=1.5, label='Actual Position')
            plt.plot(time_data, error_data, 'g-', linewidth=1, label='Error')
            plt.title('Position Tracking and Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Position/Error (mm)')
            plt.grid(True)
            plt.legend(loc='upper right')
            
            # 找出阶段变化点
            phase_changes = []
            if len(phase_data) > 0:
                current_phase = phase_data[0]
                for i in range(1, len(phase_data)):
                    if phase_data[i] != current_phase:
                        # 标记阶段变化点
                        plt.axvline(time_data[i], color='black', linestyle='--', alpha=0.5)
                        phase_changes.append((time_data[i], phase_data[i]))
                        current_phase = phase_data[i]
            
            # 子图2: Kp参数
            plt.subplot(3, 1, 2, sharex=ax1)
            plt.plot(time_data, kp_data, 'g-', linewidth=1.5)
            plt.title('PID Parameter - Proportional Gain (Kp) Dynamic Change')
            plt.xlabel('Time (s)')
            plt.ylabel('Kp Value')
            plt.grid(True)
            
            # 复制阶段变化线
            for t, _ in phase_changes:
                plt.axvline(t, color='black', linestyle='--', alpha=0.5)
            
            # 子图3: Kd参数
            plt.subplot(3, 1, 3, sharex=ax1)
            plt.plot(time_data, kd_data, 'm-', linewidth=1.5)
            plt.title('PID Parameter - Derivative Gain (Kd) Dynamic Change')
            plt.xlabel('Time (s)')
            plt.ylabel('Kd Value')
            plt.grid(True)
            
            # 复制阶段变化线
            for t, _ in phase_changes:
                plt.axvline(t, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            pid_params_plot_filename = os.path.join(save_path, f"pid_params_{timestamp}.png")
            plt.savefig(pid_params_plot_filename)
            plt.close()
            print(f"PID parameters plot saved to {pid_params_plot_filename}")
            
            # 标记数据已保存
            data_already_saved = True
            
    except Exception as e:
        print(f"Error saving data and plotting: {e}")

def ensure_camera_closed():
    """在程序启动时确保相机设备已经关闭"""
    try:
        # 尝试枚举所有相机设备
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) == 0:
            print("No camera found!")
            return
            
        # 获取第一个相机的信息
        DevInfo = DevList[0]
        
        # 检查相机设备是否已经打开 (通过简单的操作并捕获错误)
        try:
            # 尝试打开并立即关闭，如果打开成功说明之前没有打开
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            if hCamera:
                print("Successfully initialized camera, closing...")
                mvsdk.CameraUnInit(hCamera)
        except mvsdk.CameraException as e:
            if e.error_code == -18:  # 设备已经打开
                print("Camera is already open, attempting to close it...")
                
                # 这里需要更直接的方法来关闭相机，但SDK通常不提供
                # 我们可以尝试通过系统命令关闭占用相机的进程
                try:
                    if os.name == 'posix':  # Linux
                        # 查找并终止所有可能占用相机的进程
                        os.system("pkill -f python.*320_Junhua.py")
                        print("Terminated potential camera-using processes")
                        time.sleep(2)  # 给系统时间释放资源
                    else:
                        print("Camera is in use and automatic release is only supported on Linux")
                except Exception as e:
                    print(f"Failed to release camera: {e}")
    except Exception as e:
        print(f"Error checking camera state: {e}")

# 生成Markdown报告
def generate_markdown_report():
    global data_already_saved, output_collector, cycle_time
    global pid_params_plot_filename  # 新增PID参数图表文件名
    
    # 确保report目录存在
    report_dir = "/home/pi/Downloads/Xiaohui/Test_py/Xiaohui_camera_test/407_Zhenyuan_data/report"
    try:
        os.makedirs(report_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating report directory: {e}")
        return
    
    try:
        # 收集终端输出
        output_text = output_collector.get_output()
        
        # 创建Markdown内容
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        markdown_content = f"""# Experiment Report - {timestamp}

## Experiment Overview
- Time: {timestamp}
- Control Method: Visual-based Ball Position Control
- Controller: Fuzzy PID Controller (Dynamic Real-time PID)
- Trajectory Cycle Time: {cycle_time}s

## Experimental Parameters
- Sampling Period: {DT*1000:.1f}ms
- PWM Frequency: {pwm_freq}Hz
- Image Processing: ROI cropping, Thresholding, Connected Component Analysis, Circularity Check

## Controller Configuration
- Fuzzy PID Base Parameters: 
  - Kp_base={0.3}
  - Ki_base={0.0}
  - Kd_base={12.0}
- Dynamic Parameter Adjustment: PID parameters are continuously adjusted based on error magnitude and error rate

## Terminal Output
{output_text}

## Experimental Results
The results plots are as below:
![Position Control Curve]({position_plot_filename})
![Error Curve]({error_plot_filename})
![Control Output Curve]({output_plot_filename})
![Timing Jitter Curve]({jitter_plot_filename})
![PID Parameters]({pid_params_plot_filename})

"""
            
        # 保存Markdown报告
        report_filename = os.path.join(report_dir, f"report_{timestamp}.md")
        with open(report_filename, 'w') as f:
            f.write(markdown_content)
        
        print(f"Markdown report saved to {report_filename}")
        
    except Exception as e:
        print(f"Error generating markdown report: {e}")

def main():
    global experiment_running, data_already_saved
    
    # 确保相机设备已关闭
    ensure_camera_closed()
    
    # 注册程序退出时的回调函数，确保保存数据和生成图表
    atexit.register(save_data_and_plot)
    
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return
        
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]

    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 设置ISP输出MONO8格式
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    
    # 相机模式切换成软触发模式
    mvsdk.CameraSetTriggerMode(hCamera, 1)  # 设置为软触发模式
    print("Camera set to software trigger mode")

    # 手动曝光设置
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 4 * 1000)  # 曝光时间4ms4
    mvsdk.CameraSetAnalogGain(hCamera, 100)          # 模拟增益
    mvsdk.CameraSetGamma(hCamera, 250)               # 伽马
    mvsdk.CameraSetContrast(hCamera, 200)            # 对比度

    # 设置分辨率
    resolution = mvsdk.CameraGetImageResolution(hCamera)
    resolution.iIndex = 0  # 使用索引0对应640*480分辨率
    mvsdk.CameraSetImageResolution(hCamera, resolution)

    # 设置高帧率模式
    mvsdk.CameraSetFrameSpeed(hCamera, 2)           # 2:高速模式

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算buffer大小并申请buffer
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    # 创建帧处理器
    processor = FrameProcessor()
    
    # 启动相机采集线程
    capture_thread = Thread(target=capture_frames, args=(hCamera, pFrameBuffer, processor))
    capture_thread.daemon = True
    capture_thread.start()
    
    # 等待用户启动控制实验
    input("按Enter键开始控制实验...")
    
    # 启动控制实验线程
    control_thread = Thread(target=run_control_experiment)
    control_thread.daemon = True
    control_thread.start()

    # 处理显示图像和用户交互
    try:
        while True:
            frame = processor.get_frame()
            if frame is not None:
                # 获取SDK帧率
                sdk_fps = processor.get_sdk_fps()
                
                # 在图像上显示帧率信息
                if frame.ndim < 3:  # 如果是灰度图，转换为彩色
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()
                
                cv2.putText(display_frame, f"Camera FPS: {sdk_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow("Camera", display_frame)
            
            # 检查是否按下ESC键退出
            key = cv2.waitKey(1)
            if key == 27:  # ESC键的ASCII码为27
                break
                
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        # 清理资源前确保数据已保存
        if experiment_running and not data_already_saved:
            save_data_and_plot()
        
        # 停止采集线程
        processor.running = False
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
            
        # 清理资源
        cv2.destroyAllWindows()
        mvsdk.CameraStop(hCamera)
        mvsdk.CameraUnInit(hCamera)
        mvsdk.CameraAlignFree(pFrameBuffer)
        print("Camera resources released")

    

if __name__ == '__main__':
    main()
    generate_markdown_report()