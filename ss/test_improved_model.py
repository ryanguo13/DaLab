#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版线性化模型
==================
比较原始线性化模型和增强版线性化模型的效果
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.state_space_model import PneumaticStateSpaceModel
from modules.enhanced_compare import generate_comparison_report

# 创建两个模型实例：一个用默认线性化方法，一个用增强版
def create_models():
    # 共用相同的参数，针对15cm管道和9mm球体优化
    params = {
        'm_ball': 0.003,     # 小球质量 (kg) - 9mm球的估计质量
        'A': 0.0001,         # 管道横截面积 (m²) - 适合9mm球
        'f_v': 0.05,         # 粘性摩擦系数 - 针对短管径小球优化
        'V_tube': 0.0015,    # 管道总体积 (m³) - 15cm管道
        'L_tube': 0.15,      # 管道长度 (m) - 15cm
        'x_eq': 0.075,       # 平衡位置 (m) - 设为15cm管道的中点
        'P_l_eq': 180000,    # 左气室平衡压力 (Pa)
        'P_r_eq': 180000     # 右气室平衡压力 (Pa)
    }
    
    # 创建原始模型
    original_model = PneumaticStateSpaceModel(params.copy())
    
    # 使用不带增强特性的线性化
    class OriginalModel(PneumaticStateSpaceModel):
        def linearize_system(self):
            """使用原始线性化方法的重载函数"""
            # 计算平衡点的体积
            V_l_eq = self.A * self.x_eq
            V_r_eq = self.V_tube - self.A * self.x_eq
            
            # 基于气体状态方程和质量流量关系的常数
            gamma = 1.4  # 气体绝热指数
            
            # 压力-体积变化率常数
            k_l = gamma * self.P_l_eq / V_l_eq
            k_r = gamma * self.P_r_eq / V_r_eq
            
            # A矩阵元素计算
            a11 = 0  
            a12 = 1  
            a13 = 0  
            a14 = 0  
            
            a21 = 0  
            a22 = -self.f_v / self.m_ball  
            a23 = self.A / self.m_ball  
            a24 = -self.A / self.m_ball  
            
            a31 = -self.P_l_eq * self.A / V_l_eq  
            a32 = -k_l * self.A  
            a33 = 0  
            a34 = 0  
            
            a41 = self.P_r_eq * self.A / V_r_eq  
            a42 = k_r * self.A  
            a43 = 0  
            a44 = 0  
            
            # B矩阵元素计算
            b1 = 0  
            b2 = 0  
            b3 = k_l * self.C_max * (self.P_s - self.P_l_eq)  
            b4 = -k_r * self.C_max * (self.P_r_eq - self.P_atm)  
            
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
            import control as ctrl
            self.sys = ctrl.ss(self.A_matrix, self.B_matrix, self.C_matrix, self.D_matrix)
            
            return self.A_matrix, self.B_matrix, self.C_matrix, self.D_matrix
    
    # 创建使用原始线性化方法的模型实例
    original_model = OriginalModel(params.copy())
    original_model.linearize_system()
    
    # 创建增强版模型
    enhanced_model = PneumaticStateSpaceModel(params.copy())
    # 使用参数调优后的增强线性化方法
    enhanced_model.linearize_system(leakage_coeff=0.02, ball_clearance=0.0001)
    
    return original_model, enhanced_model

def main():
    # 创建模型
    original_model, enhanced_model = create_models()
    
    # 在15cm管道长度范围内选择多个测试点
    test_points = np.linspace(0.01, 0.14, 10)  # 从1cm到14cm，共10个测试点
    
    # 打印原始模型和增强模型摘要
    print("===== 原始线性化模型 =====")
    original_model.print_summary()
    
    print("\n===== 增强版线性化模型 =====")
    enhanced_model.print_summary()
    
    # 评估原始模型
    print("\n===== 原始线性化模型评估 =====")
    original_summary, _ = generate_comparison_report(
        original_model, 
        test_points=test_points,
        t_span=(0, 3), 
        dt=0.01,
        save_path="original_model_comparison.png"
    )
    
    # 评估增强版模型
    print("\n===== 增强版线性化模型评估 =====")
    enhanced_summary, _ = generate_comparison_report(
        enhanced_model, 
        test_points=test_points,
        t_span=(0, 3), 
        dt=0.01,
        save_path="enhanced_model_comparison.png"
    )
    
    # 比较改进效果
    improvement_percentage = (1 - enhanced_summary['average_mean_position_error'] / 
                             original_summary['average_mean_position_error']) * 100
    
    print("\n===== 模型改进效果 =====")
    print(f"原始模型平均位置误差: {original_summary['average_mean_position_error']:.6f} m")
    print(f"增强模型平均位置误差: {enhanced_summary['average_mean_position_error']:.6f} m")
    print(f"误差减少百分比: {improvement_percentage:.2f}%")
    print("==========================")
    
    # 返回结果用于进一步分析
    return {
        "original_model": original_model,
        "enhanced_model": enhanced_model,
        "original_summary": original_summary,
        "enhanced_summary": enhanced_summary,
        "improvement_percentage": improvement_percentage
    }

if __name__ == "__main__":
    results = main()
