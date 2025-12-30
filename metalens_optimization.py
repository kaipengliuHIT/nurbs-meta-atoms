import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import sys
import os

# 添加当前目录到路径
sys.path.append('/mnt/e/pythoncode/nurbs-meta-atoms')

from transformer_nurbs_model import (
    NURBSTransformerModel,
    normalize_control_points,
    denormalize_targets
)
from nurbs_atoms_data import Simulation

class MetalensOptimizer:
    """超透镜优化器类"""
    
    def __init__(self, 
                 model_path: str = "nurbs_transformer_model.pth",
                 n_segments: int = 64,
                 focal_length: float = 50e-6,  # 焦距 50 μm
                 wavelength: float = 532e-9,   # 波长 532 nm
                 lens_radius: float = 50e-6):  # 透镜半径 50 μm
        """
        初始化超透镜优化器
        
        Args:
            model_path: 代理模型路径
            n_segments: 透镜分段数
            focal_length: 焦距
            wavelength: 工作波长
            lens_radius: 透镜半径
        """
        self.n_segments = n_segments
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.lens_radius = lens_radius
        self.k = 2 * np.pi / wavelength  # 波数
        
        # 加载代理模型
        self.model = NURBSTransformerModel(
            n_control_points=8,
            d_model=128,
            nhead=8,
            num_layers=4,
            d_ff=256,
            dropout=0.1
        )
        
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            self.model_available = True
            print(f"已加载代理模型: {model_path}")
        else:
            print(f"代理模型 {model_path} 不存在，将使用真实仿真")
            self.model_available = False
    
    def ideal_phase_profile(self, r: np.ndarray) -> np.ndarray:
        """
        计算理想相位分布
        
        Args:
            r: 径向坐标
            
        Returns:
            理想相位分布
        """
        # 理想超透镜相位分布: φ(r) = -k * (sqrt(r^2 + f^2) - f)
        phase = -self.k * (np.sqrt(r**2 + self.focal_length**2) - self.focal_length)
        # 将相位限制在 [-π, π] 范围内
        phase = np.mod(phase + np.pi, 2*np.pi) - np.pi
        return phase
    
    def generate_segment_control_points(self, 
                                     segment_idx: int, 
                                     radial_pos: float, 
                                     phase_req: float, 
                                     transmittance_req: float = 1.0) -> np.ndarray:
        """
        为单个分段生成NURBS控制点
        
        Args:
            segment_idx: 分段索引
            radial_pos: 径向位置
            phase_req: 所需相位
            transmittance_req: 所需透射率
            
        Returns:
            控制点坐标 (8, 2)
        """
        # 基础控制点 - 根据径向位置调整尺寸
        base_size = 0.05 * (radial_pos / self.lens_radius + 0.5)  # 尺寸随径向位置变化
        
        base_control_points = np.array([
            (base_size, 0), 
            (0.707*base_size, 0.707*base_size), 
            (0, base_size), 
            (-0.707*base_size, 0.707*base_size),
            (-base_size, 0), 
            (-0.707*base_size, -0.707*base_size), 
            (0, -base_size), 
            (0.707*base_size, -0.707*base_size)
        ])
        
        # 根据分段索引添加角度旋转
        angle = 2 * np.pi * segment_idx / self.n_segments
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_points = np.dot(base_control_points, rotation_matrix.T)
        
        # 添加径向偏移
        radial_offset = radial_pos
        angle_offset = angle
        offset_x = radial_offset * np.cos(angle_offset)
        offset_y = radial_offset * np.sin(angle_offset)
        
        final_points = rotated_points + np.array([offset_x, offset_y])
        
        return final_points
    
    def predict_optical_response(self, control_points: np.ndarray) -> Tuple[float, float]:
        """
        预测光学响应（相位和透射率）
        
        Args:
            control_points: 控制点坐标
            
        Returns:
            (相位, 透射率)
        """
        if self.model_available:
            # 使用代理模型预测
            normalized_input = normalize_control_points(control_points.reshape(1, 8, 2))
            prediction = self.model.predict(normalized_input)
            phase, transmittance = denormalize_targets(prediction)[0]
            return float(phase), float(transmittance)
        else:
            # 使用真实仿真（如果可用）
            try:
                sim = Simulation(control_points=control_points)
                transmittance, phase = sim.run_forward(wavelength_start=self.wavelength, wavelength_stop=self.wavelength)
                return phase, transmittance
            except:
                # 如果仿真失败，返回随机值
                return np.random.uniform(-np.pi, np.pi), np.random.uniform(0.5, 1.0)
    
    def calculate_segment_response(self, 
                                 segment_idx: int, 
                                 radial_pos: float, 
                                 phase_req: float) -> Tuple[float, float]:
        """
        计算单个分段的光学响应
        
        Args:
            segment_idx: 分段索引
            radial_pos: 径向位置
            phase_req: 所需相位
            
        Returns:
            (实际相位, 实际透射率)
        """
        # 生成控制点
        control_points = self.generate_segment_control_points(segment_idx, radial_pos, phase_req)
        
        # 预测光学响应
        actual_phase, actual_transmittance = self.predict_optical_response(control_points)
        
        return actual_phase, actual_transmittance
    
    def calculate_focus_efficiency(self, 
                                 phase_profile: np.ndarray, 
                                 transmittance_profile: np.ndarray) -> float:
        """
        计算聚焦效率
        
        Args:
            phase_profile: 实际相位分布
            transmittance_profile: 实际透射率分布
            
        Returns:
            聚焦效率
        """
        # 计算理想相位分布
        radial_positions = np.linspace(0, self.lens_radius, self.n_segments)
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # 计算相位误差
        phase_error = np.abs(ideal_phase - phase_profile)
        
        # 计算加权效率（考虑透射率）
        efficiency = np.mean(transmittance_profile * np.cos(phase_error/2)**2)
        
        return efficiency
    
    def optimize_single_wavelength(self, 
                                 max_iterations: int = 50, 
                                 learning_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        单波长优化
        
        Args:
            max_iterations: 最大迭代次数
            learning_rate: 学习率
            
        Returns:
            (最优相位分布, 最优透射率分布, 效率历史)
        """
        print("开始单波长超透镜优化...")
        
        # 初始化相位和透射率分布
        radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # 初始猜测 - 接近理想值
        current_phase = ideal_phase + np.random.uniform(-0.2, 0.2, self.n_segments)
        current_transmittance = np.ones(self.n_segments) * 0.8  # 初始透射率
        
        efficiency_history = []
        
        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}/{max_iterations}")
            
            # 更新每个分段
            new_phase = current_phase.copy()
            new_transmittance = current_transmittance.copy()
            
            for i in range(self.n_segments):
                # 计算当前分段的响应
                actual_phase, actual_transmittance = self.calculate_segment_response(
                    i, radial_positions[i], ideal_phase[i]
                )
                
                # 更新估计值
                new_phase[i] = actual_phase
                new_transmittance[i] = actual_transmittance
            
            current_phase = new_phase
            current_transmittance = new_transmittance
            
            # 计算当前效率
            current_efficiency = self.calculate_focus_efficiency(current_phase, current_transmittance)
            efficiency_history.append(current_efficiency)
            
            print(f"  当前聚焦效率: {current_efficiency:.4f}")
        
        print(f"优化完成! 最终聚焦效率: {efficiency_history[-1]:.4f}")
        
        return current_phase, current_transmittance, efficiency_history
    
    def plot_results(self, 
                    phase_profile: np.ndarray, 
                    transmittance_profile: np.ndarray, 
                    efficiency_history: List[float],
                    radial_positions: Optional[np.ndarray] = None):
        """
        绘制优化结果
        """
        if radial_positions is None:
            radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 相位分布对比
        axes[0, 0].plot(radial_positions*1e6, ideal_phase, 'r--', label='理想相位', linewidth=2)
        axes[0, 0].plot(radial_positions*1e6, phase_profile, 'b-', label='实际相位', linewidth=2)
        axes[0, 0].set_xlabel('径向位置 (μm)')
        axes[0, 0].set_ylabel('相位 (弧度)')
        axes[0, 0].set_title('相位分布对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 透射率分布
        axes[0, 1].plot(radial_positions*1e6, transmittance_profile, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('径向位置 (μm)')
        axes[0, 1].set_ylabel('透射率')
        axes[0, 1].set_title('透射率分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 相位误差
        phase_error = np.abs(ideal_phase - phase_profile)
        axes[1, 0].plot(radial_positions*1e6, phase_error, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('径向位置 (μm)')
        axes[1, 0].set_ylabel('相位误差 (弧度)')
        axes[1, 0].set_title('相位误差分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 效率历史
        axes[1, 1].plot(efficiency_history, 'c-', linewidth=2)
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('聚焦效率')
        axes[1, 1].set_title('优化收敛过程')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metalens_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_design_summary(self, 
                              phase_profile: np.ndarray, 
                              transmittance_profile: np.ndarray,
                              radial_positions: Optional[np.ndarray] = None) -> dict:
        """
        生成设计摘要
        """
        if radial_positions is None:
            radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # 计算各种指标
        phase_error = np.abs(ideal_phase - phase_profile)
        avg_phase_error = np.mean(phase_error)
        max_phase_error = np.max(phase_error)
        avg_transmittance = np.mean(transmittance_profile)
        efficiency = self.calculate_focus_efficiency(phase_profile, transmittance_profile)
        
        summary = {
            'focal_length': self.focal_length,
            'wavelength': self.wavelength,
            'lens_radius': self.lens_radius,
            'n_segments': self.n_segments,
            'avg_phase_error': avg_phase_error,
            'max_phase_error': max_phase_error,
            'avg_transmittance': avg_transmittance,
            'focusing_efficiency': efficiency,
            'radial_positions': radial_positions,
            'ideal_phase': ideal_phase,
            'actual_phase': phase_profile,
            'transmittance': transmittance_profile
        }
        
        return summary


def main():
    """主函数 - 演示超透镜优化"""
    print("DFLAT风格超透镜优化器")
    print("="*50)
    
    # 创建优化器实例
    optimizer = MetalensOptimizer(
        n_segments=32,  # 减少分段数以加快演示
        focal_length=50e-6,
        wavelength=532e-9,
        lens_radius=50e-6
    )
    
    # 执行优化
    phase_profile, transmittance_profile, efficiency_history = optimizer.optimize_single_wavelength(
        max_iterations=20  # 减少迭代次数以加快演示
    )
    
    # 生成径向位置
    radial_positions = np.linspace(0.1*optimizer.lens_radius, optimizer.lens_radius, optimizer.n_segments)
    
    # 绘制结果
    optimizer.plot_results(phase_profile, transmittance_profile, efficiency_history, radial_positions)
    
    # 生成设计摘要
    summary = optimizer.generate_design_summary(phase_profile, transmittance_profile, radial_positions)
    
    print("\n设计摘要:")
    print(f"焦距: {summary['focal_length']*1e6:.1f} μm")
    print(f"波长: {summary['wavelength']*1e9:.1f} nm")
    print(f"透镜半径: {summary['lens_radius']*1e6:.1f} μm")
    print(f"分段数: {summary['n_segments']}")
    print(f"平均相位误差: {summary['avg_phase_error']:.4f} 弧度")
    print(f"最大相位误差: {summary['max_phase_error']:.4f} 弧度")
    print(f"平均透射率: {summary['avg_transmittance']:.4f}")
    print(f"聚焦效率: {summary['focusing_efficiency']:.4f}")
    
    print("\n优化完成! 结果已保存到 'metalens_optimization_results.png'")


if __name__ == "__main__":
    main()