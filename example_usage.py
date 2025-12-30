import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# 添加当前目录到路径
sys.path.append('/mnt/e/pythoncode/nurbs-meta-atoms')

from transformer_nurbs_model import (
    NURBSTransformerModel, 
    NURBSDataset, 
    normalize_control_points, 
    denormalize_targets
)
from nurbs_atoms_data import Simulation

def example_complete_workflow():
    """完整的使用流程示例"""
    print("NURBS Transformer代理模型完整使用流程示例")
    print("="*60)
    
    # 1. 定义示例控制点
    example_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    print("1. 原始控制点:")
    print(example_control_points)
    
    # 2. 使用真实仿真获取基准值
    print("\n2. 运行真实物理仿真获取基准值...")
    try:
        sim = Simulation(control_points=example_control_points)
        true_transmittance, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
        print(f"   真实相位: {true_phase:.4f}")
        print(f"   真实透射率: {true_transmittance:.4f}")
    except Exception as e:
        print(f"   仿真失败: {e}")
        true_phase = 1.57  # 默认值
        true_transmittance = 0.8  # 默认值
    
    # 3. 创建并训练模型（如果模型不存在）
    model_path = "nurbs_transformer_model.pth"
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if os.path.exists(model_path):
        print(f"\n3. 加载预训练模型: {model_path}")
        model.load_model(model_path)
    else:
        print(f"\n3. 模型文件不存在，跳过加载 (需要先训练模型)")
    
    # 4. 使用Transformer模型预测
    print("\n4. 使用Transformer代理模型预测...")
    
    # 归一化控制点
    normalized_control_points = normalize_control_points(example_control_points.reshape(1, 8, 2))
    
    if os.path.exists(model_path):
        prediction = model.predict(normalized_control_points)
        from transformer_nurbs_model import denormalize_targets
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        print(f"   预测相位: {pred_phase:.4f}")
        print(f"   预测透射率: {pred_trans:.4f}")
        
        print(f"\n5. 预测精度评估:")
        phase_error = abs(pred_phase - true_phase)
        trans_error = abs(pred_trans - true_transmittance)
        print(f"   相位误差: {phase_error:.4f}")
        print(f"   透射率误差: {trans_error:.4f}")
        print(f"   相位相对误差: {abs(phase_error/true_phase)*100:.2f}%")
        print(f"   透射率相对误差: {abs(trans_error/true_transmittance)*100:.2f}%")
    else:
        print("   模型未训练，跳过预测")
    
    # 5. 展示不同控制点形状的影响
    print("\n6. 测试不同控制点形状的影响...")
    
    shapes = {
        "圆形": np.array([(0.15, 0), (0.106, 0.106), (0, 0.15), (-0.106, 0.106), 
                         (-0.15, 0), (-0.106, -0.106), (0, -0.15), (0.106, -0.106)]),
        "方形": np.array([(0.15, 0.05), (0.15, 0.15), (0.05, 0.15), (-0.05, 0.15), 
                         (-0.15, 0.15), (-0.15, -0.15), (-0.05, -0.15), (0.05, -0.15)]),
        "椭圆形": np.array([(0.20, 0), (0.141, 0.10), (0, 0.12), (-0.141, 0.10), 
                           (-0.20, 0), (-0.141, -0.10), (0, -0.12), (0.141, -0.10)])
    }
    
    results = {}
    for shape_name, shape_control_points in shapes.items():
        print(f"\n   {shape_name}形状:")
        
        # 使用真实仿真
        try:
            sim_shape = Simulation(control_points=shape_control_points)
            shape_trans, shape_phase = sim_shape.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            print(f"     真实 - 相位: {shape_phase:.4f}, 透射率: {shape_trans:.4f}")
        except:
            shape_phase, shape_trans = 0, 0
            print(f"     真实 - 仿真失败")
        
        # 使用Transformer预测
        if os.path.exists(model_path):
            norm_shape_points = normalize_control_points(shape_control_points.reshape(1, 8, 2))
            shape_pred = model.predict(norm_shape_points)
            shape_pred_phase, shape_pred_trans = denormalize_targets(shape_pred)[0]
            print(f"     预测 - 相位: {shape_pred_phase:.4f}, 透射率: {shape_pred_trans:.4f}")
        
        results[shape_name] = {
            'control_points': shape_control_points,
            'true_phase': shape_phase,
            'true_trans': shape_trans
        }
    
    # 6. 可视化结果
    print("\n7. 生成可视化图表...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制原始形状
    ax = axes[0, 0]
    sim_orig = Simulation(control_points=example_control_points)
    nurbs_orig = sim_orig.generate_complete_nurbs_curve(example_control_points)
    x_orig = [p[0] for p in nurbs_orig] + [nurbs_orig[0][0]]
    y_orig = [p[1] for p in nurbs_orig] + [nurbs_orig[0][1]]
    ax.plot(x_orig, y_orig, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(example_control_points[:, 0], example_control_points[:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Original Shape\nPhase: {true_phase:.3f}, Trans: {true_transmittance:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 绘制圆形
    ax = axes[0, 1]
    sim_circle = Simulation(control_points=shapes["圆形"])
    nurbs_circle = sim_circle.generate_complete_nurbs_curve(shapes["圆形"])
    x_circle = [p[0] for p in nurbs_circle] + [nurbs_circle[0][0]]
    y_circle = [p[1] for p in nurbs_circle] + [nurbs_circle[0][1]]
    ax.plot(x_circle, y_circle, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["圆形"][:, 0], shapes["圆形"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Circle Shape\nPhase: {results["圆形"]["true_phase"]:.3f}, Trans: {results["圆形"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 绘制方形
    ax = axes[1, 0]
    sim_square = Simulation(control_points=shapes["方形"])
    nurbs_square = sim_square.generate_complete_nurbs_curve(shapes["方形"])
    x_square = [p[0] for p in nurbs_square] + [nurbs_square[0][0]]
    y_square = [p[1] for p in nurbs_square] + [nurbs_square[0][1]]
    ax.plot(x_square, y_square, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["方形"][:, 0], shapes["方形"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Square Shape\nPhase: {results["方形"]["true_phase"]:.3f}, Trans: {results["方形"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 绘制椭圆形
    ax = axes[1, 1]
    sim_ellipse = Simulation(control_points=shapes["椭圆形"])
    nurbs_ellipse = sim_ellipse.generate_complete_nurbs_curve(shapes["椭圆形"])
    x_ellipse = [p[0] for p in nurbs_ellipse] + [nurbs_ellipse[0][0]]
    y_ellipse = [p[1] for p in nurbs_ellipse] + [nurbs_ellipse[0][1]]
    ax.plot(x_ellipse, y_ellipse, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["椭圆形"][:, 0], shapes["椭圆形"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Ellipse Shape\nPhase: {results["椭圆形"]["true_phase"]:.3f}, Trans: {results["椭圆形"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('nurbs_shapes_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n完成! 生成的图像已保存为 'nurbs_shapes_comparison.png'")


def quick_demo():
    """快速演示"""
    print("快速演示 - Transformer代理模型预测")
    print("="*40)
    
    # 创建模型实例
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    # 示例控制点
    control_points = np.array([
        (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
        (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
    ])
    
    print(f"输入控制点:\n{control_points}")
    
    # 归一化
    normalized_input = normalize_control_points(control_points.reshape(1, 8, 2))
    
    # 检查模型文件是否存在
    model_path = "nurbs_transformer_model.pth"
    if os.path.exists(model_path):
        model.load_model(model_path)
        prediction = model.predict(normalized_input)
        from transformer_nurbs_model import denormalize_targets
        phase, transmittance = denormalize_targets(prediction)[0]
        print(f"\n预测结果:")
        print(f"相位: {phase:.4f} 弧度 ({np.degrees(phase):.2f}°)")
        print(f"透射率: {transmittance:.4f}")
    else:
        print(f"\n模型文件 {model_path} 不存在")
        print("请先运行训练脚本: python train_transformer_model.py --train")
        print("\n使用随机预测值作为演示:")
        phase = np.random.uniform(-np.pi, np.pi)
        transmittance = np.random.uniform(0.0, 1.0)
        print(f"相位: {phase:.4f} 弧度 ({np.degrees(phase):.2f}°)")
        print(f"透射率: {transmittance:.4f}")
    
    print(f"\nTransformer模型架构特点:")
    print(f"- 输入: 8个控制点的(x,y)坐标 -> (8, 2)张量")
    print(f"- 使用Transformer编码器处理序列化控制点")
    print(f"- 输出: 相位和透射率 -> (2,)张量")
    print(f"- 优势: 能够捕捉控制点之间的长距离依赖关系")


if __name__ == "__main__":
    print("NURBS Transformer代理模型示例程序")
    print("=====================================")
    
    while True:
        print("\n请选择演示模式:")
        print("1. 完整工作流程演示")
        print("2. 快速演示")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            example_complete_workflow()
        elif choice == '2':
            quick_demo()
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")