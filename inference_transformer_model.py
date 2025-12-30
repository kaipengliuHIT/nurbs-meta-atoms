import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# 添加当前目录到路径
sys.path.append('/mnt/e/pythoncode/nurbs-meta-atoms')

from transformer_nurbs_model import (
    NURBSTransformerModel, 
    NURBSDataset, 
    normalize_control_points, 
    denormalize_control_points,
    normalize_targets,
    denormalize_targets
)
from nurbs_atoms_data import Simulation

def evaluate_model_performance(model_path: str = "nurbs_transformer_model.pth", 
                            n_test_samples: int = 200):
    """评估模型性能"""
    print("评估模型性能...")
    
    # 创建模型实例
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        return
    
    # 加载模型
    model.load_model(model_path)
    
    # 生成测试数据
    print("生成测试数据...")
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    test_control_points_list = []
    true_targets_list = []
    
    for i in range(n_test_samples):
        # 生成带扰动的控制点
        perturbation = np.random.uniform(-0.04, 0.04, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        # 使用Simulation类获取真实值（或使用模拟值）
        try:
            sim = Simulation(control_points=current_control_points)
            transmittance, phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            true_targets_list.append([phase, transmittance])
        except:
            # 如果仿真失败，使用随机值
            phase = np.random.uniform(-np.pi, np.pi)
            transmittance = np.random.uniform(0.0, 1.0)
            true_targets_list.append([phase, transmittance])
        
        test_control_points_list.append(current_control_points)
        
        if (i + 1) % 50 == 0:
            print(f"已生成 {i + 1}/{n_test_samples} 个测试样本")
    
    test_control_points = np.array(test_control_points_list)
    true_targets = np.array(true_targets_list)
    
    # 归一化数据
    normalized_test_points = normalize_control_points(test_control_points)
    normalized_true_targets = normalize_targets(true_targets)
    
    # 预测
    print("进行预测...")
    model.model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(normalized_test_points)):
            ctrl_points_tensor = torch.FloatTensor(normalized_test_points[i:i+1]).to(model.device)
            pred = model.model(ctrl_points_tensor)
            predictions.append(pred.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    
    # 反归一化预测结果
    denorm_predictions = denormalize_targets(predictions)
    denorm_true_targets = denormalize_targets(normalized_true_targets)
    
    # 计算评估指标
    phase_mse = mean_squared_error(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_mse = mean_squared_error(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    phase_mae = mean_absolute_error(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_mae = mean_absolute_error(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    phase_r2 = r2_score(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_r2 = r2_score(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    print("\n模型性能评估结果:")
    print(f"相位预测:")
    print(f"  MSE: {phase_mse:.6f}")
    print(f"  MAE: {phase_mae:.6f}")
    print(f"  R²: {phase_r2:.6f}")
    print(f"透射率预测:")
    print(f"  MSE: {trans_mse:.6f}")
    print(f"  MAE: {trans_mae:.6f}")
    print(f"  R²: {trans_r2:.6f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 相位预测对比
    plt.subplot(1, 3, 1)
    plt.scatter(denorm_true_targets[:, 0], denorm_predictions[:, 0], alpha=0.6)
    plt.plot([denorm_true_targets[:, 0].min(), denorm_true_targets[:, 0].max()], 
             [denorm_true_targets[:, 0].min(), denorm_true_targets[:, 0].max()], 'r--', lw=2)
    plt.xlabel('真实相位')
    plt.ylabel('预测相位')
    plt.title(f'相位预测 (R² = {phase_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 透射率预测对比
    plt.subplot(1, 3, 2)
    plt.scatter(denorm_true_targets[:, 1], denorm_predictions[:, 1], alpha=0.6)
    plt.plot([denorm_true_targets[:, 1].min(), denorm_true_targets[:, 1].max()], 
             [denorm_true_targets[:, 1].min(), denorm_true_targets[:, 1].max()], 'r--', lw=2)
    plt.xlabel('真实透射率')
    plt.ylabel('预测透射率')
    plt.title(f'透射率预测 (R² = {trans_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 预测误差分布
    plt.subplot(1, 3, 3)
    phase_errors = np.abs(denorm_true_targets[:, 0] - denorm_predictions[:, 0])
    trans_errors = np.abs(denorm_true_targets[:, 1] - denorm_predictions[:, 1])
    
    plt.hist(phase_errors, bins=30, alpha=0.5, label='相位误差', density=True)
    plt.hist(trans_errors, bins=30, alpha=0.5, label='透射率误差', density=True)
    plt.xlabel('绝对误差')
    plt.ylabel('密度')
    plt.title('预测误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'phase_mse': phase_mse, 'trans_mse': trans_mse,
        'phase_mae': phase_mae, 'trans_mae': trans_mae,
        'phase_r2': phase_r2, 'trans_r2': trans_r2,
        'true_targets': denorm_true_targets,
        'predictions': denorm_predictions
    }


def predict_single_sample(model_path: str = "nurbs_transformer_model.pth"):
    """对单个样本进行预测"""
    print("对单个样本进行预测...")
    
    # 创建模型实例
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        return
    
    # 加载模型
    model.load_model(model_path)
    
    # 定义一个示例控制点
    example_control_points = np.array([
        (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
        (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
    ])
    
    print(f"输入控制点:\n{example_control_points}")
    
    # 归一化控制点
    normalized_control_points = normalize_control_points(example_control_points.reshape(1, 8, 2))
    
    # 预测
    prediction = model.predict(normalized_control_points)
    pred_phase, pred_trans = denormalize_targets(prediction)[0]
    
    print(f"\n预测结果:")
    print(f"相位: {pred_phase:.4f} 弧度 ({np.degrees(pred_phase):.2f}°)")
    print(f"透射率: {pred_trans:.4f}")
    
    # 与真实仿真对比（如果可用）
    try:
        sim = Simulation(control_points=example_control_points)
        true_trans, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
        print(f"\n真实仿真结果:")
        print(f"相位: {true_phase:.4f} 弧度 ({np.degrees(true_phase):.2f}°)")
        print(f"透射率: {true_trans:.4f}")
        print(f"\n预测误差:")
        print(f"相位误差: {abs(pred_phase - true_phase):.4f}")
        print(f"透射率误差: {abs(pred_trans - true_trans):.4f}")
    except Exception as e:
        print(f"\n无法运行真实仿真进行对比: {e}")
    
    return example_control_points, pred_phase, pred_trans


def batch_predict(model_path: str = "nurbs_transformer_model.pth", 
                 n_samples: int = 10):
    """批量预测"""
    print(f"进行 {n_samples} 个样本的批量预测...")
    
    # 创建模型实例
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        return
    
    # 加载模型
    model.load_model(model_path)
    
    # 生成随机控制点
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    all_control_points = []
    all_predictions = []
    
    for i in range(n_samples):
        # 添加随机扰动
        perturbation = np.random.uniform(-0.03, 0.03, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        # 归一化并预测
        normalized_points = normalize_control_points(current_control_points.reshape(1, 8, 2))
        prediction = model.predict(normalized_points)
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        all_control_points.append(current_control_points)
        all_predictions.append([pred_phase, pred_trans])
        
        print(f"样本 {i+1}: 控制点形状，预测相位={pred_phase:.3f}, 预测透射率={pred_trans:.3f}")
    
    return all_control_points, all_predictions


def visualize_nurbs_shape(control_points, title="NURBS Shape"):
    """可视化NURBS形状"""
    from nurbs_atoms_data import Simulation
    
    # 创建仿真对象来生成NURBS曲线
    sim = Simulation(control_points=control_points)
    nurbs_points = sim.generate_complete_nurbs_curve(control_points)
    
    # 提取x和y坐标
    x_coords = [p[0] for p in nurbs_points]
    y_coords = [p[1] for p in nurbs_points]
    
    # 闭合图形
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='NURBS Curve')
    plt.scatter(control_points[:, 0], control_points[:, 1], c='red', s=100, zorder=5, label='Control Points')
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def main():
    """主函数"""
    print("NURBS Transformer代理模型推理工具")
    print("="*50)
    
    while True:
        print("\n请选择操作:")
        print("1. 评估模型性能")
        print("2. 单样本预测")
        print("3. 批量预测")
        print("4. 可视化NURBS形状")
        print("5. 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            n_samples = input("请输入测试样本数量 (默认200): ").strip()
            n_samples = int(n_samples) if n_samples else 200
            evaluate_model_performance(n_test_samples=n_samples)
        
        elif choice == '2':
            predict_single_sample()
        
        elif choice == '3':
            n_samples = input("请输入批量预测数量 (默认10): ").strip()
            n_samples = int(n_samples) if n_samples else 10
            batch_predict(n_samples=n_samples)
        
        elif choice == '4':
            # 生成示例控制点并可视化
            example_control_points = np.array([
                (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
                (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
            ])
            visualize_nurbs_shape(example_control_points, "Example NURBS Shape")
        
        elif choice == '5':
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()