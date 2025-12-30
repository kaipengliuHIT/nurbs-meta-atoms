import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

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

def generate_sample_data(n_samples=1000, n_control_points=8):
    """
    生成示例数据用于训练
    在实际应用中，您需要使用真实的仿真数据
    """
    print(f"生成 {n_samples} 个样本的训练数据...")
    
    # 生成随机控制点 (围绕基本形状的小扰动)
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    control_points_list = []
    phase_trans_list = []
    
    for i in range(n_samples):
        # 在基础形状上添加随机扰动
        perturbation = np.random.uniform(-0.05, 0.05, (n_control_points, 2))
        current_control_points = base_control_points + perturbation
        
        # 确保控制点在合理范围内
        current_control_points = np.clip(current_control_points, -0.25, 0.25)
        
        control_points_list.append(current_control_points)
        
        # 生成模拟的相位和透射率值
        # 在实际应用中，这些应该来自真实的物理仿真
        phase = np.random.uniform(-np.pi, np.pi)  # 相位范围 [-π, π]
        transmittance = np.random.uniform(0.0, 1.0)  # 透射率范围 [0, 1]
        
        phase_trans_list.append([phase, transmittance])
        
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{n_samples} 个样本")
    
    control_points = np.array(control_points_list)
    targets = np.array(phase_trans_list)
    
    return control_points, targets


def generate_realistic_data(n_samples=500):
    """
    使用nurbs_atoms_data.py中的Simulation类生成更真实的数据
    注意：这会运行实际的仿真，可能比较耗时
    """
    print(f"使用Simulation类生成 {n_samples} 个真实数据样本...")
    
    control_points_list = []
    phase_trans_list = []
    
    for i in range(n_samples):
        # 生成随机控制点
        base_control_points = np.array([
            (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
            (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
        ])
        
        # 添加随机扰动
        perturbation = np.random.uniform(-0.03, 0.03, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        try:
            # 创建仿真对象并运行
            sim = Simulation(control_points=current_control_points)
            transmittance, phase = sim.run_forward(wavelength_start=500e-9, wavelength_stop=600e-9)
            
            control_points_list.append(current_control_points)
            phase_trans_list.append([phase, transmittance])
            
            if (i + 1) % 50 == 0:
                print(f"已生成 {i + 1}/{n_samples} 个真实样本，相位: {phase:.3f}, 透射率: {transmittance:.3f}")
                
        except Exception as e:
            print(f"仿真失败，样本 {i}: {e}")
            # 如果仿真失败，使用随机值作为备选
            control_points_list.append(current_control_points)
            phase = np.random.uniform(-np.pi, np.pi)
            transmittance = np.random.uniform(0.0, 1.0)
            phase_trans_list.append([phase, transmittance])
    
    return np.array(control_points_list), np.array(phase_trans_list)


def train_model():
    """训练Transformer模型"""
    print("开始训练NURBS Transformer代理模型...")
    
    # 生成训练数据
    # 注意：在实际应用中，您可能需要使用真实的仿真数据
    # 这里我们使用模拟数据作为示例
    print("生成训练数据...")
    control_points, targets = generate_sample_data(n_samples=2000)
    
    # 数据归一化
    print("归一化数据...")
    normalized_control_points = normalize_control_points(control_points)
    normalized_targets = normalize_targets(targets)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        normalized_control_points, normalized_targets, 
        test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = NURBSDataset(X_train, y_train)
    val_dataset = NURBSDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    
    # 创建模型
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    print("开始训练...")
    model.train(train_loader, val_loader, epochs=100, save_path="nurbs_transformer_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("模型训练完成！")


def test_model():
    """测试训练好的模型"""
    print("测试训练好的模型...")
    
    # 创建模型实例
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    # 加载训练好的模型
    if os.path.exists("nurbs_transformer_model.pth"):
        model.load_model("nurbs_transformer_model.pth")
        
        # 测试预测
        base_control_points = np.array([
            (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
            (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
        ])
        
        # 添加一些扰动
        test_control_points = base_control_points + np.random.uniform(-0.02, 0.02, (8, 2))
        test_control_points = np.clip(test_control_points, -0.2, 0.2)
        
        # 归一化控制点
        normalized_test_points = normalize_control_points(test_control_points.reshape(1, 8, 2))
        normalized_test_points = torch.FloatTensor(normalized_test_points)
        
        # 预测
        prediction = model.predict(normalized_test_points)
        
        # 反归一化预测结果
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        print(f"输入控制点: {test_control_points}")
        print(f"预测相位: {pred_phase:.4f}")
        print(f"预测透射率: {pred_trans:.4f}")
        
        # 与真实仿真对比（如果可用）
        try:
            sim = Simulation(control_points=test_control_points)
            true_trans, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            print(f"真实相位: {true_phase:.4f}")
            print(f"真实透射率: {true_trans:.4f}")
            print(f"相位误差: {abs(pred_phase - true_phase):.4f}")
            print(f"透射率误差: {abs(pred_trans - true_trans):.4f}")
        except:
            print("无法运行真实仿真进行对比")
    else:
        print("未找到训练好的模型文件，请先运行训练程序")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练NURBS Transformer代理模型')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.test:
        test_model()
    else:
        print("请指定 --train 或 --test 参数")
        print("示例: python train_transformer_model.py --train")
        print("示例: python train_transformer_model.py --test")