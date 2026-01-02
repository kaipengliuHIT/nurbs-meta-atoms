"""
改进版NURBS Transformer代理模型
支持波长作为条件输入，适用于400-700nm波段的超表面仿真

主要改进:
1. 波长作为条件输入，可以查询任意波长的响应
2. 增大模型容量以适应50万样本的训练
3. 添加傅里叶特征编码提升波长表示能力
4. 支持多波长批量预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple, Optional, List


class FourierFeatureEncoding(nn.Module):
    """
    傅里叶特征编码，用于提升波长等连续变量的表示能力
    参考: Fourier Features Let Networks Learn High Frequency Functions
    """
    def __init__(self, input_dim: int = 1, num_frequencies: int = 32, scale: float = 10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # 随机傅里叶特征
        B = torch.randn(input_dim, num_frequencies) * scale
        self.register_buffer('B', B)
        self.output_dim = num_frequencies * 2  # sin + cos
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_dim)
        returns: (..., num_frequencies * 2)
        """
        x_proj = 2 * math.pi * x @ self.B  # (..., num_frequencies)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class WavelengthConditionedNURBSTransformer(nn.Module):
    """
    波长条件化的NURBS Transformer代理模型
    
    输入:
        - control_points: (batch, 8, 2) NURBS控制点坐标
        - wavelength: (batch, 1) 波长 (nm)，范围400-700nm
    
    输出:
        - (batch, 2) 相位和透射率
    """
    def __init__(self, 
                 input_dim: int = 2,
                 n_control_points: int = 8,
                 d_model: int = 256,        # 增大模型维度
                 nhead: int = 8,
                 num_layers: int = 8,       # 增加层数
                 d_ff: int = 1024,          # 增大FFN维度
                 dropout: float = 0.1,
                 wavelength_encoding_dim: int = 64,
                 output_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_control_points = n_control_points
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 波长编码
        self.wavelength_fourier = FourierFeatureEncoding(
            input_dim=1, 
            num_frequencies=wavelength_encoding_dim // 2,
            scale=10.0
        )
        self.wavelength_projection = nn.Sequential(
            nn.Linear(wavelength_encoding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 控制点编码
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_control_points + 1)
        
        # 可学习的CLS token（用于聚合全局信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 分别处理相位和透射率的输出激活
        # 相位: 无限制 (后续会wrap到[-π, π])
        # 透射率: sigmoid确保在[0, 1]
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, control_points: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            control_points: (batch, n_control_points, 2) 控制点坐标
            wavelength: (batch, 1) 波长 (nm)，需要归一化到[0, 1]
        
        Returns:
            output: (batch, 2) [phase, transmittance]
        """
        batch_size = control_points.size(0)
        
        # 编码波长
        wavelength_features = self.wavelength_fourier(wavelength)  # (batch, wavelength_encoding_dim)
        wavelength_token = self.wavelength_projection(wavelength_features)  # (batch, d_model)
        wavelength_token = wavelength_token.unsqueeze(1)  # (batch, 1, d_model)
        
        # 编码控制点
        point_features = self.input_projection(control_points)  # (batch, n_points, d_model)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # 拼接: [CLS, wavelength, point1, point2, ..., point8]
        x = torch.cat([cls_tokens, wavelength_token, point_features], dim=1)  # (batch, 10, d_model)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, 10, d_model)
        
        # 使用CLS token的输出
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # 输出投影
        output = self.output_head(cls_output)  # (batch, 2)
        
        # 对透射率应用sigmoid
        phase = output[:, 0:1]  # (batch, 1)
        transmittance = torch.sigmoid(output[:, 1:2])  # (batch, 1), 确保在[0, 1]
        
        return torch.cat([phase, transmittance], dim=1)


class WavelengthConditionedDataset(Dataset):
    """
    波长条件化的数据集
    
    支持两种数据格式:
    1. 单波长数据: 每个样本对应一个波长
    2. 多波长数据: 每个控制点对应多个波长的响应
    """
    def __init__(self, 
                 control_points: np.ndarray,
                 wavelengths: np.ndarray,
                 phases: np.ndarray,
                 transmittances: np.ndarray,
                 wavelength_min: float = 400.0,
                 wavelength_max: float = 700.0,
                 point_min: float = -0.25,
                 point_max: float = 0.25):
        """
        Args:
            control_points: (n_samples, 8, 2) 控制点坐标
            wavelengths: (n_samples,) 波长 (nm)
            phases: (n_samples,) 相位 (rad)
            transmittances: (n_samples,) 透射率
        """
        # 归一化控制点到[0, 1]
        self.control_points = torch.FloatTensor(
            (control_points - point_min) / (point_max - point_min)
        )
        
        # 归一化波长到[0, 1]
        self.wavelengths = torch.FloatTensor(
            (wavelengths - wavelength_min) / (wavelength_max - wavelength_min)
        ).unsqueeze(-1)  # (n_samples, 1)
        
        # 归一化相位到[0, 1] (从[-π, π])
        self.phases = torch.FloatTensor(
            (phases + np.pi) / (2 * np.pi)
        )
        
        # 透射率已经在[0, 1]
        self.transmittances = torch.FloatTensor(transmittances)
        
        # 存储归一化参数
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.point_min = point_min
        self.point_max = point_max
    
    def __len__(self):
        return len(self.control_points)
    
    def __getitem__(self, idx):
        return (
            self.control_points[idx],
            self.wavelengths[idx],
            torch.stack([self.phases[idx], self.transmittances[idx]])
        )


class WavelengthConditionedNURBSModel:
    """
    波长条件化NURBS代理模型的训练和推理封装
    """
    def __init__(self,
                 n_control_points: int = 8,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 wavelength_encoding_dim: int = 64,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = WavelengthConditionedNURBSTransformer(
            n_control_points=n_control_points,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            wavelength_encoding_dim=wavelength_encoding_dim
        ).to(self.device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 使用组合损失：MSE + 相位周期性损失
        self.mse_loss = nn.MSELoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        self.train_losses = []
        self.val_losses = []
        
        # 归一化参数
        self.wavelength_min = 400.0
        self.wavelength_max = 700.0
        self.point_min = -0.25
        self.point_max = 0.25
    
    def phase_loss(self, pred_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
        """
        相位损失，考虑周期性
        相位在[-π, π]范围内，需要处理边界情况
        """
        # 将归一化的相位转换回弧度
        pred_rad = pred_phase * 2 * np.pi - np.pi
        target_rad = target_phase * 2 * np.pi - np.pi
        
        # 计算相位差，考虑周期性
        diff = pred_rad - target_rad
        # 将差值wrap到[-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return torch.mean(diff ** 2)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for control_points, wavelengths, targets in train_loader:
            control_points = control_points.to(self.device)
            wavelengths = wavelengths.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(control_points, wavelengths)
            
            # 分别计算相位和透射率损失
            phase_l = self.phase_loss(outputs[:, 0], targets[:, 0])
            trans_l = self.mse_loss(outputs[:, 1], targets[:, 1])
            
            # 组合损失（可以调整权重）
            loss = phase_l + trans_l
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        phase_errors = []
        trans_errors = []
        
        with torch.no_grad():
            for control_points, wavelengths, targets in val_loader:
                control_points = control_points.to(self.device)
                wavelengths = wavelengths.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(control_points, wavelengths)
                
                phase_l = self.phase_loss(outputs[:, 0], targets[:, 0])
                trans_l = self.mse_loss(outputs[:, 1], targets[:, 1])
                loss = phase_l + trans_l
                
                total_loss += loss.item()
                
                # 计算实际误差（反归一化后）
                pred_phase = outputs[:, 0].cpu().numpy() * 2 * np.pi - np.pi
                true_phase = targets[:, 0].cpu().numpy() * 2 * np.pi - np.pi
                phase_diff = np.arctan2(np.sin(pred_phase - true_phase), 
                                        np.cos(pred_phase - true_phase))
                phase_errors.extend(np.abs(phase_diff))
                
                trans_errors.extend(
                    np.abs(outputs[:, 1].cpu().numpy() - targets[:, 1].cpu().numpy())
                )
        
        avg_loss = total_loss / len(val_loader)
        avg_phase_error = np.mean(phase_errors)  # rad
        avg_trans_error = np.mean(trans_errors)
        
        return avg_loss, avg_phase_error, avg_trans_error
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 200,
              save_path: str = "best_wavelength_conditioned_model.pth",
              early_stopping_patience: int = 30):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, phase_error, trans_error = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'phase_error': phase_error,
                    'trans_error': trans_error,
                }, save_path)
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, '
                      f'Phase Error: {np.degrees(phase_error):.2f}°, '
                      f'Trans Error: {trans_error:.4f}')
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    def predict(self, 
                control_points: np.ndarray, 
                wavelength_nm: float) -> Tuple[float, float]:
        """
        预测单个样本
        
        Args:
            control_points: (8, 2) 控制点坐标 (μm)
            wavelength_nm: 波长 (nm)
        
        Returns:
            phase: 相位 (rad)
            transmittance: 透射率
        """
        self.model.eval()
        
        with torch.no_grad():
            # 归一化
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            wl_norm = (wavelength_nm - self.wavelength_min) / (self.wavelength_max - self.wavelength_min)
            
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            wl_tensor = torch.FloatTensor([[wl_norm]]).to(self.device)
            
            output = self.model(cp_tensor, wl_tensor)
            
            # 反归一化
            phase = output[0, 0].item() * 2 * np.pi - np.pi
            transmittance = output[0, 1].item()
            
            return phase, transmittance
    
    def predict_spectrum(self,
                        control_points: np.ndarray,
                        wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测多个波长的响应（光谱）
        
        Args:
            control_points: (8, 2) 控制点坐标 (μm)
            wavelengths: (n_wavelengths,) 波长数组 (nm)
        
        Returns:
            phases: (n_wavelengths,) 相位数组 (rad)
            transmittances: (n_wavelengths,) 透射率数组
        """
        self.model.eval()
        
        with torch.no_grad():
            # 归一化控制点
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            cp_tensor = cp_tensor.expand(len(wavelengths), -1, -1)  # (n_wl, 8, 2)
            
            # 归一化波长
            wl_norm = (wavelengths - self.wavelength_min) / (self.wavelength_max - self.wavelength_min)
            wl_tensor = torch.FloatTensor(wl_norm).unsqueeze(-1).to(self.device)  # (n_wl, 1)
            
            output = self.model(cp_tensor, wl_tensor)  # (n_wl, 2)
            
            # 反归一化
            phases = output[:, 0].cpu().numpy() * 2 * np.pi - np.pi
            transmittances = output[:, 1].cpu().numpy()
            
            return phases, transmittances
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")


def estimate_model_capacity():
    """
    估算模型容量和训练数据需求
    """
    print("=" * 60)
    print("模型容量分析")
    print("=" * 60)
    
    # 创建模型
    model = WavelengthConditionedNURBSTransformer(
        d_model=256,
        nhead=8,
        num_layers=8,
        d_ff=1024
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\n经验法则分析:")
    print(f"  - 通常需要 10-100 倍于参数量的训练样本")
    print(f"  - 最小建议样本数: {total_params * 10:,} ({total_params * 10 / 1e6:.1f}M)")
    print(f"  - 推荐样本数: {total_params * 50:,} ({total_params * 50 / 1e6:.1f}M)")
    
    print(f"\n对于50万样本:")
    samples = 500000
    ratio = samples / total_params
    print(f"  - 样本/参数比: {ratio:.1f}")
    if ratio >= 50:
        print(f"  - ✅ 样本量充足，模型可以充分训练")
    elif ratio >= 10:
        print(f"  - ⚠️ 样本量适中，建议使用正则化和数据增强")
    else:
        print(f"  - ❌ 样本量不足，建议减小模型或增加数据")
    
    print(f"\n不同模型配置对比:")
    configs = [
        ("小型 (d=128, L=4)", 128, 4, 512),
        ("中型 (d=256, L=6)", 256, 6, 1024),
        ("大型 (d=256, L=8)", 256, 8, 1024),
        ("超大 (d=512, L=8)", 512, 8, 2048),
    ]
    
    for name, d_model, num_layers, d_ff in configs:
        m = WavelengthConditionedNURBSTransformer(
            d_model=d_model, num_layers=num_layers, d_ff=d_ff
        )
        params = sum(p.numel() for p in m.parameters())
        ratio = samples / params
        status = "✅" if ratio >= 10 else "⚠️"
        print(f"  {name}: {params:,} params, 比例={ratio:.1f} {status}")
    
    return total_params


if __name__ == "__main__":
    # 运行容量分析
    estimate_model_capacity()
    
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    # 创建模型
    model_wrapper = WavelengthConditionedNURBSModel(
        d_model=256,
        nhead=8,
        num_layers=8,
        d_ff=1024
    )
    
    # 示例预测
    control_points = np.array([
        (0.16, 0.02), (0.14, 0.14), (0.02, 0.16), (-0.14, 0.14),
        (-0.16, -0.02), (-0.14, -0.14), (-0.02, -0.16), (0.14, -0.14)
    ])
    
    print(f"\n示例控制点:\n{control_points}")
    
    # 单波长预测
    phase, trans = model_wrapper.predict(control_points, wavelength_nm=550)
    print(f"\n550nm波长预测 (未训练模型，随机输出):")
    print(f"  相位: {phase:.4f} rad ({np.degrees(phase):.2f}°)")
    print(f"  透射率: {trans:.4f}")
    
    # 光谱预测
    wavelengths = np.linspace(400, 700, 31)
    phases, transmittances = model_wrapper.predict_spectrum(control_points, wavelengths)
    print(f"\n光谱预测 (400-700nm, 31个点):")
    print(f"  相位范围: {np.min(phases):.2f} ~ {np.max(phases):.2f} rad")
    print(f"  透射率范围: {np.min(transmittances):.4f} ~ {np.max(transmittances):.4f}")

