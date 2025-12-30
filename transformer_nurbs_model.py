import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class NURBSTransformer(nn.Module):
    """基于Transformer的NURBS代理模型"""
    def __init__(self, 
                 input_dim: int = 2,  # 每个控制点的维度 (x, y)
                 n_control_points: int = 8,  # 控制点数量
                 d_model: int = 64,  # 模型维度
                 nhead: int = 8,  # 注意力头数
                 num_layers: int = 6,  # Transformer层数
                 d_ff: int = 256,  # 前馈网络维度
                 dropout: float = 0.1,
                 output_dim: int = 2):  # 输出维度 (phase, transmittance)
        super(NURBSTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.n_control_points = n_control_points
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_control_points)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * n_control_points, d_model * 2),  # 扩展特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),  # 压缩
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)  # phase and transmittance
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (batch_size, n_control_points, input_dim) 控制点坐标
        Returns:
            output: (batch_size, output_dim) phase and transmittance
        """
        batch_size, n_points, input_dim = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, n_points, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # (n_points, batch_size, d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (n_points, batch_size, d_model)
        x = x.transpose(0, 1)  # (batch_size, n_points, d_model)
        
        # 展平特征
        x = x.reshape(batch_size, -1)  # (batch_size, n_points * d_model)
        
        # 输出投影
        output = self.output_projection(x)  # (batch_size, output_dim)
        
        return output


class NURBSDataset(Dataset):
    """NURBS数据集类"""
    def __init__(self, control_points: np.ndarray, targets: np.ndarray):
        """
        Args:
            control_points: (n_samples, n_control_points, 2) 控制点坐标
            targets: (n_samples, 2) phase and transmittance
        """
        self.control_points = torch.FloatTensor(control_points)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.control_points)
    
    def __getitem__(self, idx):
        return self.control_points[idx], self.targets[idx]


def normalize_control_points(control_points: np.ndarray, 
                           min_val: float = -0.2, 
                           max_val: float = 0.2) -> np.ndarray:
    """
    归一化控制点坐标
    """
    return (control_points - min_val) / (max_val - min_val)


def denormalize_control_points(normalized_points: np.ndarray, 
                             min_val: float = -0.2, 
                             max_val: float = 0.2) -> np.ndarray:
    """
    反归一化控制点坐标
    """
    return normalized_points * (max_val - min_val) + min_val


def normalize_targets(targets: np.ndarray, 
                     phase_min: float = -np.pi, 
                     phase_max: float = np.pi,
                     trans_min: float = 0.0,
                     trans_max: float = 1.0) -> np.ndarray:
    """
    归一化目标值 (相位和透射率)
    """
    normalized = targets.copy()
    # 归一化相位到 [0, 1]
    normalized[:, 0] = (targets[:, 0] - phase_min) / (phase_max - phase_min)
    # 归一化透射率到 [0, 1]
    normalized[:, 1] = (targets[:, 1] - trans_min) / (trans_max - trans_min)
    return normalized


def denormalize_targets(normalized_targets: np.ndarray,
                       phase_min: float = -np.pi, 
                       phase_max: float = np.pi,
                       trans_min: float = 0.0,
                       trans_max: float = 1.0) -> np.ndarray:
    """
    反归一化目标值
    """
    targets = normalized_targets.copy()
    # 反归一化相位
    targets[:, 0] = normalized_targets[:, 0] * (phase_max - phase_min) + phase_min
    # 反归一化透射率
    targets[:, 1] = normalized_targets[:, 1] * (trans_max - trans_min) + trans_min
    return targets


class NURBSTransformerModel:
    """NURBS Transformer代理模型包装器"""
    def __init__(self, 
                 n_control_points: int = 8,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 256,
                 dropout: float = 0.1):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_control_points = n_control_points
        
        self.model = NURBSTransformer(
            n_control_points=n_control_points,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: int = 100,
              save_path: str = "best_nurbs_transformer.pth"):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_points, batch_targets in train_loader:
                batch_points = batch_points.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_points)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_points, batch_targets in val_loader:
                    batch_points = batch_points.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_points)
                    loss = self.criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            self.scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, save_path)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
    
    def predict(self, control_points: np.ndarray) -> np.ndarray:
        """预测相位和透射率"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(control_points, np.ndarray):
                control_points = torch.FloatTensor(control_points).unsqueeze(0)  # 添加批次维度
            control_points = control_points.to(self.device)
            
            outputs = self.model(control_points)
            return outputs.cpu().numpy()
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {model_path} 加载")
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        phase_errors = []
        trans_errors = []
        
        with torch.no_grad():
            for batch_points, batch_targets in test_loader:
                batch_points = batch_points.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_points)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item() * batch_points.size(0)
                total_samples += batch_points.size(0)
                
                # 计算相位和透射率的单独误差
                phase_errors.extend(torch.abs(outputs[:, 0] - batch_targets[:, 0]).cpu().numpy())
                trans_errors.extend(torch.abs(outputs[:, 1] - batch_targets[:, 1]).cpu().numpy())
        
        avg_loss = total_loss / total_samples
        avg_phase_error = np.mean(phase_errors)
        avg_trans_error = np.mean(trans_errors)
        
        return avg_loss, avg_phase_error, avg_trans_error


if __name__ == "__main__":
    print("NURBS Transformer代理模型定义完成")
    print("模型结构:")
    print("- 输入: 控制点坐标 (N, 2) -> NURBS形状")
    print("- 输出: 相位和透射率 (2,)")
    print("- 使用Transformer架构处理控制点序列")