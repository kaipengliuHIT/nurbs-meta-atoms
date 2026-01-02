"""
Improved NURBS Transformer Surrogate Model
Supports wavelength as conditional input for 400-700nm metasurface simulation

Key improvements:
1. Wavelength as conditional input for querying response at any wavelength
2. Increased model capacity for training with 500k samples
3. Fourier feature encoding for better wavelength representation
4. Support for multi-wavelength batch prediction
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
    Fourier feature encoding for improving representation of continuous variables like wavelength
    Reference: Fourier Features Let Networks Learn High Frequency Functions
    """
    def __init__(self, input_dim: int = 1, num_frequencies: int = 32, scale: float = 10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Random Fourier features
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
    """Positional encoding module"""
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
    Wavelength-conditioned NURBS Transformer surrogate model
    
    Input:
        - control_points: (batch, 8, 2) NURBS control point coordinates
        - wavelength: (batch, 1) wavelength (nm), range 400-700nm
    
    Output:
        - (batch, 2) phase and transmittance
    """
    def __init__(self, 
                 input_dim: int = 2,
                 n_control_points: int = 8,
                 d_model: int = 256,        # Increased model dimension
                 nhead: int = 8,
                 num_layers: int = 8,       # Increased number of layers
                 d_ff: int = 1024,          # Increased FFN dimension
                 dropout: float = 0.1,
                 wavelength_encoding_dim: int = 64,
                 output_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_control_points = n_control_points
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Wavelength encoding
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
        
        # Control point encoding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_control_points + 1)
        
        # Learnable CLS token (for aggregating global information)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
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
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Output activation for phase and transmittance
        # Phase: unbounded (will be wrapped to [-π, π] later)
        # Transmittance: sigmoid to ensure [0, 1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, control_points: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            control_points: (batch, n_control_points, 2) control point coordinates
            wavelength: (batch, 1) wavelength (nm), normalized to [0, 1]
        
        Returns:
            output: (batch, 2) [phase, transmittance]
        """
        batch_size = control_points.size(0)
        
        # Encode wavelength
        wavelength_features = self.wavelength_fourier(wavelength)  # (batch, wavelength_encoding_dim)
        wavelength_token = self.wavelength_projection(wavelength_features)  # (batch, d_model)
        wavelength_token = wavelength_token.unsqueeze(1)  # (batch, 1, d_model)
        
        # Encode control points
        point_features = self.input_projection(control_points)  # (batch, n_points, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # Concatenate: [CLS, wavelength, point1, point2, ..., point8]
        x = torch.cat([cls_tokens, wavelength_token, point_features], dim=1)  # (batch, 10, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, 10, d_model)
        
        # Use CLS token output
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Output projection
        output = self.output_head(cls_output)  # (batch, 2)
        
        # Apply sigmoid to transmittance
        phase = output[:, 0:1]  # (batch, 1)
        transmittance = torch.sigmoid(output[:, 1:2])  # (batch, 1), ensure [0, 1]
        
        return torch.cat([phase, transmittance], dim=1)


class WavelengthConditionedDataset(Dataset):
    """
    Wavelength-conditioned dataset
    
    Supports two data formats:
    1. Single wavelength data: each sample corresponds to one wavelength
    2. Multi-wavelength data: each control point corresponds to multiple wavelength responses
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
            control_points: (n_samples, 8, 2) control point coordinates
            wavelengths: (n_samples,) wavelength (nm)
            phases: (n_samples,) phase (rad)
            transmittances: (n_samples,) transmittance
        """
        # Normalize control points to [0, 1]
        self.control_points = torch.FloatTensor(
            (control_points - point_min) / (point_max - point_min)
        )
        
        # Normalize wavelength to [0, 1]
        self.wavelengths = torch.FloatTensor(
            (wavelengths - wavelength_min) / (wavelength_max - wavelength_min)
        ).unsqueeze(-1)  # (n_samples, 1)
        
        # Normalize phase to [0, 1] (from [-π, π])
        self.phases = torch.FloatTensor(
            (phases + np.pi) / (2 * np.pi)
        )
        
        # Transmittance is already in [0, 1]
        self.transmittances = torch.FloatTensor(transmittances)
        
        # Store normalization parameters
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
    Wavelength-conditioned NURBS surrogate model wrapper for training and inference
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
        
        # Print model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Combined loss: MSE + phase periodicity loss
        self.mse_loss = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        self.train_losses = []
        self.val_losses = []
        
        # Normalization parameters
        self.wavelength_min = 400.0
        self.wavelength_max = 700.0
        self.point_min = -0.25
        self.point_max = 0.25
    
    def phase_loss(self, pred_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
        """
        Phase loss considering periodicity
        Phase is in [-π, π] range, need to handle boundary cases
        """
        # Convert normalized phase back to radians
        pred_rad = pred_phase * 2 * np.pi - np.pi
        target_rad = target_phase * 2 * np.pi - np.pi
        
        # Calculate phase difference considering periodicity
        diff = pred_rad - target_rad
        # Wrap difference to [-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return torch.mean(diff ** 2)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for control_points, wavelengths, targets in train_loader:
            control_points = control_points.to(self.device)
            wavelengths = wavelengths.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(control_points, wavelengths)
            
            # Calculate phase and transmittance loss separately
            phase_l = self.phase_loss(outputs[:, 0], targets[:, 0])
            trans_l = self.mse_loss(outputs[:, 1], targets[:, 1])
            
            # Combined loss (weights can be adjusted)
            loss = phase_l + trans_l
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validation"""
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
                
                # Calculate actual error (after denormalization)
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
        """Train model"""
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
                      f'Phase Error: {np.degrees(phase_error):.2f} deg, '
                      f'Trans Error: {trans_error:.4f}')
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    def predict(self, 
                control_points: np.ndarray, 
                wavelength_nm: float) -> Tuple[float, float]:
        """
        Predict single sample
        
        Args:
            control_points: (8, 2) control point coordinates (um)
            wavelength_nm: wavelength (nm)
        
        Returns:
            phase: phase (rad)
            transmittance: transmittance
        """
        self.model.eval()
        
        with torch.no_grad():
            # Normalize
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            wl_norm = (wavelength_nm - self.wavelength_min) / (self.wavelength_max - self.wavelength_min)
            
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            wl_tensor = torch.FloatTensor([[wl_norm]]).to(self.device)
            
            output = self.model(cp_tensor, wl_tensor)
            
            # Denormalize
            phase = output[0, 0].item() * 2 * np.pi - np.pi
            transmittance = output[0, 1].item()
            
            return phase, transmittance
    
    def predict_spectrum(self,
                        control_points: np.ndarray,
                        wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict response at multiple wavelengths (spectrum)
        
        Args:
            control_points: (8, 2) control point coordinates (um)
            wavelengths: (n_wavelengths,) wavelength array (nm)
        
        Returns:
            phases: (n_wavelengths,) phase array (rad)
            transmittances: (n_wavelengths,) transmittance array
        """
        self.model.eval()
        
        with torch.no_grad():
            # Normalize control points
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            cp_tensor = cp_tensor.expand(len(wavelengths), -1, -1)  # (n_wl, 8, 2)
            
            # Normalize wavelength
            wl_norm = (wavelengths - self.wavelength_min) / (self.wavelength_max - self.wavelength_min)
            wl_tensor = torch.FloatTensor(wl_norm).unsqueeze(-1).to(self.device)  # (n_wl, 1)
            
            output = self.model(cp_tensor, wl_tensor)  # (n_wl, 2)
            
            # Denormalize
            phases = output[:, 0].cpu().numpy() * 2 * np.pi - np.pi
            transmittances = output[:, 1].cpu().numpy()
            
            return phases, transmittances
    
    def load_model(self, model_path: str):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")


def estimate_model_capacity():
    """
    Estimate model capacity and training data requirements
    """
    print("=" * 60)
    print("Model Capacity Analysis")
    print("=" * 60)
    
    # Create model
    model = WavelengthConditionedNURBSTransformer(
        d_model=256,
        nhead=8,
        num_layers=8,
        d_ff=1024
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\nRule of thumb analysis:")
    print(f"  - Typically need 10-100x parameters in training samples")
    print(f"  - Minimum recommended samples: {total_params * 10:,} ({total_params * 10 / 1e6:.1f}M)")
    print(f"  - Recommended samples: {total_params * 50:,} ({total_params * 50 / 1e6:.1f}M)")
    
    print(f"\nFor 500k samples:")
    samples = 500000
    ratio = samples / total_params
    print(f"  - Sample/parameter ratio: {ratio:.1f}")
    if ratio >= 50:
        print(f"  - OK: Sufficient samples, model can be fully trained")
    elif ratio >= 10:
        print(f"  - WARNING: Moderate samples, recommend regularization and data augmentation")
    else:
        print(f"  - ERROR: Insufficient samples, recommend smaller model or more data")
    
    print(f"\nDifferent model configurations comparison:")
    configs = [
        ("Small (d=128, L=4)", 128, 4, 512),
        ("Medium (d=256, L=6)", 256, 6, 1024),
        ("Large (d=256, L=8)", 256, 8, 1024),
        ("XLarge (d=512, L=8)", 512, 8, 2048),
    ]
    
    for name, d_model, num_layers, d_ff in configs:
        m = WavelengthConditionedNURBSTransformer(
            d_model=d_model, num_layers=num_layers, d_ff=d_ff
        )
        params = sum(p.numel() for p in m.parameters())
        ratio = samples / params
        status = "OK" if ratio >= 10 else "WARNING"
        print(f"  {name}: {params:,} params, ratio={ratio:.1f} {status}")
    
    return total_params


if __name__ == "__main__":
    # Run capacity analysis
    estimate_model_capacity()
    
    print("\n" + "=" * 60)
    print("Usage Example")
    print("=" * 60)
    
    # Create model
    model_wrapper = WavelengthConditionedNURBSModel(
        d_model=256,
        nhead=8,
        num_layers=8,
        d_ff=1024
    )
    
    # Example prediction
    control_points = np.array([
        (0.16, 0.02), (0.14, 0.14), (0.02, 0.16), (-0.14, 0.14),
        (-0.16, -0.02), (-0.14, -0.14), (-0.02, -0.16), (0.14, -0.14)
    ])
    
    print(f"\nExample control points:\n{control_points}")
    
    # Single wavelength prediction
    phase, trans = model_wrapper.predict(control_points, wavelength_nm=550)
    print(f"\n550nm wavelength prediction (untrained model, random output):")
    print(f"  Phase: {phase:.4f} rad ({np.degrees(phase):.2f} deg)")
    print(f"  Transmittance: {trans:.4f}")
    
    # Spectrum prediction
    wavelengths = np.linspace(400, 700, 31)
    phases, transmittances = model_wrapper.predict_spectrum(control_points, wavelengths)
    print(f"\nSpectrum prediction (400-700nm, 31 points):")
    print(f"  Phase range: {np.min(phases):.2f} ~ {np.max(phases):.2f} rad")
    print(f"  Transmittance range: {np.min(transmittances):.4f} ~ {np.max(transmittances):.4f}")
