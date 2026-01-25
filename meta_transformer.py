"""
NURBS Meta-Atom Transformer Model - Paper Matched Version

This implementation matches the paper description:
- Transformer-based surrogate model with Encoder-Decoder architecture (Fig. 1c)
- 12 attention heads
- 8 encoder/decoder layers
- Adam optimizer (lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98)
- Dual-output: complex optical response (amplitude/phase) + parametric gradients
- 500,000 samples (90% training, 10% validation)
- Training convergence at 10,000 epochs with MAE 0.0187

Computational Framework:
- FDTD simulations using MEEP
- 500,000 unique NURBS meta-atom geometries
- Curvature radii: 50-300nm, aspect ratios: 0.2-5.0
- 10nm mesh grid, 400-700nm spectrum
- <0.5% residual error
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple, Optional, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence inputs"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BinaryGridEncoder(nn.Module):
    """
    Binary encoded grid indices for NURBS control points
    Maps control points to discretized Cartesian grid with binary encoding
    """
    def __init__(self, grid_size: int = 64, encoding_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.num_bits = int(math.ceil(math.log2(grid_size)))
        
        # Binary encoding projection
        self.binary_projection = nn.Linear(self.num_bits * 2, encoding_dim)
        self.layer_norm = nn.LayerNorm(encoding_dim)
    
    def _to_binary(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert grid indices to binary encoding"""
        batch_size, n_points = indices.shape
        binary = torch.zeros(batch_size, n_points, self.num_bits, device=indices.device)
        
        for bit in range(self.num_bits):
            binary[:, :, bit] = (indices >> bit) & 1
        
        return binary.float()
    
    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_points: (batch, n_points, 2) normalized to [0, 1]
        Returns:
            encoded: (batch, n_points, encoding_dim)
        """
        # Discretize to grid indices
        grid_indices = (control_points * (self.grid_size - 1)).long().clamp(0, self.grid_size - 1)
        
        # Binary encode x and y separately
        x_binary = self._to_binary(grid_indices[:, :, 0])  # (batch, n_points, num_bits)
        y_binary = self._to_binary(grid_indices[:, :, 1])  # (batch, n_points, num_bits)
        
        # Concatenate x and y binary encodings
        binary_encoding = torch.cat([x_binary, y_binary], dim=-1)  # (batch, n_points, num_bits*2)
        
        # Project to encoding dimension
        encoded = self.binary_projection(binary_encoding)
        encoded = self.layer_norm(encoded)
        
        return encoded


class NURBSEncoderDecoderTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for NURBS meta-atom surrogate modeling
    
    Architecture matches Fig. 1c:
    - Encoder: processes input NURBS parameter encoding
    - Decoder: with masked multi-head attention for output generation
    - Dual output: optical response (amplitude/phase) + parametric gradients
    
    Paper specifications:
    - 12 attention heads
    - 8 encoder/decoder layers
    """
    def __init__(self,
                 input_dim: int = 2,
                 n_control_points: int = 8,
                 d_model: int = 384,         # Must be divisible by nhead (384/12=32)
                 nhead: int = 12,           # Paper: 12 attention heads
                 num_encoder_layers: int = 8,  # Paper: 8 encoder layers
                 num_decoder_layers: int = 8,  # Matching decoder layers
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 n_wavelengths: int = 31,   # 400-700nm spectrum output
                 output_dim: int = 2,       # phase and amplitude/transmittance
                 use_binary_encoding: bool = True,
                 grid_size: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_control_points = n_control_points
        self.d_model = d_model
        self.n_wavelengths = n_wavelengths
        self.output_dim = output_dim
        
        # Input NURBS parameter encoding (Fig. 1c bottom-left)
        if use_binary_encoding:
            self.input_encoder = BinaryGridEncoder(grid_size=grid_size, encoding_dim=d_model)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        self.use_binary_encoding = use_binary_encoding
        
        # Additional linear layers as shown in Fig. 1c
        self.encoder_input_linear1 = nn.Linear(d_model, d_model)
        self.encoder_input_linear2 = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_control_points + 10, dropout=dropout)
        
        # Wavelength conditioning (for spectrum output)
        self.wavelength_embedding = nn.Embedding(n_wavelengths, d_model)
        
        # Learnable query tokens for decoder
        self.query_tokens = nn.Parameter(torch.randn(1, n_wavelengths, d_model))
        
        # Transformer Encoder (Fig. 1c middle-left)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Transformer Decoder (Fig. 1c right side with Masked Multi-Head Attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output NURBS parameter encoding (decoder input, Fig. 1c bottom-right)
        self.decoder_input_linear1 = nn.Linear(d_model, d_model)
        self.decoder_input_linear2 = nn.Linear(d_model, d_model)
        
        # Dual Output Heads
        # Head 1: Optical response (amplitude/phase) - Fig. 1c top
        self.optical_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)  # [phase, amplitude/transmittance]
        )
        
        # Head 2: Parametric gradients for end-to-end optimization
        self.gradient_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_control_points * input_dim)  # gradients w.r.t. control points
        )
        
        # Final output projection with Softmax (as shown in Fig. 1c)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, 
                control_points: torch.Tensor,
                wavelength_indices: Optional[torch.Tensor] = None,
                return_gradients: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            control_points: (batch, n_control_points, 2) normalized control points
            wavelength_indices: (batch, n_query) wavelength indices to query (optional)
            return_gradients: whether to return parametric gradients
        
        Returns:
            Dict containing:
                - 'optical': (batch, n_wavelengths, 2) phase and transmittance
                - 'gradients': (batch, n_wavelengths, n_control_points * 2) if return_gradients
        """
        batch_size = control_points.size(0)
        device = control_points.device
        
        # === ENCODER PATH (Fig. 1c left side) ===
        
        # Input NURBS parameter encoding -> Linear -> Linear
        if self.use_binary_encoding:
            encoder_input = self.input_encoder(control_points)
        else:
            encoder_input = self.input_encoder(control_points)
        
        encoder_input = self.encoder_input_linear1(encoder_input)
        encoder_input = torch.relu(encoder_input)
        encoder_input = self.encoder_input_linear2(encoder_input)
        
        # Add positional encoding
        encoder_input = self.pos_encoder(encoder_input)
        
        # Transformer encoder: Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
        encoder_output = self.transformer_encoder(encoder_input)
        
        # === DECODER PATH (Fig. 1c right side) ===
        
        # Prepare decoder input (query tokens or wavelength embeddings)
        if wavelength_indices is not None:
            # Use specific wavelength queries
            decoder_input = self.wavelength_embedding(wavelength_indices)
            n_queries = wavelength_indices.size(1)
        else:
            # Use learnable query tokens for all wavelengths
            decoder_input = self.query_tokens.expand(batch_size, -1, -1)
            n_queries = self.n_wavelengths
        
        # Output NURBS parameter encoding -> Linear -> Linear (Fig. 1c bottom-right)
        decoder_input = self.decoder_input_linear1(decoder_input)
        decoder_input = torch.relu(decoder_input)
        decoder_input = self.decoder_input_linear2(decoder_input)
        
        # Add positional encoding
        decoder_input = self.pos_encoder(decoder_input)
        
        # Generate causal mask for decoder (Masked Multi-Head Attention)
        tgt_mask = self.generate_square_subsequent_mask(n_queries, device)
        
        # Transformer decoder:
        # Masked Multi-Head Attention -> Add & Norm -> 
        # Multi-Head Attention (cross) -> Add & Norm -> 
        # Feed Forward -> Add & Norm
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask
        )
        
        # === OUTPUT HEADS ===
        
        # Linear -> Softmax (Fig. 1c top)
        output_features = self.output_linear(decoder_output)
        
        # Dual output architecture
        # Head 1: Optical response (phase, transmittance)
        optical_output = self.optical_head(output_features)
        
        # Apply appropriate activations
        phase = optical_output[:, :, 0:1]  # Unbounded, will be wrapped
        transmittance = torch.sigmoid(optical_output[:, :, 1:2])  # [0, 1]
        optical_output = torch.cat([phase, transmittance], dim=-1)
        
        result = {'optical': optical_output}
        
        # Head 2: Parametric gradients (for end-to-end optimization)
        if return_gradients:
            gradient_output = self.gradient_head(output_features)
            gradient_output = gradient_output.view(batch_size, n_queries, 
                                                   self.n_control_points, self.input_dim)
            result['gradients'] = gradient_output
        
        return result


class NURBSMetaAtomDataset(Dataset):
    """
    Dataset for NURBS meta-atom training
    
    Paper specs:
    - 500,000 samples
    - Curvature radii: 50-300nm
    - Aspect ratios: 0.2-5.0
    - Spectrum: 400-700nm
    """
    def __init__(self,
                 control_points: np.ndarray,
                 wavelengths: np.ndarray,
                 phases: np.ndarray,
                 transmittances: np.ndarray,
                 wavelength_min: float = 400.0,
                 wavelength_max: float = 700.0,
                 point_min: float = -0.3,
                 point_max: float = 0.3):
        """
        Args:
            control_points: (n_samples, 8, 2) control point coordinates
            wavelengths: (n_samples,) or (n_samples, n_wavelengths) wavelength(s)
            phases: (n_samples,) or (n_samples, n_wavelengths) phase(s)
            transmittances: (n_samples,) or (n_samples, n_wavelengths) transmittance(s)
        """
        # Normalize control points to [0, 1]
        self.control_points = torch.FloatTensor(
            (control_points - point_min) / (point_max - point_min)
        )
        
        # Handle single or multi-wavelength data
        if wavelengths.ndim == 1:
            wavelengths = wavelengths[:, np.newaxis]
            phases = phases[:, np.newaxis]
            transmittances = transmittances[:, np.newaxis]
        
        # Normalize wavelength to indices (0 to n_wavelengths-1)
        n_wavelengths = 31  # 400-700nm with 10nm steps
        self.wavelength_indices = torch.LongTensor(
            np.round((wavelengths - wavelength_min) / (wavelength_max - wavelength_min) * (n_wavelengths - 1))
        ).clamp(0, n_wavelengths - 1)
        
        # Normalize phase to [0, 1] from [-π, π]
        self.phases = torch.FloatTensor((phases + np.pi) / (2 * np.pi))
        
        # Transmittance already in [0, 1]
        self.transmittances = torch.FloatTensor(transmittances)
        
        self.point_min = point_min
        self.point_max = point_max
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
    
    def __len__(self):
        return len(self.control_points)
    
    def __getitem__(self, idx):
        targets = torch.stack([self.phases[idx], self.transmittances[idx]], dim=-1)
        return (
            self.control_points[idx],
            self.wavelength_indices[idx],
            targets
        )


class NURBSSurrogateModel:
    """
    Complete model wrapper matching paper specifications:
    
    - 12 attention heads
    - 8 encoder/decoder layers
    - Adam optimizer: lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98
    - 500,000 samples (90% training, 10% validation)
    - Training: 10,000 epochs
    - Target MAE: 0.0187
    """
    def __init__(self,
                 n_control_points: int = 8,
                 d_model: int = 256,
                 nhead: int = 12,              # Paper: 12 attention heads
                 num_encoder_layers: int = 8,  # Paper: 8 layers
                 num_decoder_layers: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 n_wavelengths: int = 31,
                 use_binary_encoding: bool = True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = NURBSEncoderDecoderTransformer(
            n_control_points=n_control_points,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_wavelengths=n_wavelengths,
            use_binary_encoding=use_binary_encoding
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Architecture: Encoder-Decoder Transformer (Fig. 1c)")
        print(f"  - Attention heads: {nhead}")
        print(f"  - Encoder layers: {num_encoder_layers}")
        print(f"  - Decoder layers: {num_decoder_layers}")
        print(f"  - Model dimension: {d_model}")
        print(f"  - FFN dimension: {d_ff}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Paper-specified optimizer: Adam (lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=5e-5,                    # Paper: 5×10⁻⁵
            betas=(0.9, 0.98)           # Paper: β₁ = 0.9, β₂ = 0.98
        )
        print(f"Optimizer: Adam (lr=5e-5, β₁=0.9, β₂=0.98)")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Learning rate scheduler (warmup + decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        # Normalization parameters
        self.wavelength_min = 400.0
        self.wavelength_max = 700.0
        self.point_min = -0.3
        self.point_max = 0.3
    
    def phase_aware_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Phase-aware loss considering periodicity
        """
        # Phase (normalized to [0, 1])
        pred_phase = pred[:, :, 0]
        target_phase = target[:, :, 0]
        
        # Convert to radians
        pred_rad = pred_phase * 2 * np.pi - np.pi
        target_rad = target_phase * 2 * np.pi - np.pi
        
        # Periodic loss
        phase_diff = torch.atan2(torch.sin(pred_rad - target_rad),
                                  torch.cos(pred_rad - target_rad))
        phase_loss = torch.mean(phase_diff ** 2)
        
        # Transmittance MSE
        trans_loss = self.mse_loss(pred[:, :, 1], target[:, :, 1])
        
        return phase_loss + trans_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for control_points, wavelength_indices, targets in train_loader:
            control_points = control_points.to(self.device)
            wavelength_indices = wavelength_indices.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(control_points, wavelength_indices)
            optical_output = outputs['optical']
            
            loss = self.phase_aware_loss(optical_output, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation with MAE computation"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for control_points, wavelength_indices, targets in val_loader:
                control_points = control_points.to(self.device)
                wavelength_indices = wavelength_indices.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(control_points, wavelength_indices)
                optical_output = outputs['optical']
                
                loss = self.phase_aware_loss(optical_output, targets)
                total_loss += loss.item()
                
                # Compute MAE
                mae = self.mae_loss(optical_output, targets)
                total_mae += mae.item() * control_points.size(0)
                n_samples += control_points.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / n_samples
        
        return avg_loss, avg_mae
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10000,          # Paper: 10,000 epochs
              save_path: str = "nurbs_surrogate_model.pth",
              target_mae: float = 0.0187,   # Paper: MAE 0.0187
              early_stopping_patience: int = 500,
              log_interval: int = 100):
        """
        Train model to paper specifications
        
        Paper specs:
        - 10,000 epochs
        - Validation loss plateau at MAE 0.0187
        """
        print(f"\nStarting training:")
        print(f"  - Target epochs: {epochs}")
        print(f"  - Target MAE: {target_mae}")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        
        best_val_loss = float('inf')
        best_mae = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_mae = val_mae
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_maes': self.val_maes,
                }, save_path)
            else:
                patience_counter += 1
            
            # Logging
            if epoch % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, '
                      f'Val MAE: {val_mae:.4f}, '
                      f'LR: {current_lr:.2e}')
            
            # Check if target MAE reached
            if val_mae <= target_mae:
                print(f"\n✓ Target MAE {target_mae} reached at epoch {epoch}!")
                print(f"  Final MAE: {val_mae:.4f}")
                break
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"  Best MAE: {best_mae:.4f}")
                break
        
        print(f"\nTraining complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Best MAE: {best_mae:.4f}")
        
        return best_mae
    
    def predict(self,
                control_points: np.ndarray,
                wavelength_nm: float) -> Tuple[float, float]:
        """
        Single wavelength prediction
        
        Args:
            control_points: (8, 2) control points in um
            wavelength_nm: wavelength in nm (400-700)
        
        Returns:
            phase: phase in radians
            transmittance: transmittance [0, 1]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Normalize control points
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            
            # Convert wavelength to index
            n_wavelengths = 31
            wl_idx = int(round((wavelength_nm - self.wavelength_min) / 
                              (self.wavelength_max - self.wavelength_min) * (n_wavelengths - 1)))
            wl_idx = max(0, min(n_wavelengths - 1, wl_idx))
            wl_tensor = torch.LongTensor([[wl_idx]]).to(self.device)
            
            outputs = self.model(cp_tensor, wl_tensor)
            optical = outputs['optical']
            
            # Denormalize phase
            phase_norm = optical[0, 0, 0].item()
            phase = phase_norm * 2 * np.pi - np.pi
            transmittance = optical[0, 0, 1].item()
            
            return phase, transmittance
    
    def predict_spectrum(self,
                        control_points: np.ndarray,
                        wavelengths: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict full spectrum response
        
        Args:
            control_points: (8, 2) control points in um
            wavelengths: wavelength array in nm (default: 400-700nm, 10nm steps)
        
        Returns:
            phases: phase array in radians
            transmittances: transmittance array
        """
        if wavelengths is None:
            wavelengths = np.linspace(400, 700, 31)
        
        self.model.eval()
        
        with torch.no_grad():
            # Normalize control points
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            
            # Use all wavelength indices
            outputs = self.model(cp_tensor, wavelength_indices=None)
            optical = outputs['optical']
            
            # Denormalize
            phases = optical[0, :, 0].cpu().numpy() * 2 * np.pi - np.pi
            transmittances = optical[0, :, 1].cpu().numpy()
            
            return phases, transmittances
    
    def predict_with_gradients(self,
                               control_points: np.ndarray,
                               wavelength_nm: float) -> Dict[str, np.ndarray]:
        """
        Predict with parametric gradients for optimization
        
        Returns:
            Dict with 'phase', 'transmittance', 'gradients'
        """
        self.model.eval()
        
        with torch.no_grad():
            cp_norm = (control_points - self.point_min) / (self.point_max - self.point_min)
            cp_tensor = torch.FloatTensor(cp_norm).unsqueeze(0).to(self.device)
            
            n_wavelengths = 31
            wl_idx = int(round((wavelength_nm - self.wavelength_min) / 
                              (self.wavelength_max - self.wavelength_min) * (n_wavelengths - 1)))
            wl_idx = max(0, min(n_wavelengths - 1, wl_idx))
            wl_tensor = torch.LongTensor([[wl_idx]]).to(self.device)
            
            outputs = self.model(cp_tensor, wl_tensor, return_gradients=True)
            
            phase = outputs['optical'][0, 0, 0].item() * 2 * np.pi - np.pi
            transmittance = outputs['optical'][0, 0, 1].item()
            gradients = outputs['gradients'][0, 0].cpu().numpy()
            
            return {
                'phase': phase,
                'transmittance': transmittance,
                'gradients': gradients  # (n_control_points, 2)
            }
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        print(f"  Val MAE: {checkpoint.get('val_mae', 'N/A'):.4f}")


def create_data_loaders(control_points: np.ndarray,
                        wavelengths: np.ndarray,
                        phases: np.ndarray,
                        transmittances: np.ndarray,
                        batch_size: int = 256,
                        train_split: float = 0.9):  # Paper: 90% training
    """
    Create train/val data loaders with paper-specified split
    
    Paper specs: 90% training, 10% validation
    """
    n_samples = len(control_points)
    n_train = int(n_samples * train_split)
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"Data split (Paper spec: 90/10):")
    print(f"  Training samples: {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"  Validation samples: {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)")
    
    # Create datasets
    train_dataset = NURBSMetaAtomDataset(
        control_points[train_indices],
        wavelengths[train_indices],
        phases[train_indices],
        transmittances[train_indices]
    )
    
    val_dataset = NURBSMetaAtomDataset(
        control_points[val_indices],
        wavelengths[val_indices],
        phases[val_indices],
        transmittances[val_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def print_paper_specs():
    """Print paper specifications for reference"""
    print("=" * 70)
    print("Paper Specifications")
    print("=" * 70)
    print("""
Computational Framework:
  - FDTD simulations using MEEP
  - 500,000 unique NURBS meta-atom geometries
  - Curvature radii: 50-300nm
  - Aspect ratios: 0.2-5.0
  - 10nm mesh grid
  - Spectrum: 400-700nm
  - Residual error: <0.5%

Model Architecture (Fig. 1c):
  - Encoder-Decoder Transformer
  - 12 attention heads
  - 8 encoder/decoder layers
  - Binary encoded grid indices for control points

Training:
  - Adam optimizer (lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98)
  - 500,000 samples (90% training, 10% validation)
  - 10,000 epochs
  - Convergence at MAE 0.0187

Output:
  - Dual-output architecture
  - Complex optical response (amplitude/phase)
  - Parametric gradients for end-to-end optimization
""")
    print("=" * 70)


if __name__ == "__main__":
    print_paper_specs()
    
    print("\nCreating NURBS surrogate model...")
    model = NURBSSurrogateModel(
        n_control_points=8,
        d_model=256,
        nhead=12,              # Paper: 12 attention heads
        num_encoder_layers=8,  # Paper: 8 layers
        num_decoder_layers=8,
        d_ff=1024,
        dropout=0.1,
        use_binary_encoding=True
    )
    
    # Example usage
    print("\nExample prediction (untrained model):")
    control_points = np.array([
        (0.16, 0.02), (0.14, 0.14), (0.02, 0.16), (-0.14, 0.14),
        (-0.16, -0.02), (-0.14, -0.14), (-0.02, -0.16), (0.14, -0.14)
    ])
    
    phase, trans = model.predict(control_points, wavelength_nm=550)
    print(f"  Wavelength: 550nm")
    print(f"  Phase: {phase:.4f} rad ({np.degrees(phase):.2f}°)")
    print(f"  Transmittance: {trans:.4f}")
    
    # Prediction with gradients
    print("\nPrediction with gradients:")
    result = model.predict_with_gradients(control_points, wavelength_nm=550)
    print(f"  Gradients shape: {result['gradients'].shape}")
    
    # Spectrum prediction
    print("\nSpectrum prediction (400-700nm):")
    phases, transmittances = model.predict_spectrum(control_points)
    print(f"  Phase range: {np.min(phases):.2f} ~ {np.max(phases):.2f} rad")
    print(f"  Transmittance range: {np.min(transmittances):.4f} ~ {np.max(transmittances):.4f}")
