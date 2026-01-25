"""
Training script for NURBS Transformer Surrogate Model

Paper specifications:
- 500,000 training samples (90% train, 10% validation)
- Adam optimizer (lr = 5×10⁻⁵, β₁ = 0.9, β₂ = 0.98)
- 10,000 epochs
- Target MAE: 0.0187
- 12 attention heads, 8 encoder/decoder layers
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meta_transformer import (
    NURBSSurrogateModel,
    NURBSMetaAtomDataset,
    create_data_loaders,
    print_paper_specs
)


def generate_synthetic_training_data(n_samples: int = 500000,
                                     n_control_points: int = 8,
                                     n_wavelengths_per_sample: int = 1):
    """
    Generate synthetic training data
    
    In production, this should be replaced with actual FDTD simulation data from MEEP
    
    Paper specs:
    - 500,000 unique NURBS meta-atom geometries
    - Curvature radii: 50-300nm
    - Aspect ratios: 0.2-5.0
    - 10nm mesh grid
    - Spectrum: 400-700nm
    """
    print(f"Generating {n_samples:,} synthetic training samples...")
    print("Note: Replace with actual MEEP FDTD simulation data for production")
    
    # Base control points (circular shape)
    base_radius = 0.18  # um
    angles = np.linspace(0, 2 * np.pi, n_control_points, endpoint=False)
    base_control_points = np.stack([
        base_radius * np.cos(angles),
        base_radius * np.sin(angles)
    ], axis=-1)
    
    control_points_list = []
    wavelengths_list = []
    phases_list = []
    transmittances_list = []
    
    # Paper specs: curvature radii 50-300nm, aspect ratios 0.2-5.0
    min_radius = 0.05  # 50nm in um
    max_radius = 0.30  # 300nm in um
    
    for i in range(n_samples):
        # Random perturbation for shape variation
        # Simulate different curvature radii and aspect ratios
        radius_scale = np.random.uniform(min_radius / base_radius, max_radius / base_radius)
        aspect_ratio = np.random.uniform(0.2, 5.0)
        
        # Apply aspect ratio (stretch in x or y)
        if aspect_ratio > 1:
            x_scale = 1.0
            y_scale = 1.0 / aspect_ratio
        else:
            x_scale = aspect_ratio
            y_scale = 1.0
        
        # Generate control points with variations
        perturbation = np.random.uniform(-0.03, 0.03, (n_control_points, 2))
        current_cp = base_control_points * radius_scale
        current_cp[:, 0] *= x_scale
        current_cp[:, 1] *= y_scale
        current_cp += perturbation
        
        # Clip to valid range
        current_cp = np.clip(current_cp, -0.28, 0.28)
        
        # Random wavelength (400-700nm)
        wavelength = np.random.uniform(400, 700)
        
        # Simulate physics-based response (simplified model)
        # In production, use actual MEEP FDTD results
        area = np.abs(np.sum(current_cp[:, 0] * np.roll(current_cp[:, 1], 1) - 
                            current_cp[:, 1] * np.roll(current_cp[:, 0], 1))) / 2
        
        # Phase depends on geometry and wavelength
        effective_path = area * 10  # Simplified
        phase = (2 * np.pi * effective_path / (wavelength / 1000)) % (2 * np.pi) - np.pi
        phase += np.random.normal(0, 0.1)  # Add noise
        phase = np.clip(phase, -np.pi, np.pi)
        
        # Transmittance depends on geometry
        transmittance = 0.3 + 0.6 * np.exp(-area * 5) + np.random.normal(0, 0.05)
        transmittance = np.clip(transmittance, 0.01, 0.99)
        
        control_points_list.append(current_cp)
        wavelengths_list.append(wavelength)
        phases_list.append(phase)
        transmittances_list.append(transmittance)
        
        if (i + 1) % 50000 == 0:
            print(f"  Generated {i + 1:,}/{n_samples:,} samples...")
    
    control_points = np.array(control_points_list)
    wavelengths = np.array(wavelengths_list)
    phases = np.array(phases_list)
    transmittances = np.array(transmittances_list)
    
    print(f"Data generation complete!")
    print(f"  Control points shape: {control_points.shape}")
    print(f"  Wavelengths shape: {wavelengths.shape}")
    print(f"  Phases range: [{phases.min():.3f}, {phases.max():.3f}]")
    print(f"  Transmittances range: [{transmittances.min():.3f}, {transmittances.max():.3f}]")
    
    return control_points, wavelengths, phases, transmittances


def load_meep_simulation_data(data_dir: str):
    """
    Load actual MEEP FDTD simulation data
    
    Expected files:
    - control_points.npy: (n_samples, 8, 2)
    - wavelengths.npy: (n_samples,)
    - phases.npy: (n_samples,)
    - transmittances.npy: (n_samples,)
    """
    print(f"Loading MEEP simulation data from {data_dir}...")
    
    control_points = np.load(os.path.join(data_dir, 'control_points.npy'))
    wavelengths = np.load(os.path.join(data_dir, 'wavelengths.npy'))
    phases = np.load(os.path.join(data_dir, 'phases.npy'))
    transmittances = np.load(os.path.join(data_dir, 'transmittances.npy'))
    
    print(f"Loaded {len(control_points):,} samples")
    
    return control_points, wavelengths, phases, transmittances


def train_model(args):
    """Main training function"""
    
    print_paper_specs()
    
    # Load or generate data
    if args.data_dir and os.path.exists(args.data_dir):
        control_points, wavelengths, phases, transmittances = load_meep_simulation_data(args.data_dir)
    else:
        print("\nNo data directory provided or not found. Generating synthetic data...")
        control_points, wavelengths, phases, transmittances = generate_synthetic_training_data(
            n_samples=args.n_samples
        )
    
    # Save generated data if requested
    if args.save_data:
        os.makedirs('training_data', exist_ok=True)
        np.save('training_data/control_points.npy', control_points)
        np.save('training_data/wavelengths.npy', wavelengths)
        np.save('training_data/phases.npy', phases)
        np.save('training_data/transmittances.npy', transmittances)
        print("Training data saved to training_data/")
    
    # Create data loaders with paper-specified 90/10 split
    train_loader, val_loader = create_data_loaders(
        control_points=control_points,
        wavelengths=wavelengths,
        phases=phases,
        transmittances=transmittances,
        batch_size=args.batch_size,
        train_split=0.9  # Paper: 90% training, 10% validation
    )
    
    # Create model with paper specifications
    print("\nCreating NURBS surrogate model...")
    model = NURBSSurrogateModel(
        n_control_points=8,
        d_model=args.d_model,
        nhead=12,                      # Paper: 12 attention heads
        num_encoder_layers=8,          # Paper: 8 encoder layers
        num_decoder_layers=8,          # Paper: 8 decoder layers
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_binary_encoding=args.use_binary_encoding
    )
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"nurbs_model_{timestamp}.pth"
    
    best_mae = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,            # Paper: 10,000 epochs
        save_path=save_path,
        target_mae=0.0187,             # Paper: MAE 0.0187
        early_stopping_patience=args.patience,
        log_interval=args.log_interval
    )
    
    # Plot training curves
    plot_training_curves(model, save_path=f"training_curves_{timestamp}.png")
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Testing Trained Model")
    print("=" * 70)
    
    test_model_predictions(model, control_points[:5], wavelengths[:5], phases[:5], transmittances[:5])
    
    return model, best_mae


def plot_training_curves(model, save_path: str = "training_curves.png"):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(len(model.train_losses))
    
    # Loss curves
    axes[0].plot(epochs, model.train_losses, label='Train Loss', alpha=0.8)
    axes[0].plot(epochs, model.val_losses, label='Val Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # MAE curve
    axes[1].plot(epochs, model.val_maes, label='Val MAE', color='green', alpha=0.8)
    axes[1].axhline(y=0.0187, color='red', linestyle='--', label='Target MAE (0.0187)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    axes[2].plot(epochs, model.train_losses, label='Train', alpha=0.5)
    axes[2].plot(epochs, model.val_losses, label='Val', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss Convergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def test_model_predictions(model, control_points, wavelengths, phases, transmittances):
    """Test model predictions on sample data"""
    print("\nSample Predictions:")
    print("-" * 70)
    
    for i in range(min(5, len(control_points))):
        pred_phase, pred_trans = model.predict(control_points[i], wavelengths[i])
        true_phase = phases[i]
        true_trans = transmittances[i]
        
        phase_error = np.abs(pred_phase - true_phase)
        trans_error = np.abs(pred_trans - true_trans)
        
        print(f"Sample {i+1}:")
        print(f"  Wavelength: {wavelengths[i]:.1f}nm")
        print(f"  Phase:  Pred={pred_phase:.4f}, True={true_phase:.4f}, Error={phase_error:.4f}")
        print(f"  Trans:  Pred={pred_trans:.4f}, True={true_trans:.4f}, Error={trans_error:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Paper-Matched NURBS Transformer Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing MEEP simulation data')
    parser.add_argument('--n_samples', type=int, default=500000,
                       help='Number of training samples (paper: 500,000)')
    parser.add_argument('--save_data', action='store_true',
                       help='Save generated training data')
    
    # Model arguments (paper-specified defaults)
    parser.add_argument('--d_model', type=int, default=384,
                       help='Model dimension (must be divisible by 12 heads)')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use_binary_encoding', action='store_true', default=True,
                       help='Use binary grid encoding for control points')
    
    # Training arguments (paper-specified defaults)
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Number of training epochs (paper: 10,000)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=500,
                       help='Early stopping patience')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (epochs)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with reduced samples and epochs')
    
    args = parser.parse_args()
    
    # Quick test mode overrides
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***")
        args.n_samples = 5000
        args.epochs = 100
        args.log_interval = 10
        args.patience = 50
    
    print("\nTraining Configuration:")
    print(f"  Samples: {args.n_samples:,}")
    print(f"  Epochs: {args.epochs:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model dim: {args.d_model}")
    print(f"  FFN dim: {args.d_ff}")
    print(f"  Binary encoding: {args.use_binary_encoding}")
    
    # Paper-specified fixed parameters
    print("\nPaper-Specified Parameters (Fixed):")
    print(f"  Attention heads: 12")
    print(f"  Encoder layers: 8")
    print(f"  Decoder layers: 8")
    print(f"  Optimizer: Adam (lr=5e-5, β₁=0.9, β₂=0.98)")
    print(f"  Train/Val split: 90/10")
    print(f"  Target MAE: 0.0187")
    
    model, best_mae = train_model(args)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best MAE achieved: {best_mae:.4f}")
    print(f"Target MAE: 0.0187")
    
    if best_mae <= 0.0187:
        print("✓ Target MAE reached!")
    else:
        print(f"Note: Target MAE not reached. May need more training or data.")


if __name__ == "__main__":
    main()
