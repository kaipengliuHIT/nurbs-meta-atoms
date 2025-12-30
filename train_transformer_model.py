import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
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
    Generate sample data for training
    In practical applications, you need to use real simulation data
    """
    print(f"Generating training data for {n_samples} samples...")
    
    # Generate random control points (small perturbations around base shape)
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    control_points_list = []
    phase_trans_list = []
    
    for i in range(n_samples):
        # Add random perturbation to base shape
        perturbation = np.random.uniform(-0.05, 0.05, (n_control_points, 2))
        current_control_points = base_control_points + perturbation
        
        # Ensure control points are within reasonable range
        current_control_points = np.clip(current_control_points, -0.25, 0.25)
        
        control_points_list.append(current_control_points)
        
        # Generate simulated phase and transmittance values
        # In practical applications, these should come from real physics simulation
        phase = np.random.uniform(-np.pi, np.pi)  # Phase range [-π, π]
        transmittance = np.random.uniform(0.0, 1.0)  # Transmittance range [0, 1]
        
        phase_trans_list.append([phase, transmittance])
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")
    
    control_points = np.array(control_points_list)
    targets = np.array(phase_trans_list)
    
    return control_points, targets


def generate_realistic_data(n_samples=500):
    """
    Generate more realistic data using Simulation class from nurbs_atoms_data.py
    Note: This will run actual simulations, which may be time-consuming
    """
    print(f"Generating {n_samples} realistic data samples using Simulation class...")
    
    control_points_list = []
    phase_trans_list = []
    
    for i in range(n_samples):
        # Generate random control points
        base_control_points = np.array([
            (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
            (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
        ])
        
        # Add random perturbation
        perturbation = np.random.uniform(-0.03, 0.03, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        try:
            # Create simulation object and run
            sim = Simulation(control_points=current_control_points)
            transmittance, phase = sim.run_forward(wavelength_start=500e-9, wavelength_stop=600e-9)
            
            control_points_list.append(current_control_points)
            phase_trans_list.append([phase, transmittance])
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{n_samples} realistic samples, Phase: {phase:.3f}, Transmittance: {transmittance:.3f}")
                
        except Exception as e:
            print(f"Simulation failed, sample {i}: {e}")
            # If simulation fails, use random values as fallback
            control_points_list.append(current_control_points)
            phase = np.random.uniform(-np.pi, np.pi)
            transmittance = np.random.uniform(0.0, 1.0)
            phase_trans_list.append([phase, transmittance])
    
    return np.array(control_points_list), np.array(phase_trans_list)


def train_model():
    """Train Transformer model"""
    print("Starting training of NURBS Transformer surrogate model...")
    
    # Generate training data
    # Note: In practical applications, you may need to use real simulation data
    # Here we use simulated data as an example
    print("Generating training data...")
    control_points, targets = generate_sample_data(n_samples=2000)
    
    # Data normalization
    print("Normalizing data...")
    normalized_control_points = normalize_control_points(control_points)
    normalized_targets = normalize_targets(targets)
    
    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        normalized_control_points, normalized_targets, 
        test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = NURBSDataset(X_train, y_train)
    val_dataset = NURBSDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Create model
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    print("Starting training...")
    model.train(train_loader, val_loader, epochs=100, save_path="nurbs_transformer_model.pth")
    
    # Plot training curves
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
    
    print("Model training complete!")


def test_model():
    """Test the trained model"""
    print("Testing the trained model...")
    
    # Create model instance
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    # Load trained model
    if os.path.exists("nurbs_transformer_model.pth"):
        model.load_model("nurbs_transformer_model.pth")
        
        # Test prediction
        base_control_points = np.array([
            (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
            (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
        ])
        
        # Add some perturbation
        test_control_points = base_control_points + np.random.uniform(-0.02, 0.02, (8, 2))
        test_control_points = np.clip(test_control_points, -0.2, 0.2)
        
        # Normalize control points
        normalized_test_points = normalize_control_points(test_control_points.reshape(1, 8, 2))
        normalized_test_points = torch.FloatTensor(normalized_test_points)
        
        # Predict
        prediction = model.predict(normalized_test_points)
        
        # Denormalize prediction results
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        print(f"Input control points: {test_control_points}")
        print(f"Predicted phase: {pred_phase:.4f}")
        print(f"Predicted transmittance: {pred_trans:.4f}")
        
        # Compare with real simulation (if available)
        try:
            sim = Simulation(control_points=test_control_points)
            true_trans, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            print(f"True phase: {true_phase:.4f}")
            print(f"True transmittance: {true_trans:.4f}")
            print(f"Phase error: {abs(pred_phase - true_phase):.4f}")
            print(f"Transmittance error: {abs(pred_trans - true_trans):.4f}")
        except:
            print("Cannot run real simulation for comparison")
    else:
        print("Trained model file not found, please run training first")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NURBS Transformer surrogate model')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Test model')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.test:
        test_model()
    else:
        print("Please specify --train or --test parameter")
        print("Example: python train_transformer_model.py --train")
        print("Example: python train_transformer_model.py --test")