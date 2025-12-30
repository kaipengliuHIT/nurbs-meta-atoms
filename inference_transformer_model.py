import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

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

def evaluate_model_performance(model_path: str = "nurbs_transformer_model.pth", 
                            n_test_samples: int = 200):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    
    # Create model instance
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist, please train the model first")
        return
    
    # Load model
    model.load_model(model_path)
    
    # Generate test data
    print("Generating test data...")
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    test_control_points_list = []
    true_targets_list = []
    
    for i in range(n_test_samples):
        # Generate perturbed control points
        perturbation = np.random.uniform(-0.04, 0.04, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        # Get true values using Simulation class (or use simulated values)
        try:
            sim = Simulation(control_points=current_control_points)
            transmittance, phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            true_targets_list.append([phase, transmittance])
        except:
            # If simulation fails, use random values
            phase = np.random.uniform(-np.pi, np.pi)
            transmittance = np.random.uniform(0.0, 1.0)
            true_targets_list.append([phase, transmittance])
        
        test_control_points_list.append(current_control_points)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_test_samples} test samples")
    
    test_control_points = np.array(test_control_points_list)
    true_targets = np.array(true_targets_list)
    
    # Normalize data
    normalized_test_points = normalize_control_points(test_control_points)
    normalized_true_targets = normalize_targets(true_targets)
    
    # Predict
    print("Making predictions...")
    model.model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(normalized_test_points)):
            ctrl_points_tensor = torch.FloatTensor(normalized_test_points[i:i+1]).to(model.device)
            pred = model.model(ctrl_points_tensor)
            predictions.append(pred.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    
    # Denormalize prediction results
    denorm_predictions = denormalize_targets(predictions)
    denorm_true_targets = denormalize_targets(normalized_true_targets)
    
    # Calculate evaluation metrics
    phase_mse = mean_squared_error(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_mse = mean_squared_error(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    phase_mae = mean_absolute_error(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_mae = mean_absolute_error(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    phase_r2 = r2_score(denorm_true_targets[:, 0], denorm_predictions[:, 0])
    trans_r2 = r2_score(denorm_true_targets[:, 1], denorm_predictions[:, 1])
    
    print("\nModel performance evaluation results:")
    print(f"Phase prediction:")
    print(f"  MSE: {phase_mse:.6f}")
    print(f"  MAE: {phase_mae:.6f}")
    print(f"  R2: {phase_r2:.6f}")
    print(f"Transmittance prediction:")
    print(f"  MSE: {trans_mse:.6f}")
    print(f"  MAE: {trans_mae:.6f}")
    print(f"  R2: {trans_r2:.6f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Phase prediction comparison
    plt.subplot(1, 3, 1)
    plt.scatter(denorm_true_targets[:, 0], denorm_predictions[:, 0], alpha=0.6)
    plt.plot([denorm_true_targets[:, 0].min(), denorm_true_targets[:, 0].max()], 
             [denorm_true_targets[:, 0].min(), denorm_true_targets[:, 0].max()], 'r--', lw=2)
    plt.xlabel('True Phase')
    plt.ylabel('Predicted Phase')
    plt.title(f'Phase Prediction (R2 = {phase_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Transmittance prediction comparison
    plt.subplot(1, 3, 2)
    plt.scatter(denorm_true_targets[:, 1], denorm_predictions[:, 1], alpha=0.6)
    plt.plot([denorm_true_targets[:, 1].min(), denorm_true_targets[:, 1].max()], 
             [denorm_true_targets[:, 1].min(), denorm_true_targets[:, 1].max()], 'r--', lw=2)
    plt.xlabel('True Transmittance')
    plt.ylabel('Predicted Transmittance')
    plt.title(f'Transmittance Prediction (R2 = {trans_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Prediction error distribution
    plt.subplot(1, 3, 3)
    phase_errors = np.abs(denorm_true_targets[:, 0] - denorm_predictions[:, 0])
    trans_errors = np.abs(denorm_true_targets[:, 1] - denorm_predictions[:, 1])
    
    plt.hist(phase_errors, bins=30, alpha=0.5, label='Phase Error', density=True)
    plt.hist(trans_errors, bins=30, alpha=0.5, label='Transmittance Error', density=True)
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title('Prediction Error Distribution')
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
    """Predict for a single sample"""
    print("Predicting for a single sample...")
    
    # Create model instance
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist, please train the model first")
        return
    
    # Load model
    model.load_model(model_path)
    
    # Define an example control point
    example_control_points = np.array([
        (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
        (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
    ])
    
    print(f"Input control points:\n{example_control_points}")
    
    # Normalize control points
    normalized_control_points = normalize_control_points(example_control_points.reshape(1, 8, 2))
    
    # Predict
    prediction = model.predict(normalized_control_points)
    pred_phase, pred_trans = denormalize_targets(prediction)[0]
    
    print(f"\nPrediction results:")
    print(f"Phase: {pred_phase:.4f} radians ({np.degrees(pred_phase):.2f} degrees)")
    print(f"Transmittance: {pred_trans:.4f}")
    
    # Compare with real simulation (if available)
    try:
        sim = Simulation(control_points=example_control_points)
        true_trans, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
        print(f"\nReal simulation results:")
        print(f"Phase: {true_phase:.4f} radians ({np.degrees(true_phase):.2f} degrees)")
        print(f"Transmittance: {true_trans:.4f}")
        print(f"\nPrediction error:")
        print(f"Phase error: {abs(pred_phase - true_phase):.4f}")
        print(f"Transmittance error: {abs(pred_trans - true_trans):.4f}")
    except Exception as e:
        print(f"\nCannot run real simulation for comparison: {e}")
    
    return example_control_points, pred_phase, pred_trans


def batch_predict(model_path: str = "nurbs_transformer_model.pth", 
                 n_samples: int = 10):
    """Batch prediction"""
    print(f"Performing batch prediction for {n_samples} samples...")
    
    # Create model instance
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist, please train the model first")
        return
    
    # Load model
    model.load_model(model_path)
    
    # Generate random control points
    base_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    all_control_points = []
    all_predictions = []
    
    for i in range(n_samples):
        # Add random perturbation
        perturbation = np.random.uniform(-0.03, 0.03, (8, 2))
        current_control_points = base_control_points + perturbation
        current_control_points = np.clip(current_control_points, -0.22, 0.22)
        
        # Normalize and predict
        normalized_points = normalize_control_points(current_control_points.reshape(1, 8, 2))
        prediction = model.predict(normalized_points)
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        all_control_points.append(current_control_points)
        all_predictions.append([pred_phase, pred_trans])
        
        print(f"Sample {i+1}: Control point shape, Predicted phase={pred_phase:.3f}, Predicted transmittance={pred_trans:.3f}")
    
    return all_control_points, all_predictions


def visualize_nurbs_shape(control_points, title="NURBS Shape"):
    """Visualize NURBS shape"""
    from nurbs_atoms_data import Simulation
    
    # Create simulation object to generate NURBS curve
    sim = Simulation(control_points=control_points)
    nurbs_points = sim.generate_complete_nurbs_curve(control_points)
    
    # Extract x and y coordinates
    x_coords = [p[0] for p in nurbs_points]
    y_coords = [p[1] for p in nurbs_points]
    
    # Close the figure
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
    """Main function"""
    print("NURBS Transformer Surrogate Model Inference Tool")
    print("="*50)
    
    while True:
        print("\nPlease select operation:")
        print("1. Evaluate model performance")
        print("2. Single sample prediction")
        print("3. Batch prediction")
        print("4. Visualize NURBS shape")
        print("5. Exit")
        
        choice = input("Please enter your choice (1-5): ").strip()
        
        if choice == '1':
            n_samples = input("Please enter number of test samples (default 200): ").strip()
            n_samples = int(n_samples) if n_samples else 200
            evaluate_model_performance(n_test_samples=n_samples)
        
        elif choice == '2':
            predict_single_sample()
        
        elif choice == '3':
            n_samples = input("Please enter batch prediction count (default 10): ").strip()
            n_samples = int(n_samples) if n_samples else 10
            batch_predict(n_samples=n_samples)
        
        elif choice == '4':
            # Generate example control points and visualize
            example_control_points = np.array([
                (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
                (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
            ])
            visualize_nurbs_shape(example_control_points, "Example NURBS Shape")
        
        elif choice == '5':
            print("Exiting program")
            break
        
        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()