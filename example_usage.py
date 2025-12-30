import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# Add current directory to path
sys.path.append('/mnt/e/pythoncode/nurbs-meta-atoms')

from transformer_nurbs_model import (
    NURBSTransformerModel, 
    NURBSDataset, 
    normalize_control_points, 
    denormalize_targets
)
from nurbs_atoms_data import Simulation

def example_complete_workflow():
    """Complete workflow example"""
    print("NURBS Transformer Surrogate Model - Complete Workflow Example")
    print("="*60)
    
    # 1. Define example control points
    example_control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    print("1. Original control points:")
    print(example_control_points)
    
    # 2. Get reference values using real simulation
    print("\n2. Running real physics simulation to get reference values...")
    try:
        sim = Simulation(control_points=example_control_points)
        true_transmittance, true_phase = sim.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
        print(f"   True phase: {true_phase:.4f}")
        print(f"   True transmittance: {true_transmittance:.4f}")
    except Exception as e:
        print(f"   Simulation failed: {e}")
        true_phase = 1.57  # Default value
        true_transmittance = 0.8  # Default value
    
    # 3. Create and train model (if model does not exist)
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
        print(f"\n3. Loading pre-trained model: {model_path}")
        model.load_model(model_path)
    else:
        print(f"\n3. Model file does not exist, skipping load (need to train model first)")
    
    # 4. Use Transformer model for prediction
    print("\n4. Using Transformer surrogate model for prediction...")
    
    # Normalize control points
    normalized_control_points = normalize_control_points(example_control_points.reshape(1, 8, 2))
    
    if os.path.exists(model_path):
        prediction = model.predict(normalized_control_points)
        from transformer_nurbs_model import denormalize_targets
        pred_phase, pred_trans = denormalize_targets(prediction)[0]
        
        print(f"   Predicted phase: {pred_phase:.4f}")
        print(f"   Predicted transmittance: {pred_trans:.4f}")
        
        print(f"\n5. Prediction accuracy evaluation:")
        phase_error = abs(pred_phase - true_phase)
        trans_error = abs(pred_trans - true_transmittance)
        print(f"   Phase error: {phase_error:.4f}")
        print(f"   Transmittance error: {trans_error:.4f}")
        print(f"   Phase relative error: {abs(phase_error/true_phase)*100:.2f}%")
        print(f"   Transmittance relative error: {abs(trans_error/true_transmittance)*100:.2f}%")
    else:
        print("   Model not trained, skipping prediction")
    
    # 5. Show the influence of different control point shapes
    print("\n6. Testing the influence of different control point shapes...")
    
    shapes = {
        "circle": np.array([(0.15, 0), (0.106, 0.106), (0, 0.15), (-0.106, 0.106), 
                         (-0.15, 0), (-0.106, -0.106), (0, -0.15), (0.106, -0.106)]),
        "square": np.array([(0.15, 0.05), (0.15, 0.15), (0.05, 0.15), (-0.05, 0.15), 
                         (-0.15, 0.15), (-0.15, -0.15), (-0.05, -0.15), (0.05, -0.15)]),
        "ellipse": np.array([(0.20, 0), (0.141, 0.10), (0, 0.12), (-0.141, 0.10), 
                           (-0.20, 0), (-0.141, -0.10), (0, -0.12), (0.141, -0.10)])
    }
    
    results = {}
    for shape_name, shape_control_points in shapes.items():
        print(f"\n   {shape_name} shape:")
        
        # Use real simulation
        try:
            sim_shape = Simulation(control_points=shape_control_points)
            shape_trans, shape_phase = sim_shape.run_forward(wavelength_start=550e-9, wavelength_stop=550e-9)
            print(f"     True - Phase: {shape_phase:.4f}, Transmittance: {shape_trans:.4f}")
        except:
            shape_phase, shape_trans = 0, 0
            print(f"     True - Simulation failed")
        
        # Use Transformer prediction
        if os.path.exists(model_path):
            norm_shape_points = normalize_control_points(shape_control_points.reshape(1, 8, 2))
            shape_pred = model.predict(norm_shape_points)
            shape_pred_phase, shape_pred_trans = denormalize_targets(shape_pred)[0]
            print(f"     Predicted - Phase: {shape_pred_phase:.4f}, Transmittance: {shape_pred_trans:.4f}")
        
        results[shape_name] = {
            'control_points': shape_control_points,
            'true_phase': shape_phase,
            'true_trans': shape_trans
        }
    
    # 6. Visualize results
    print("\n7. Generating visualization charts...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original shape
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
    
    # Plot circle
    ax = axes[0, 1]
    sim_circle = Simulation(control_points=shapes["circle"])
    nurbs_circle = sim_circle.generate_complete_nurbs_curve(shapes["circle"])
    x_circle = [p[0] for p in nurbs_circle] + [nurbs_circle[0][0]]
    y_circle = [p[1] for p in nurbs_circle] + [nurbs_circle[0][1]]
    ax.plot(x_circle, y_circle, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["circle"][:, 0], shapes["circle"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Circle Shape\nPhase: {results["circle"]["true_phase"]:.3f}, Trans: {results["circle"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot square
    ax = axes[1, 0]
    sim_square = Simulation(control_points=shapes["square"])
    nurbs_square = sim_square.generate_complete_nurbs_curve(shapes["square"])
    x_square = [p[0] for p in nurbs_square] + [nurbs_square[0][0]]
    y_square = [p[1] for p in nurbs_square] + [nurbs_square[0][1]]
    ax.plot(x_square, y_square, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["square"][:, 0], shapes["square"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Square Shape\nPhase: {results["square"]["true_phase"]:.3f}, Trans: {results["square"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot ellipse
    ax = axes[1, 1]
    sim_ellipse = Simulation(control_points=shapes["ellipse"])
    nurbs_ellipse = sim_ellipse.generate_complete_nurbs_curve(shapes["ellipse"])
    x_ellipse = [p[0] for p in nurbs_ellipse] + [nurbs_ellipse[0][0]]
    y_ellipse = [p[1] for p in nurbs_ellipse] + [nurbs_ellipse[0][1]]
    ax.plot(x_ellipse, y_ellipse, 'b-', linewidth=2, label='NURBS Curve')
    ax.scatter(shapes["ellipse"][:, 0], shapes["ellipse"][:, 1], c='red', s=100, label='Control Points')
    ax.set_title(f'Ellipse Shape\nPhase: {results["ellipse"]["true_phase"]:.3f}, Trans: {results["ellipse"]["true_trans"]:.3f}')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('nurbs_shapes_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDone! Generated image saved as 'nurbs_shapes_comparison.png'")


def quick_demo():
    """Quick demo"""
    print("Quick Demo - Transformer Surrogate Model Prediction")
    print("="*40)
    
    # Create model instance
    model = NURBSTransformerModel(
        n_control_points=8,
        d_model=128,
        nhead=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    # Example control points
    control_points = np.array([
        (0.17, 0.02), (0.15, 0.15), (0.02, 0.17), (-0.15, 0.15),
        (-0.17, -0.02), (-0.15, -0.15), (-0.02, -0.17), (0.15, -0.15)
    ])
    
    print(f"Input control points:\n{control_points}")
    
    # Normalize
    normalized_input = normalize_control_points(control_points.reshape(1, 8, 2))
    
    # Check if model file exists
    model_path = "nurbs_transformer_model.pth"
    if os.path.exists(model_path):
        model.load_model(model_path)
        prediction = model.predict(normalized_input)
        from transformer_nurbs_model import denormalize_targets
        phase, transmittance = denormalize_targets(prediction)[0]
        print(f"\nPrediction results:")
        print(f"Phase: {phase:.4f} radians ({np.degrees(phase):.2f} degrees)")
        print(f"Transmittance: {transmittance:.4f}")
    else:
        print(f"\nModel file {model_path} does not exist")
        print("Please run training script first: python train_transformer_model.py --train")
        print("\nUsing random prediction values for demo:")
        phase = np.random.uniform(-np.pi, np.pi)
        transmittance = np.random.uniform(0.0, 1.0)
        print(f"Phase: {phase:.4f} radians ({np.degrees(phase):.2f} degrees)")
        print(f"Transmittance: {transmittance:.4f}")
    
    print(f"\nTransformer model architecture features:")
    print(f"- Input: (x,y) coordinates of 8 control points -> (8, 2) tensor")
    print(f"- Uses Transformer encoder to process serialized control points")
    print(f"- Output: phase and transmittance -> (2,) tensor")
    print(f"- Advantage: Can capture long-range dependencies between control points")


if __name__ == "__main__":
    print("NURBS Transformer Surrogate Model Example Program")
    print("=====================================")
    
    while True:
        print("\nPlease select demo mode:")
        print("1. Complete workflow demo")
        print("2. Quick demo")
        print("3. Exit")
        
        choice = input("Please enter your choice (1-3): ").strip()
        
        if choice == '1':
            example_complete_workflow()
        elif choice == '2':
            quick_demo()
        elif choice == '3':
            print("Exiting program")
            break
        else:
            print("Invalid choice, please try again")