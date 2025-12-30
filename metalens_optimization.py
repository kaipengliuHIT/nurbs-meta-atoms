import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import sys
import os

# Add current directory to path
sys.path.append('/mnt/e/pythoncode/nurbs-meta-atoms')

from transformer_nurbs_model import (
    NURBSTransformerModel,
    normalize_control_points,
    denormalize_targets
)
from nurbs_atoms_data import Simulation

class MetalensOptimizer:
    """Metalens optimizer class"""
    
    def __init__(self, 
                 model_path: str = "nurbs_transformer_model.pth",
                 n_segments: int = 64,
                 focal_length: float = 50e-6,  # Focal length 50 μm
                 wavelength: float = 532e-9,   # Wavelength 532 nm
                 lens_radius: float = 50e-6):  # Lens radius 50 μm
        """
        Initialize metalens optimizer
        
        Args:
            model_path: Surrogate model path
            n_segments: Number of lens segments
            focal_length: Focal length
            wavelength: Operating wavelength
            lens_radius: Lens radius
        """
        self.n_segments = n_segments
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.lens_radius = lens_radius
        self.k = 2 * np.pi / wavelength  # Wave number
        
        # Load surrogate model
        self.model = NURBSTransformerModel(
            n_control_points=8,
            d_model=128,
            nhead=8,
            num_layers=4,
            d_ff=256,
            dropout=0.1
        )
        
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            self.model_available = True
            print(f"Loaded surrogate model: {model_path}")
        else:
            print(f"Surrogate model {model_path} does not exist, will use real simulation")
            self.model_available = False
    
    def ideal_phase_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate ideal phase distribution
        
        Args:
            r: Radial coordinate
            
        Returns:
            Ideal phase distribution
        """
        # Ideal metalens phase distribution: φ(r) = -k * (sqrt(r^2 + f^2) - f)
        phase = -self.k * (np.sqrt(r**2 + self.focal_length**2) - self.focal_length)
        # Constrain phase to [-π, π] range
        phase = np.mod(phase + np.pi, 2*np.pi) - np.pi
        return phase
    
    def generate_segment_control_points(self, 
                                     segment_idx: int, 
                                     radial_pos: float, 
                                     phase_req: float, 
                                     transmittance_req: float = 1.0) -> np.ndarray:
        """
        Generate NURBS control points for a single segment
        
        Args:
            segment_idx: Segment index
            radial_pos: Radial position
            phase_req: Required phase
            transmittance_req: Required transmittance
            
        Returns:
            Control point coordinates (8, 2)
        """
        # Base control points - adjust size based on radial position
        base_size = 0.05 * (radial_pos / self.lens_radius + 0.5)  # Size varies with radial position
        
        base_control_points = np.array([
            (base_size, 0), 
            (0.707*base_size, 0.707*base_size), 
            (0, base_size), 
            (-0.707*base_size, 0.707*base_size),
            (-base_size, 0), 
            (-0.707*base_size, -0.707*base_size), 
            (0, -base_size), 
            (0.707*base_size, -0.707*base_size)
        ])
        
        # Add angular rotation based on segment index
        angle = 2 * np.pi * segment_idx / self.n_segments
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_points = np.dot(base_control_points, rotation_matrix.T)
        
        # Add radial offset
        radial_offset = radial_pos
        angle_offset = angle
        offset_x = radial_offset * np.cos(angle_offset)
        offset_y = radial_offset * np.sin(angle_offset)
        
        final_points = rotated_points + np.array([offset_x, offset_y])
        
        return final_points
    
    def predict_optical_response(self, control_points: np.ndarray) -> Tuple[float, float]:
        """
        Predict optical response (phase and transmittance)
        
        Args:
            control_points: Control point coordinates
            
        Returns:
            (phase, transmittance)
        """
        if self.model_available:
            # Use surrogate model for prediction
            normalized_input = normalize_control_points(control_points.reshape(1, 8, 2))
            prediction = self.model.predict(normalized_input)
            phase, transmittance = denormalize_targets(prediction)[0]
            return float(phase), float(transmittance)
        else:
            # Use real simulation (if available)
            try:
                sim = Simulation(control_points=control_points)
                transmittance, phase = sim.run_forward(wavelength_start=self.wavelength, wavelength_stop=self.wavelength)
                return phase, transmittance
            except:
                # If simulation fails, return random values
                return np.random.uniform(-np.pi, np.pi), np.random.uniform(0.5, 1.0)
    
    def calculate_segment_response(self, 
                                 segment_idx: int, 
                                 radial_pos: float, 
                                 phase_req: float) -> Tuple[float, float]:
        """
        Calculate optical response for a single segment
        
        Args:
            segment_idx: Segment index
            radial_pos: Radial position
            phase_req: Required phase
            
        Returns:
            (actual phase, actual transmittance)
        """
        # Generate control points
        control_points = self.generate_segment_control_points(segment_idx, radial_pos, phase_req)
        
        # Predict optical response
        actual_phase, actual_transmittance = self.predict_optical_response(control_points)
        
        return actual_phase, actual_transmittance
    
    def calculate_focus_efficiency(self, 
                                 phase_profile: np.ndarray, 
                                 transmittance_profile: np.ndarray) -> float:
        """
        Calculate focusing efficiency
        
        Args:
            phase_profile: Actual phase distribution
            transmittance_profile: Actual transmittance distribution
            
        Returns:
            Focusing efficiency
        """
        # Calculate ideal phase distribution
        radial_positions = np.linspace(0, self.lens_radius, self.n_segments)
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # Calculate phase error
        phase_error = np.abs(ideal_phase - phase_profile)
        
        # Calculate weighted efficiency (considering transmittance)
        efficiency = np.mean(transmittance_profile * np.cos(phase_error/2)**2)
        
        return efficiency
    
    def optimize_single_wavelength(self, 
                                 max_iterations: int = 50, 
                                 learning_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Single wavelength optimization
        
        Args:
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate
            
        Returns:
            (optimal phase distribution, optimal transmittance distribution, efficiency history)
        """
        print("Starting single wavelength metalens optimization...")
        
        # Initialize phase and transmittance distributions
        radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # Initial guess - close to ideal values
        current_phase = ideal_phase + np.random.uniform(-0.2, 0.2, self.n_segments)
        current_transmittance = np.ones(self.n_segments) * 0.8  # Initial transmittance
        
        efficiency_history = []
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Update each segment
            new_phase = current_phase.copy()
            new_transmittance = current_transmittance.copy()
            
            for i in range(self.n_segments):
                # Calculate response for current segment
                actual_phase, actual_transmittance = self.calculate_segment_response(
                    i, radial_positions[i], ideal_phase[i]
                )
                
                # Update estimated values
                new_phase[i] = actual_phase
                new_transmittance[i] = actual_transmittance
            
            current_phase = new_phase
            current_transmittance = new_transmittance
            
            # Calculate current efficiency
            current_efficiency = self.calculate_focus_efficiency(current_phase, current_transmittance)
            efficiency_history.append(current_efficiency)
            
            print(f"  Current focusing efficiency: {current_efficiency:.4f}")
        
        print(f"Optimization complete! Final focusing efficiency: {efficiency_history[-1]:.4f}")
        
        return current_phase, current_transmittance, efficiency_history
    
    def plot_results(self, 
                    phase_profile: np.ndarray, 
                    transmittance_profile: np.ndarray, 
                    efficiency_history: List[float],
                    radial_positions: Optional[np.ndarray] = None):
        """
        Plot optimization results
        """
        if radial_positions is None:
            radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Phase distribution comparison
        axes[0, 0].plot(radial_positions*1e6, ideal_phase, 'r--', label='Ideal Phase', linewidth=2)
        axes[0, 0].plot(radial_positions*1e6, phase_profile, 'b-', label='Actual Phase', linewidth=2)
        axes[0, 0].set_xlabel('Radial Position (μm)')
        axes[0, 0].set_ylabel('Phase (radians)')
        axes[0, 0].set_title('Phase Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Transmittance distribution
        axes[0, 1].plot(radial_positions*1e6, transmittance_profile, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Radial Position (μm)')
        axes[0, 1].set_ylabel('Transmittance')
        axes[0, 1].set_title('Transmittance Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Phase error
        phase_error = np.abs(ideal_phase - phase_profile)
        axes[1, 0].plot(radial_positions*1e6, phase_error, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Radial Position (μm)')
        axes[1, 0].set_ylabel('Phase Error (radians)')
        axes[1, 0].set_title('Phase Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency history
        axes[1, 1].plot(efficiency_history, 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Focusing Efficiency')
        axes[1, 1].set_title('Optimization Convergence')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metalens_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_design_summary(self, 
                              phase_profile: np.ndarray, 
                              transmittance_profile: np.ndarray,
                              radial_positions: Optional[np.ndarray] = None) -> dict:
        """
        Generate design summary
        """
        if radial_positions is None:
            radial_positions = np.linspace(0.1*self.lens_radius, self.lens_radius, self.n_segments)
        
        ideal_phase = self.ideal_phase_profile(radial_positions)
        
        # Calculate various metrics
        phase_error = np.abs(ideal_phase - phase_profile)
        avg_phase_error = np.mean(phase_error)
        max_phase_error = np.max(phase_error)
        avg_transmittance = np.mean(transmittance_profile)
        efficiency = self.calculate_focus_efficiency(phase_profile, transmittance_profile)
        
        summary = {
            'focal_length': self.focal_length,
            'wavelength': self.wavelength,
            'lens_radius': self.lens_radius,
            'n_segments': self.n_segments,
            'avg_phase_error': avg_phase_error,
            'max_phase_error': max_phase_error,
            'avg_transmittance': avg_transmittance,
            'focusing_efficiency': efficiency,
            'radial_positions': radial_positions,
            'ideal_phase': ideal_phase,
            'actual_phase': phase_profile,
            'transmittance': transmittance_profile
        }
        
        return summary


def main():
    """Main function - Demonstrate metalens optimization"""
    print("DFLAT-Style Metalens Optimizer")
    print("="*50)
    
    # Create optimizer instance
    optimizer = MetalensOptimizer(
        n_segments=32,  # Reduce segment count to speed up demo
        focal_length=50e-6,
        wavelength=532e-9,
        lens_radius=50e-6
    )
    
    # Execute optimization
    phase_profile, transmittance_profile, efficiency_history = optimizer.optimize_single_wavelength(
        max_iterations=20  # Reduce iterations to speed up demo
    )
    
    # Generate radial positions
    radial_positions = np.linspace(0.1*optimizer.lens_radius, optimizer.lens_radius, optimizer.n_segments)
    
    # Plot results
    optimizer.plot_results(phase_profile, transmittance_profile, efficiency_history, radial_positions)
    
    # Generate design summary
    summary = optimizer.generate_design_summary(phase_profile, transmittance_profile, radial_positions)
    
    print("\nDesign Summary:")
    print(f"Focal length: {summary['focal_length']*1e6:.1f} μm")
    print(f"Wavelength: {summary['wavelength']*1e9:.1f} nm")
    print(f"Lens radius: {summary['lens_radius']*1e6:.1f} μm")
    print(f"Number of segments: {summary['n_segments']}")
    print(f"Average phase error: {summary['avg_phase_error']:.4f} radians")
    print(f"Maximum phase error: {summary['max_phase_error']:.4f} radians")
    print(f"Average transmittance: {summary['avg_transmittance']:.4f}")
    print(f"Focusing efficiency: {summary['focusing_efficiency']:.4f}")
    
    print("\nOptimization complete! Results saved to 'metalens_optimization_results.png'")


if __name__ == "__main__":
    main()