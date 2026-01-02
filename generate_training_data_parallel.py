#!/usr/bin/env python
"""
Parallel generation of NURBS metasurface training data
Supports MPI parallel computing for multi-core workstations (e.g., 128 cores)

Usage:
    # Single machine multiprocessing mode (recommended for workstations)
    mpirun -np 128 python generate_training_data_parallel.py --num_samples 50000 --output_dir ./training_data
    
    # Or use Python multiprocessing (no MPI required)
    python generate_training_data_parallel.py --num_samples 50000 --num_workers 128 --mode multiprocessing
    
    # Test mode (generate few samples for verification)
    python generate_training_data_parallel.py --num_samples 100 --num_workers 4 --mode multiprocessing --test
"""

import numpy as np
import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constraint parameters
CELL_SIZE = 0.5  # Unit cell size (um)
MIN_RADIUS = 0.05  # Minimum radius (um)
MAX_RADIUS = 0.22  # Maximum radius (um), ensure structure fits in cell
MIN_FEATURE_SIZE = 0.03  # Minimum feature size (um), manufacturing constraint


def generate_random_control_points(seed=None):
    """
    Generate random control points satisfying physical constraints
    
    Physical constraints:
    1. All points must be within the unit cell (+/-0.25um)
    2. Structure must have minimum feature size (manufacturability)
    3. Control points arranged counter-clockwise
    4. Structure should be convex or nearly convex (avoid self-intersection)
    5. Structure should have reasonable size (not too small or too large)
    
    Returns:
        control_points: (8, 2) numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Method: Use polar coordinates to ensure points are sorted by angle
    # 8 control points correspond to 8 angular directions
    base_angles = np.linspace(0, 2*np.pi, 9)[:-1]  # [0, pi/4, pi/2, ..., 7pi/4]
    
    # Add angle perturbation (maintain order)
    angle_perturbation = np.random.uniform(-np.pi/12, np.pi/12, 8)  # +/-15 degrees
    angles = base_angles + angle_perturbation
    
    # Ensure angles are monotonically increasing
    for i in range(1, 8):
        if angles[i] <= angles[i-1]:
            angles[i] = angles[i-1] + 0.05
    
    # Generate radii (with variation to produce different shapes)
    # Base radius
    base_radius = np.random.uniform(MIN_RADIUS + 0.02, MAX_RADIUS - 0.02)
    
    # Add radius variation (produce ellipse, square, etc.)
    # Use Fourier modes for smooth variation
    n_modes = np.random.randint(1, 4)  # 1-3 Fourier modes
    radius_variation = np.zeros(8)
    
    for mode in range(1, n_modes + 1):
        amplitude = np.random.uniform(0, 0.05) / mode  # Higher modes have smaller amplitude
        phase = np.random.uniform(0, 2*np.pi)
        radius_variation += amplitude * np.cos(mode * base_angles + phase)
    
    radii = base_radius + radius_variation
    
    # Ensure radii are within valid range
    radii = np.clip(radii, MIN_RADIUS, MAX_RADIUS)
    
    # Convert to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Add small random offset (increase diversity)
    x += np.random.uniform(-0.01, 0.01, 8)
    y += np.random.uniform(-0.01, 0.01, 8)
    
    # Ensure all points are within unit cell
    max_coord = CELL_SIZE / 2 - 0.02  # Leave some margin
    x = np.clip(x, -max_coord, max_coord)
    y = np.clip(y, -max_coord, max_coord)
    
    control_points = np.column_stack([x, y])
    
    # Validate generated control points
    if not validate_control_points(control_points):
        # If validation fails, use more conservative parameters
        return generate_conservative_control_points(seed)
    
    return control_points


def generate_conservative_control_points(seed=None):
    """
    Generate conservative control points (fallback when validation fails)
    """
    if seed is not None:
        np.random.seed(seed + 1000000)
    
    # Use simple ellipse shape
    base_angles = np.linspace(0, 2*np.pi, 9)[:-1]
    
    # Ellipse parameters
    a = np.random.uniform(0.08, 0.16)  # Semi-major axis
    b = np.random.uniform(0.08, 0.16)  # Semi-minor axis
    rotation = np.random.uniform(0, np.pi)  # Rotation angle
    
    # Generate points on ellipse
    x = a * np.cos(base_angles)
    y = b * np.sin(base_angles)
    
    # Rotate
    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    
    return np.column_stack([x_rot, y_rot])


def validate_control_points(control_points):
    """
    Validate if control points satisfy physical constraints
    
    Returns:
        bool: whether valid
    """
    # Check 1: All points within unit cell
    max_coord = CELL_SIZE / 2
    if np.any(np.abs(control_points) > max_coord):
        return False
    
    # Check 2: Structure not too small
    centroid = np.mean(control_points, axis=0)
    distances = np.linalg.norm(control_points - centroid, axis=1)
    if np.mean(distances) < MIN_RADIUS:
        return False
    
    # Check 3: Distance between adjacent points not too small (minimum feature size)
    for i in range(8):
        next_i = (i + 1) % 8
        dist = np.linalg.norm(control_points[i] - control_points[next_i])
        if dist < MIN_FEATURE_SIZE:
            return False
    
    # Check 4: Check for self-intersection (simplified: ensure points sorted by angle)
    angles = np.arctan2(control_points[:, 1] - centroid[1], 
                        control_points[:, 0] - centroid[0])
    # Convert angles to [0, 2pi) range
    angles = np.mod(angles, 2 * np.pi)
    
    # Check if angles are roughly monotonic (allow some error)
    sorted_indices = np.argsort(angles)
    expected_order = np.arange(8)
    
    # Find starting point (point with smallest angle should be first)
    start_idx = sorted_indices[0]
    rotated_expected = np.roll(expected_order, -start_idx)
    
    # Allow some order deviation
    order_diff = np.abs(sorted_indices - np.roll(rotated_expected, start_idx))
    if np.max(order_diff) > 2:  # Allow at most 2 position deviation
        return False
    
    return True


def simulate_single_sample(args):
    """
    Simulate single sample (for multiprocessing)
    
    Args:
        args: (sample_id, control_points, wavelength_nm, output_dir)
    
    Returns:
        dict: simulation result
    """
    sample_id, control_points, wavelength_nm, output_dir = args
    
    try:
        # Import meep (import in subprocess to avoid MPI conflicts)
        import meep as mp
        from nurbs_atoms_data import Simulation
        
        # Create simulation object
        sim = Simulation(control_points=control_points)
        
        # Run simulation
        wavelength_m = wavelength_nm * 1e-9
        transmittance, phase = sim.run_forward(
            wavelength_start=wavelength_m,
            wavelength_stop=wavelength_m,
            normalize=True
        )
        
        # Reset simulation to release memory
        sim.reset()
        
        result = {
            'sample_id': sample_id,
            'control_points': control_points.tolist(),
            'wavelength_nm': wavelength_nm,
            'transmittance': float(transmittance),
            'phase': float(phase),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        result = {
            'sample_id': sample_id,
            'control_points': control_points.tolist(),
            'wavelength_nm': wavelength_nm,
            'transmittance': None,
            'phase': None,
            'success': False,
            'error': str(e)
        }
    
    return result


def run_multiprocessing(num_samples, num_workers, wavelength_nm, output_dir, test_mode=False):
    """
    Use Python multiprocessing for parallel computing
    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    
    if num_workers <= 0:
        num_workers = cpu_count()
    
    print(f"=" * 60)
    print(f"NURBS Metasurface Training Data Generation (Multiprocessing Mode)")
    print(f"=" * 60)
    print(f"Number of samples: {num_samples}")
    print(f"Number of workers: {num_workers}")
    print(f"Wavelength: {wavelength_nm} nm")
    print(f"Output directory: {output_dir}")
    print(f"Test mode: {test_mode}")
    print(f"=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all control points
    print("\nGenerating random control points...")
    all_control_points = []
    for i in tqdm(range(num_samples), desc="Generating control points"):
        cp = generate_random_control_points(seed=i)
        all_control_points.append(cp)
    
    # Prepare task arguments
    tasks = [
        (i, all_control_points[i], wavelength_nm, output_dir)
        for i in range(num_samples)
    ]
    
    # Run parallel simulation
    print(f"\nStarting parallel simulation ({num_workers} processes)...")
    start_time = time.time()
    
    results = []
    failed_count = 0
    
    # Use process pool
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better efficiency
        for result in tqdm(pool.imap_unordered(simulate_single_sample, tasks), 
                          total=num_samples, desc="Simulation progress"):
            results.append(result)
            if not result['success']:
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    # Statistics
    successful_results = [r for r in results if r['success']]
    
    print(f"\n" + "=" * 60)
    print(f"Simulation complete!")
    print(f"=" * 60)
    print(f"Total samples: {num_samples}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average per sample: {elapsed_time/num_samples:.2f} seconds")
    print(f"Throughput: {num_samples/elapsed_time:.2f} samples/second")
    
    # Save results
    save_results(successful_results, output_dir, wavelength_nm)
    
    return successful_results


def run_mpi(num_samples, wavelength_nm, output_dir, test_mode=False):
    """
    Use MPI for parallel computing (for clusters and large workstations)
    """
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"=" * 60)
        print(f"NURBS Metasurface Training Data Generation (MPI Mode)")
        print(f"=" * 60)
        print(f"Number of samples: {num_samples}")
        print(f"MPI processes: {size}")
        print(f"Wavelength: {wavelength_nm} nm")
        print(f"Output directory: {output_dir}")
        print(f"Test mode: {test_mode}")
        print(f"=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Synchronize all processes
    comm.Barrier()
    
    # Calculate sample range for each process
    samples_per_proc = num_samples // size
    remainder = num_samples % size
    
    if rank < remainder:
        local_start = rank * (samples_per_proc + 1)
        local_count = samples_per_proc + 1
    else:
        local_start = rank * samples_per_proc + remainder
        local_count = samples_per_proc
    
    if rank == 0:
        print(f"\nEach process handles approximately {samples_per_proc} samples")
    
    # Import meep
    import meep as mp
    from nurbs_atoms_data import Simulation
    
    # Local results
    local_results = []
    
    start_time = time.time()
    
    for i in range(local_count):
        global_idx = local_start + i
        
        # Generate control points
        control_points = generate_random_control_points(seed=global_idx)
        
        try:
            # Create simulation
            sim = Simulation(control_points=control_points)
            
            # Run simulation
            wavelength_m = wavelength_nm * 1e-9
            transmittance, phase = sim.run_forward(
                wavelength_start=wavelength_m,
                wavelength_stop=wavelength_m,
                normalize=True
            )
            
            # Reset simulation
            sim.reset()
            
            result = {
                'sample_id': global_idx,
                'control_points': control_points.tolist(),
                'wavelength_nm': wavelength_nm,
                'transmittance': float(transmittance),
                'phase': float(phase),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            result = {
                'sample_id': global_idx,
                'control_points': control_points.tolist(),
                'wavelength_nm': wavelength_nm,
                'transmittance': None,
                'phase': None,
                'success': False,
                'error': str(e)
            }
        
        local_results.append(result)
        
        # Print progress periodically
        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (local_count - i - 1)
            print(f"Process 0 progress: {i+1}/{local_count}, elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    # Gather all results to rank 0
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Merge results
        merged_results = []
        for proc_results in all_results:
            merged_results.extend(proc_results)
        
        elapsed_time = time.time() - start_time
        
        # Statistics
        successful_results = [r for r in merged_results if r['success']]
        failed_count = len(merged_results) - len(successful_results)
        
        print(f"\n" + "=" * 60)
        print(f"Simulation complete!")
        print(f"=" * 60)
        print(f"Total samples: {num_samples}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {failed_count}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average per sample: {elapsed_time/num_samples:.2f} seconds")
        print(f"Throughput: {num_samples/elapsed_time:.2f} samples/second")
        
        # Save results
        save_results(successful_results, output_dir, wavelength_nm)
        
        return successful_results
    
    return None


def save_results(results, output_dir, wavelength_nm):
    """
    Save simulation results
    """
    if not results:
        print("Warning: No successful results to save")
        return
    
    # Sort by sample_id
    results = sorted(results, key=lambda x: x['sample_id'])
    
    # Extract data
    control_points = np.array([r['control_points'] for r in results])
    transmittances = np.array([r['transmittance'] for r in results])
    phases = np.array([r['phase'] for r in results])
    sample_ids = np.array([r['sample_id'] for r in results])
    
    # Save as numpy format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    np.save(os.path.join(output_dir, f'control_points_{timestamp}.npy'), control_points)
    np.save(os.path.join(output_dir, f'transmittances_{timestamp}.npy'), transmittances)
    np.save(os.path.join(output_dir, f'phases_{timestamp}.npy'), phases)
    np.save(os.path.join(output_dir, f'sample_ids_{timestamp}.npy'), sample_ids)
    
    # Save combined data file
    combined_data = {
        'control_points': control_points,
        'transmittances': transmittances,
        'phases': phases,
        'sample_ids': sample_ids,
        'wavelength_nm': wavelength_nm,
        'num_samples': len(results),
        'timestamp': timestamp
    }
    np.savez(os.path.join(output_dir, f'training_data_{timestamp}.npz'), **combined_data)
    
    # Save metadata
    metadata = {
        'num_samples': len(results),
        'wavelength_nm': wavelength_nm,
        'timestamp': timestamp,
        'cell_size_um': CELL_SIZE,
        'min_radius_um': MIN_RADIUS,
        'max_radius_um': MAX_RADIUS,
        'min_feature_size_um': MIN_FEATURE_SIZE,
        'transmittance_stats': {
            'mean': float(np.mean(transmittances)),
            'std': float(np.std(transmittances)),
            'min': float(np.min(transmittances)),
            'max': float(np.max(transmittances))
        },
        'phase_stats': {
            'mean': float(np.mean(phases)),
            'std': float(np.std(phases)),
            'min': float(np.min(phases)),
            'max': float(np.max(phases))
        }
    }
    
    with open(os.path.join(output_dir, f'metadata_{timestamp}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to: {output_dir}")
    print(f"  - control_points_{timestamp}.npy: control point data {control_points.shape}")
    print(f"  - transmittances_{timestamp}.npy: transmittance data {transmittances.shape}")
    print(f"  - phases_{timestamp}.npy: phase data {phases.shape}")
    print(f"  - training_data_{timestamp}.npz: combined data file")
    print(f"  - metadata_{timestamp}.json: metadata")
    
    # Print statistics
    print(f"\nData statistics:")
    print(f"  Transmittance: {np.mean(transmittances):.4f} +/- {np.std(transmittances):.4f} "
          f"(range: {np.min(transmittances):.4f} - {np.max(transmittances):.4f})")
    print(f"  Phase: {np.mean(phases):.4f} +/- {np.std(phases):.4f} rad "
          f"(range: {np.min(phases):.4f} - {np.max(phases):.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel generation of NURBS metasurface training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use multiprocessing mode (recommended for single machine)
  python generate_training_data_parallel.py --num_samples 50000 --num_workers 128 --mode multiprocessing
  
  # Use MPI mode (requires mpi4py)
  mpirun -np 128 python generate_training_data_parallel.py --num_samples 50000 --mode mpi
  
  # Test mode
  python generate_training_data_parallel.py --num_samples 100 --num_workers 4 --test
        """
    )
    
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of samples to generate (default: 50000)')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='Number of worker processes, -1 means use all CPU cores (default: -1)')
    parser.add_argument('--wavelength', type=float, default=550,
                        help='Simulation wavelength (nm) (default: 550)')
    parser.add_argument('--output_dir', type=str, default='./training_data',
                        help='Output directory (default: ./training_data)')
    parser.add_argument('--mode', type=str, choices=['multiprocessing', 'mpi', 'auto'],
                        default='auto',
                        help='Parallel mode: multiprocessing, mpi, or auto (default: auto)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode, use fewer samples for verification')
    
    args = parser.parse_args()
    
    # Reduce sample count in test mode
    if args.test:
        args.num_samples = min(args.num_samples, 100)
        print("Test mode: sample count limited to", args.num_samples)
    
    # Auto select mode
    if args.mode == 'auto':
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                args.mode = 'mpi'
            else:
                args.mode = 'multiprocessing'
        except ImportError:
            args.mode = 'multiprocessing'
    
    # Run
    if args.mode == 'mpi':
        run_mpi(args.num_samples, args.wavelength, args.output_dir, args.test)
    else:
        run_multiprocessing(args.num_samples, args.num_workers, args.wavelength, 
                           args.output_dir, args.test)


if __name__ == '__main__':
    main()
