"""
Visualize electric field distribution in NURBS meta-atom simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from nurbs_atoms_data import Simulation

def visualize_simulation():
    """Run simulation and visualize field distribution"""
    
    # Default control points
    control_points = np.array([
        (0.18, 0), (0.16, 0.16), (0, 0.18), (-0.16, 0.16),
        (-0.18, 0), (-0.16, -0.16), (0, -0.16), (0.16, -0.16)
    ])
    
    print("Creating simulation...")
    
    # Wavelength
    wavelength_um = 0.55  # 550 nm in μm
    freq = 1 / wavelength_um
    
    # Create simulation with modified parameters for visualization
    sim_obj = Simulation(control_points=control_points)
    
    # Recreate simulation with visualization-friendly settings
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=freq),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.8),
            size=mp.Vector3(0.5, 0.5, 0)
        )
    ]
    
    sim = mp.Simulation(
        cell_size=sim_obj.cell_size,
        geometry=sim_obj.geometry,
        sources=sources,
        boundary_layers=sim_obj.pml_layers,
        resolution=sim_obj.resolution,
        dimensions=3,
        k_point=mp.Vector3(0, 0, 0)  # Periodic BC in x,y with k=0
    )
    
    print(f"\nSimulation Configuration:")
    print(f"  Cell size: {sim_obj.cell_size}")
    print(f"  Resolution: {sim_obj.resolution} pixels/μm")
    print(f"  PML: z-direction only, thickness 0.5 μm")
    print(f"  Boundary conditions:")
    print(f"    - X, Y: Periodic (Bloch with k=0)")
    print(f"    - Z: PML (absorbing)")
    print(f"  Wavelength: {wavelength_um} μm ({wavelength_um*1000} nm)")
    
    # Run for a bit to let fields develop
    print("\nRunning simulation...")
    sim.run(until=100)
    
    # Get field data at different cross-sections
    print("\nExtracting field data...")
    
    # XZ plane (y=0)
    xz_plane = sim.get_array(center=mp.Vector3(0, 0, 0), 
                              size=mp.Vector3(0.5, 0, 3.0), 
                              component=mp.Ex)
    
    # XY plane (z=0.3, through the meta-atom)
    xy_plane_meta = sim.get_array(center=mp.Vector3(0, 0, 0.3), 
                                   size=mp.Vector3(0.5, 0.5, 0), 
                                   component=mp.Ex)
    
    # XY plane (z=0.8, above the meta-atom)
    xy_plane_above = sim.get_array(center=mp.Vector3(0, 0, 0.8), 
                                    size=mp.Vector3(0.5, 0.5, 0), 
                                    component=mp.Ex)
    
    # Get epsilon for structure visualization
    eps_xz = sim.get_array(center=mp.Vector3(0, 0, 0), 
                           size=mp.Vector3(0.5, 0, 3.0), 
                           component=mp.Dielectric)
    
    eps_xy = sim.get_array(center=mp.Vector3(0, 0, 0.3), 
                           size=mp.Vector3(0.5, 0.5, 0), 
                           component=mp.Dielectric)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot XZ cross-section (side view)
    ax = axes[0, 0]
    extent_xz = [-0.25, 0.25, -1.5, 1.5]
    im = ax.imshow(np.real(xz_plane).T, cmap='RdBu', extent=extent_xz, 
                   aspect='auto', origin='lower', vmin=-np.max(np.abs(xz_plane)), 
                   vmax=np.max(np.abs(xz_plane)))
    ax.contour(eps_xz.T, levels=[3], colors='white', linewidths=1, 
               extent=extent_xz, origin='lower')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Z (μm)')
    ax.set_title('Ex Field - XZ Plane (y=0)')
    ax.axhline(y=0, color='yellow', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.6, color='yellow', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label='Ex')
    
    # Plot XY cross-section at meta-atom height
    ax = axes[0, 1]
    extent_xy = [-0.25, 0.25, -0.25, 0.25]
    im = ax.imshow(np.real(xy_plane_meta).T, cmap='RdBu', extent=extent_xy, 
                   aspect='equal', origin='lower', vmin=-np.max(np.abs(xy_plane_meta)), 
                   vmax=np.max(np.abs(xy_plane_meta)))
    ax.contour(eps_xy.T, levels=[3], colors='white', linewidths=1.5, 
               extent=extent_xy, origin='lower')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Ex Field - XY Plane (z=0.3 μm, through meta-atom)')
    plt.colorbar(im, ax=ax, label='Ex')
    
    # Plot XY cross-section above meta-atom
    ax = axes[1, 0]
    im = ax.imshow(np.real(xy_plane_above).T, cmap='RdBu', extent=extent_xy, 
                   aspect='equal', origin='lower', vmin=-np.max(np.abs(xy_plane_above)), 
                   vmax=np.max(np.abs(xy_plane_above)))
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Ex Field - XY Plane (z=0.8 μm, above meta-atom)')
    plt.colorbar(im, ax=ax, label='Ex')
    
    # Plot structure (epsilon)
    ax = axes[1, 1]
    im = ax.imshow(eps_xy.T, cmap='Blues', extent=extent_xy, 
                   aspect='equal', origin='lower')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Dielectric Structure (ε) - XY Plane (z=0.3 μm)')
    plt.colorbar(im, ax=ax, label='ε')
    
    plt.tight_layout()
    plt.savefig('field_distribution.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to 'field_distribution.png'")
    plt.show()
    
    return sim

if __name__ == "__main__":
    visualize_simulation()
