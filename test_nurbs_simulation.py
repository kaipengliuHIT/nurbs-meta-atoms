"""
Test NURBS metasurface unit cell simulation script
Uses Meep for FDTD simulation, calculates phase and transmittance, and visualizes field distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from nurbs_atoms_data import Simulation

def create_test_control_points():
    """
    Create reasonable control points for a 4-segment NURBS curve
    
    4-segment NURBS curve is defined by 8 control points, each segment uses 3 control points
    (adjacent segments share endpoints)
    Control points are distributed within the unit cell, forming a rounded rectangle shape
    
    Unit cell size: 0.5um x 0.5um
    Control point range: approximately +/-0.18um (ensure structure fits in cell)
    """
    # Define 8 control points forming a slightly asymmetric shape for interesting phase response
    # Control points arranged counter-clockwise
    control_points = np.array([
        (0.16, 0.02),    # Point 0: right side, slightly above center
        (0.14, 0.14),    # Point 1: upper right corner
        (0.02, 0.16),    # Point 2: top side, slightly right of center
        (-0.14, 0.14),   # Point 3: upper left corner
        (-0.16, -0.02),  # Point 4: left side, slightly below center
        (-0.14, -0.14),  # Point 5: lower left corner
        (-0.02, -0.16),  # Point 6: bottom side, slightly left of center
        (0.14, -0.14)    # Point 7: lower right corner
    ])
    
    return control_points


def run_simulation_and_visualize(control_points, wavelength_nm=550):
    """
    Run simulation and visualize results
    
    Args:
        control_points: (8,2) array, NURBS control point coordinates (unit: um)
        wavelength_nm: wavelength (unit: nm)
    
    Returns:
        transmittance: transmittance
        phase: phase (radians)
    """
    print("=" * 60)
    print("NURBS Metasurface Unit Cell Simulation Test")
    print("=" * 60)
    
    # 1. Display control point information
    print("\n1. Control point coordinates (unit: um):")
    for i, pt in enumerate(control_points):
        print(f"   Point {i}: ({pt[0]:.3f}, {pt[1]:.3f})")
    
    # 2. Create simulation object
    print(f"\n2. Creating simulation object...")
    sim_obj = Simulation(control_points=control_points)
    
    # Display simulation parameters
    print(f"   Cell size: {sim_obj.cell_size}")
    print(f"   Resolution: {sim_obj.resolution} pixels/um")
    print(f"   TiO2 permittivity: {sim_obj.TiO2_material.epsilon_diag.x}")
    print(f"   Wavelength: {wavelength_nm} nm")
    
    # 3. Visualize NURBS curve shape
    print(f"\n3. Generating NURBS curve...")
    nurbs_points = sim_obj.generate_complete_nurbs_curve(control_points)
    print(f"   Generated {len(nurbs_points)} curve sample points")
    
    # 4. Run simulation
    print(f"\n4. Running FDTD simulation...")
    print(f"   First running reference simulation (no structure) for normalization...")
    wavelength_m = wavelength_nm * 1e-9
    transmittance, phase = sim_obj.run_forward(
        wavelength_start=wavelength_m, 
        wavelength_stop=wavelength_m,
        normalize=True  # Enable normalization
    )
    
    print(f"\n5. Simulation results:")
    print(f"   Normalized transmittance: {transmittance:.4f} ({transmittance*100:.2f}%)")
    print(f"   Relative phase: {phase:.4f} rad ({np.degrees(phase):.2f} deg)")
    
    # 6. Extract field distribution data
    print(f"\n6. Extracting field distribution data...")
    
    # Get field at different cross-sections
    # XZ plane (y=0) - side view
    xz_field = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0), 
        size=mp.Vector3(0.5, 0, 3.0), 
        component=mp.Ex
    )
    
    # XY plane (z=0.3um) - cross-section through meta-atom
    xy_field_meta = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0.3), 
        size=mp.Vector3(0.5, 0.5, 0), 
        component=mp.Ex
    )
    
    # XY plane (z=0.8um) - cross-section above meta-atom
    xy_field_above = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0.8), 
        size=mp.Vector3(0.5, 0.5, 0), 
        component=mp.Ex
    )
    
    # Get permittivity distribution (for displaying structure)
    eps_xz = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0), 
        size=mp.Vector3(0.5, 0, 3.0), 
        component=mp.Dielectric
    )
    
    eps_xy = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0.3), 
        size=mp.Vector3(0.5, 0.5, 0), 
        component=mp.Dielectric
    )
    
    # 7. Create visualization plots
    print(f"\n7. Generating visualization plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: NURBS curve and control points
    ax1 = fig.add_subplot(2, 3, 1)
    nurbs_x = [p[0] for p in nurbs_points] + [nurbs_points[0][0]]
    nurbs_y = [p[1] for p in nurbs_points] + [nurbs_points[0][1]]
    ax1.plot(nurbs_x, nurbs_y, 'b-', linewidth=2, label='NURBS Curve')
    ax1.scatter(control_points[:, 0], control_points[:, 1], 
                c='red', s=100, zorder=5, label='Control Points')
    for i, pt in enumerate(control_points):
        ax1.annotate(f'{i}', (pt[0], pt[1]), textcoords="offset points", 
                    xytext=(5, 5), fontsize=8)
    ax1.set_xlabel('X (um)')
    ax1.set_ylabel('Y (um)')
    ax1.set_title('NURBS Meta-atom Shape')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.25, 0.25)
    ax1.set_ylim(-0.25, 0.25)
    
    # Subplot 2: XZ cross-section field distribution
    ax2 = fig.add_subplot(2, 3, 2)
    extent_xz = [-0.25, 0.25, -1.5, 1.5]
    vmax_xz = np.max(np.abs(xz_field)) if np.max(np.abs(xz_field)) > 0 else 1
    im2 = ax2.imshow(np.real(xz_field).T, cmap='RdBu', extent=extent_xz,
                     aspect='auto', origin='lower', vmin=-vmax_xz, vmax=vmax_xz)
    ax2.contour(eps_xz.T, levels=[3], colors='white', linewidths=1,
                extent=extent_xz, origin='lower')
    ax2.axhline(y=0, color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0.6, color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('X (um)')
    ax2.set_ylabel('Z (um)')
    ax2.set_title('Ex Field - XZ Plane (y=0)')
    plt.colorbar(im2, ax=ax2, label='Ex')
    
    # Subplot 3: XY cross-section field distribution (through meta-atom)
    ax3 = fig.add_subplot(2, 3, 3)
    extent_xy = [-0.25, 0.25, -0.25, 0.25]
    vmax_xy = np.max(np.abs(xy_field_meta)) if np.max(np.abs(xy_field_meta)) > 0 else 1
    im3 = ax3.imshow(np.real(xy_field_meta).T, cmap='RdBu', extent=extent_xy,
                     aspect='equal', origin='lower', vmin=-vmax_xy, vmax=vmax_xy)
    ax3.contour(eps_xy.T, levels=[3], colors='white', linewidths=1.5,
                extent=extent_xy, origin='lower')
    ax3.set_xlabel('X (um)')
    ax3.set_ylabel('Y (um)')
    ax3.set_title('Ex Field - XY Plane (z=0.3um, through meta-atom)')
    plt.colorbar(im3, ax=ax3, label='Ex')
    
    # Subplot 4: XY cross-section field distribution (above meta-atom)
    ax4 = fig.add_subplot(2, 3, 4)
    vmax_above = np.max(np.abs(xy_field_above)) if np.max(np.abs(xy_field_above)) > 0 else 1
    im4 = ax4.imshow(np.real(xy_field_above).T, cmap='RdBu', extent=extent_xy,
                     aspect='equal', origin='lower', vmin=-vmax_above, vmax=vmax_above)
    ax4.set_xlabel('X (um)')
    ax4.set_ylabel('Y (um)')
    ax4.set_title('Ex Field - XY Plane (z=0.8um, above meta-atom)')
    plt.colorbar(im4, ax=ax4, label='Ex')
    
    # Subplot 5: Permittivity distribution
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(eps_xy.T, cmap='Blues', extent=extent_xy,
                     aspect='equal', origin='lower')
    ax5.set_xlabel('X (um)')
    ax5.set_ylabel('Y (um)')
    ax5.set_title('Dielectric Distribution (z=0.3um)')
    plt.colorbar(im5, ax=ax5, label='epsilon')
    
    # Subplot 6: Simulation results text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    result_text = f"""
    Simulation Parameters:
    -------------------------
    Wavelength: {wavelength_nm} nm
    Cell size: 0.5 x 0.5 x 3.0 um^3
    Resolution: {sim_obj.resolution} pixels/um
    TiO2 height: 0.6 um
    TiO2 epsilon: 6.25
    SiO2 epsilon: 2.25
    
    Results (Normalized):
    -------------------------
    Transmittance: {transmittance:.4f} ({transmittance*100:.2f}%)
    Phase: {phase:.4f} rad
    Phase: {np.degrees(phase):.2f} deg
    
    NURBS Parameters:
    -------------------------
    Control points: 8
    Curve segments: 4
    Points per segment: 25
    """
    ax6.text(0.1, 0.5, result_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Simulation Parameters & Results')
    
    plt.tight_layout()
    output_file = 'test_nurbs_simulation_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Figure saved to: {output_file}")
    
    plt.show()
    
    return transmittance, phase, sim_obj


def main():
    """Main function"""
    # Create test control points
    control_points = create_test_control_points()
    
    # Run simulation and visualize
    transmittance, phase, sim_obj = run_simulation_and_visualize(
        control_points, 
        wavelength_nm=550
    )
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return transmittance, phase


if __name__ == "__main__":
    main()
