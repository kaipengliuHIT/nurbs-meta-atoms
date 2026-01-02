import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
import math
import random
import meep as mp

# 全局缓存参考通量（用于归一化透射率）
_reference_flux_cache = {}


def sort_points_ccw(points):
    """Sort coordinate points counter-clockwise around the origin"""
    def angle_from_positive_x(point):
        x, y = point
        # Calculate polar angle (radians) and convert result to [0, 2π) range
        angle = math.atan2(y, x)
        return angle if angle >= 0 else angle + 2 * math.pi
    
    # Sort by polar angle from small to large (i.e., counter-clockwise)
    return np.array(sorted(points, key=angle_from_positive_x))


def get_reference_flux(freq_center, freq_span, frequency_points, resolution=50, run_time=200):
    """
    获取参考通量（无结构时的入射通量），用于归一化透射率
    使用缓存避免重复计算
    """
    # 创建缓存键
    cache_key = (round(freq_center, 6), round(freq_span, 6), frequency_points, resolution)
    
    if cache_key in _reference_flux_cache:
        return _reference_flux_cache[cache_key]
    
    # 创建参考仿真（只有基底，没有超原子结构）
    cell_size = mp.Vector3(0.5, 0.5, 3.0)
    pml_layers = [mp.PML(0.5, direction=mp.Z)]
    
    # 只有SiO2基底
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, 1.0),
            center=mp.Vector3(0, 0, -0.5),
            material=mp.Medium(epsilon=2.25)
        )
    ]
    
    sources = [
        mp.Source(
            src=mp.GaussianSource(freq_center, fwidth=freq_span),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.8),
            size=mp.Vector3(0.5, 0.5, 0)
        )
    ]
    
    sim_ref = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=resolution,
        dimensions=3
    )
    
    # 添加通量监视器
    tran_region = mp.FluxRegion(
        center=mp.Vector3(0, 0, 0.8),
        size=mp.Vector3(0.4, 0.4, 0)
    )
    tran_mon = sim_ref.add_flux(freq_center, freq_span, frequency_points, tran_region)
    
    # 添加相位参考监视器
    phase_monitor_center = mp.Vector3(0, 0, 0.8)
    dft_ex_ref = sim_ref.add_dft_fields([mp.Ex], freq_center, freq_span, frequency_points,
                                         center=phase_monitor_center, size=mp.Vector3(0, 0, 0))
    
    # 运行参考仿真
    sim_ref.run(until=run_time)
    
    # 获取参考通量和相位
    ref_flux = mp.get_fluxes(tran_mon)
    
    # 获取参考相位
    try:
        ex_dft_ref = sim_ref.get_dft_array(dft_ex_ref, mp.Ex, 0)
        if np.isscalar(ex_dft_ref) or ex_dft_ref.size == 1:
            ref_phase = np.angle(ex_dft_ref)
            ref_ex_complex = complex(ex_dft_ref)
        else:
            ref_phase = np.angle(ex_dft_ref[0]) if len(ex_dft_ref) > 0 else 0.0
            ref_ex_complex = complex(ex_dft_ref[0]) if len(ex_dft_ref) > 0 else 1.0
    except:
        ref_phase = 0.0
        ref_ex_complex = 1.0
    
    # 缓存结果
    result = {
        'flux': ref_flux,
        'phase': ref_phase,
        'ex_complex': ref_ex_complex
    }
    _reference_flux_cache[cache_key] = result
    
    # 清理
    sim_ref.reset_meep()
    
    return result


class Simulation:
    def __init__(self, control_points=np.array([(0.18,0),(0.16,0.16),(0,0.18),(-0.16,0.16),(-0.18,0),(-0.16,-0.16),(0,-0.16),(0.16,-0.16)]), target_phase=0):
        self.control_points = control_points  # (N,2) control point coordinates
        # Meep uses a length unit; we choose 1 unit = 1 μm
        self.a = 1.0  # length unit in μm
        self.knots = np.array([0,0,0,1,2,3,4,5,6,7,7,7])
        self.edge_indices = [[0,1,2],[2,3,4],[4,5,6],[6,7,0]]
        self.setup_simulation()
        self.target_phase = target_phase
        self.pts = np.vstack([self.control_points, self.control_points[0]])
        # Initialize Meep simulation parameters
        self.resolution = 50  # pixels per μm
        # Set simulation domain size (x, y, z) in μm
        self.cell_size = mp.Vector3(0.5, 0.5, 3.0)  # 0.5μm x 0.5μm x 3μm
        # Set PML layers - add PML in z direction, use periodic boundaries for x and y
        self.pml_layers = [mp.PML(0.5, direction=mp.Z)]  # PML thickness 0.5 μm
        self.geometry = []
        self.sources = []
        self.monitors = []
        self.sim = None
        self.wavelength_start = 0.4  # μm
        self.wavelength_stop = 0.7   # μm
        # Define TiO2 material (using dispersion model for accuracy)
        self.TiO2_material = mp.Medium(epsilon=6.25)  # Relative permittivity
        self.phase = 0
        # Generate structure
        self.generate_structure(self.control_points)

    def set_contral_points(self, control_points):
        self.control_points = control_points
        self.pts = np.vstack([self.control_points, self.control_points[0]])

    def set_target_phase(self, target_phase):
        self.target_phase = target_phase
        
    def setup_simulation(self):
        """Basic simulation setup - configure simulation environment"""
        # Set periodic boundary conditions for x and y directions
        self.dimensions = 3  # 3D simulation
        # Periodic boundaries are default in Meep when PML is not set

    def generate_structure(self, points):
        """Generate structure based on control points (nurbs meta atoms)"""
        # Generate complete NURBS curve, simulating original Lumerical script
        nurbs_points = self.generate_complete_nurbs_curve(points)
        
        # Convert points to Meep Vector3 format (control points are in μm already)
        meep_vertices = [mp.Vector3(p[0], p[1], 0) for p in nurbs_points]
        
        # Create polygon structure - TiO2 meta atom
        tio2_height = 0.6  # Height 0.6 μm
        tio2_center = mp.Vector3(0, 0, 0.3)  # z-direction center position 0.3 μm
        
        # To avoid self-intersection, sort vertices (sort by angle to form simple polygon)
        sorted_vertices = self.sort_vertices_ccw(meep_vertices)
        
        # Create TiO2 structure
        tio2_polygon = mp.Prism(
            vertices=sorted_vertices,
            height=tio2_height,
            center=tio2_center,
            material=self.TiO2_material
        )
        
        # Substrate (SiO2) - located below z=0
        substrate = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, 1.0),
            center=mp.Vector3(0, 0, -0.5),  # Center at z=-0.5 μm
            material=mp.Medium(epsilon=2.25)  # SiO2 permittivity is approximately 2.25
        )
        
        self.geometry = [substrate, tio2_polygon]

    def generate_complete_nurbs_curve(self, points):
        """Generate points on complete NURBS curve, simulating original Lumerical script"""
        all_points = []
        
        # Extend control points (repeat first point to close)
        extended_points = np.vstack([points, points[0:1]])
        
        # Based on original code logic, generate 4 curve segments, each defined by 3 consecutive control points
        # Corresponds to 4 edges in original code
        for edge in self.edge_indices:
            # Each segment uses 3 consecutive control points
            segment_control_points = np.array([
                extended_points[edge[0]],
                extended_points[edge[1]], 
                extended_points[edge[2]]
            ])
            
            # Generate sample points on this NURBS curve segment
            segment_points = self.generate_nurbs_segment(segment_control_points, num_points=25)
            all_points.extend(segment_points)
        
        # Remove duplicate points
        unique_points = []
        seen = set()
        for point in all_points:
            point_key = (round(point[0], 6), round(point[1], 6))  # Round to avoid floating point errors
            if point_key not in seen:
                seen.add(point_key)
                unique_points.append(point)
        
        return unique_points

    def generate_nurbs_segment(self, control_points, num_points=25):
        """Generate points on a single NURBS segment using quadratic basis functions"""
        vertices = []
        
        # Quadratic basis function definition (consistent with original Lumerical script)
        def basis_function(i, t):
            if i == 0:  # (1-t)^2
                return (1-t)*(1-t)
            elif i == 1:  # 2*t*(1-t)
                return 2*t*(1-t)
            elif i == 2:  # t^2
                return t*t
            return 0
        
        for j in range(num_points):
            t = j / (num_points - 1) if num_points > 1 else 0  # Parameter t from 0 to 1
            
            x = 0
            y = 0
            for i in range(3):  # 3 control points
                weight = basis_function(i, t)
                x += control_points[i, 0] * weight
                y += control_points[i, 1] * weight
            
            vertices.append([x, y])
        
        return vertices

    def sort_vertices_ccw(self, vertices):
        """Sort vertices counter-clockwise to form a simple polygon"""
        if not vertices:
            return vertices
            
        # Convert to numpy array for processing
        pts = np.array([[v.x, v.y] for v in vertices])  # Already in μm
        
        if len(pts) < 2:
            return vertices
        
        # Calculate centroid
        centroid = np.mean(pts, axis=0)
        
        # Calculate angle of each point relative to centroid
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        # Convert back to Meep Vector3 format
        return [mp.Vector3(pt[0], pt[1], 0) for pt in sorted_pts]

    def run_forward(self, wavelength_start=400e-9, wavelength_stop=700e-9, normalize=True):
        """
        Run forward simulation
        
        参数:
            wavelength_start: 起始波长 (m)
            wavelength_stop: 终止波长 (m)
            normalize: 是否归一化透射率（相对于无结构参考）
        
        返回:
            transmittance: 归一化透射率 (0-1)
            phase: 相对相位 (rad)
        """
        # Convert wavelength units (from meters to micrometers)
        wavelength_start_um = wavelength_start * 1e6  # Convert m to μm
        wavelength_stop_um = wavelength_stop * 1e6    # Convert m to μm
        
        # Set frequency range
        freq_min = 1/wavelength_stop_um  # Frequency = c/lambda (c=1 in natural units)
        freq_max = 1/wavelength_start_um
        freq_center = (freq_min + freq_max) / 2
        freq_span = freq_max - freq_min
        
        # Handle single wavelength case (avoid division by zero)
        if freq_span == 0:
            freq_span = freq_center * 0.1  # Use 10% of center frequency as bandwidth
        
        # Calculate number of frequency points
        frequency_points = max(int((wavelength_stop-wavelength_start)/(1e-9)) + 2, 3)
        
        # 运行时间
        run_time = 200
        
        # 获取参考通量（用于归一化）
        if normalize:
            ref_data = get_reference_flux(freq_center, freq_span, frequency_points, 
                                          self.resolution, run_time)
        
        # Set light source - plane wave incident from below
        self.sources = [
            mp.Source(
                src=mp.GaussianSource(freq_center, fwidth=freq_span),
                component=mp.Ex,  # x-direction polarization
                center=mp.Vector3(0, 0, -0.8),  # Source at z=-0.8 μm
                size=mp.Vector3(0.5, 0.5, 0)  # Source size matches cell
            )
        ]
        
        # Create simulation object
        self.sim = mp.Simulation(
            cell_size=self.cell_size,
            geometry=self.geometry,
            sources=self.sources,
            boundary_layers=self.pml_layers,
            resolution=self.resolution,
            dimensions=3
        )
        
        # Add transmission monitor
        tran_region = mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.8),  # Transmission plane at z=0.8 μm
            size=mp.Vector3(0.4, 0.4, 0)  # Monitor size
        )
        
        # Add phase monitor
        phase_monitor_center = mp.Vector3(0, 0, 0.8)  # Phase monitor at z=0.8 μm
        
        # Add monitors
        tran_mon = self.sim.add_flux(freq_center, freq_span, frequency_points, tran_region)
        
        # To get phase, we need to use DFT monitors to get frequency domain data
        # Add a DFT monitor specifically for phase measurement
        dft_ex = self.sim.add_dft_fields([mp.Ex], freq_center, freq_span, frequency_points, 
                                        center=phase_monitor_center, size=mp.Vector3(0,0,0))
        
        # Run simulation
        self.sim.run(until=run_time)
        
        # Get transmission flux
        raw_flux = mp.get_fluxes(tran_mon)[0] if mp.get_fluxes(tran_mon) else 0.0
        
        # 归一化透射率
        if normalize and ref_data['flux'][0] != 0:
            self.Trans = raw_flux / ref_data['flux'][0]
        else:
            self.Trans = raw_flux
        
        # 确保透射率在合理范围内
        self.Trans = max(0.0, min(1.0, self.Trans))
        
        # Get phase information - using DFT monitor
        try:
            # Get complex value of Ex field from DFT monitor
            ex_dft = self.sim.get_dft_array(dft_ex, mp.Ex, 0)  # Get Ex value at first frequency point
            # Fix: if ex_dft is scalar rather than array, get phase directly
            if np.isscalar(ex_dft) or ex_dft.size == 1:
                raw_phase = np.angle(ex_dft)
                ex_complex = complex(ex_dft)
            else:
                raw_phase = np.angle(ex_dft[0]) if len(ex_dft) > 0 else 0.0
                ex_complex = complex(ex_dft[0]) if len(ex_dft) > 0 else 1.0
            
            # 计算相对相位（相对于参考）
            if normalize:
                # 使用复数除法计算相对相位
                if abs(ref_data['ex_complex']) > 1e-10:
                    relative_complex = ex_complex / ref_data['ex_complex']
                    self.phase = np.angle(relative_complex)
                else:
                    self.phase = raw_phase - ref_data['phase']
            else:
                self.phase = raw_phase
                
        except Exception as e:
            print(f"Error getting phase: {e}")
            # Alternative method: use get_field_point
            try:
                ex_val = self.sim.get_field_point(mp.Ex, phase_monitor_center)
                self.phase = np.angle(ex_val) if ex_val else 0.0
            except:
                self.phase = 0.0  # Default phase
        
        return self.Trans, self.phase
    
    def reset(self):
        """重置仿真对象，释放内存"""
        if self.sim is not None:
            self.sim.reset_meep()
            self.sim = None
