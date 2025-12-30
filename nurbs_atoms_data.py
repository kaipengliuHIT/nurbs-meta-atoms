import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
import math
import random
import meep as mp


def sort_points_ccw(points):
    """按围绕原点的逆时针方向排序坐标点"""
    def angle_from_positive_x(point):
        x, y = point
        # 计算极角（弧度），并将结果转换到[0, 2π)范围
        angle = math.atan2(y, x)
        return angle if angle >= 0 else angle + 2 * math.pi
    
    # 按极角从小到大排序（即逆时针方向）
    return np.array(sorted(points, key=angle_from_positive_x))


class Simulation:
    def __init__(self, control_points = np.array([(0.18,0),(0.16,0.16),(0,0.18),(-0.16,0.16),(-0.18,0),(-0.16,-0.16),(0,-0.16),(0.16,-0.16)]), target_phase = 0):
        self.control_points = control_points  # (N,2) 控制点坐标
        self.um = 1e-6
        self.knots = np.array([0,0,0,1,2,3,4,5,6,7,7,7])
        self.edge_indices = [[0,1,2],[2,3,4],[4,5,6],[6,7,0]]
        self.setup_simulation()
        self.target_phase = target_phase
        self.pts = np.vstack([self.control_points,self.control_points[0]])
        # 初始化Meep仿真参数
        self.resolution = 50  # 提高分辨率以获得更精确的结果
        # 设置仿真域大小 (x, y, z) - 匹配原始代码的尺寸
        self.cell_size = mp.Vector3(0.36e-6, 0.36e-6, 2e-6)  
        # 设置PML层 - 在z方向添加PML，x和y方向使用周期性边界
        self.pml_layers = [mp.PML(0.2e-6, direction=mp.Z)]  # PML厚度
        self.geometry = []
        self.sources = []
        self.monitors = []
        self.sim = None
        self.wavelength_start = 0.4  # μm
        self.wavelength_stop = 0.7   # μm
        # 定义TiO2材料 (使用色散模型更准确)
        self.TiO2_material = mp.Medium(epsilon=6.25)  # 相对介电常数
        self.phase = 0
        # 生成结构
        self.generate_structure(self.control_points)

    def set_contral_points(self,control_points):
        self.control_points = control_points
        self.pts = np.vstack([self.control_points,self.control_points[0]])

    def set_target_phase(self,target_phase):
        self.target_phase = target_phase
        
    def setup_simulation(self):
        """基础仿真设置 - 配置仿真环境"""
        # 设置周期性边界条件用于x和y方向
        self.dimensions = 3  # 3D仿真
        # Meep中周期性边界是默认的，当没有设置PML时

    def generate_structure(self, points):
        """根据控制点生成结构(nurbs meta atoms)"""
        # 生成完整的NURBS曲线，模拟原始代码的Lumerical脚本
        nurbs_points = self.generate_complete_nurbs_curve(points)
        
        # 将点转换为Meep的Vector3格式
        meep_vertices = [mp.Vector3(p[0]*self.um, p[1]*self.um, 0) for p in nurbs_points]
        
        # 创建多边形结构 - TiO2 meta atom
        tio2_height = 0.6 * self.um  # 高度 (0.6 μm)
        tio2_center = mp.Vector3(0, 0, 0.3 * self.um)  # z方向中心位置 (0.3 μm)
        
        # 为避免自相交，对顶点进行排序（按角度排序以形成简单多边形）
        sorted_vertices = self.sort_vertices_ccw(meep_vertices)
        
        # 创建TiO2结构
        tio2_polygon = mp.Prism(
            vertices=sorted_vertices,
            height=tio2_height,
            center=tio2_center,
            material=self.TiO2_material
        )
        
        # 基底 (SiO2) - 厚度10 μm，位于z=-5 μm位置（即z从-10 μm到0 μm）
        substrate = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, 10e-6),
            center=mp.Vector3(0, 0, -5e-6),  # 中心在z=-5 μm
            material=mp.Medium(epsilon=2.25)  # SiO2的介电常数约为2.25
        )
        
        self.geometry = [substrate, tio2_polygon]

    def generate_complete_nurbs_curve(self, points):
        """生成完整的NURBS曲线上的点，模拟原始Lumerical脚本"""
        all_points = []
        
        # 将控制点扩展（重复第一个点以闭合）
        extended_points = np.vstack([points, points[0:1]])
        
        # 根据原始代码的逻辑，生成4段曲线，每段由3个连续的控制点定义
        # 对应原始代码中的4个边
        for edge in self.edge_indices:
            # 每段使用3个连续的控制点
            segment_control_points = np.array([
                extended_points[edge[0]],
                extended_points[edge[1]], 
                extended_points[edge[2]]
            ])
            
            # 生成该段NURBS曲线上的采样点
            segment_points = self.generate_nurbs_segment(segment_control_points, num_points=25)
            all_points.extend(segment_points)
        
        # 去除重复点
        unique_points = []
        seen = set()
        for point in all_points:
            point_key = (round(point[0], 6), round(point[1], 6))  # 四舍五入避免浮点误差
            if point_key not in seen:
                seen.add(point_key)
                unique_points.append(point)
        
        return unique_points

    def generate_nurbs_segment(self, control_points, num_points=25):
        """生成单个NURBS段上的点，使用二次基函数"""
        vertices = []
        
        # 二次基函数定义 (与原始Lumerical脚本一致)
        def basis_function(i, t):
            if i == 0:  # (1-t)^2
                return (1-t)*(1-t)
            elif i == 1:  # 2*t*(1-t)
                return 2*t*(1-t)
            elif i == 2:  # t^2
                return t*t
            return 0
        
        for j in range(num_points):
            t = j / (num_points - 1) if num_points > 1 else 0  # 参数t从0到1
            
            x = 0
            y = 0
            for i in range(3):  # 3个控制点
                weight = basis_function(i, t)
                x += control_points[i, 0] * weight
                y += control_points[i, 1] * weight
            
            vertices.append([x, y])
        
        return vertices

    def sort_vertices_ccw(self, vertices):
        """将顶点按逆时针方向排序以形成简单多边形"""
        if not vertices:
            return vertices
            
        # 转换为numpy数组以便处理
        pts = np.array([[v.x/self.um, v.y/self.um] for v in vertices])  # 转换回无单位坐标
        
        if len(pts) < 2:
            return vertices
        
        # 计算重心
        centroid = np.mean(pts, axis=0)
        
        # 计算每个点相对于重心的角度
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        # 转换回Meep的Vector3格式（带单位）
        return [mp.Vector3(pt[0]*self.um, pt[1]*self.um, 0) for pt in sorted_pts]

    def run_forward(self, wavelength_start=400e-9, wavelength_stop=700e-9):
        """运行正向仿真"""
        # 转换波长单位 (从米到微米)
        wavelength_start_um = wavelength_start / self.um
        wavelength_stop_um = wavelength_stop / self.um
        
        # 设置频率范围
        freq_min = 1/wavelength_stop_um  # 频率 = c/lambda (在自然单位下c=1)
        freq_max = 1/wavelength_start_um
        freq_center = (freq_min + freq_max) / 2
        freq_span = freq_max - freq_min
        
        # 计算频率点数
        frequency_points = int((wavelength_stop-wavelength_start)/(1e-9)) + 2
        
        # 设置光源 - 从下方入射的平面波 (对应原始代码中的source)
        self.sources = [
            mp.Source(
                src=mp.GaussianSource(freq_center, fwidth=freq_span/4),
                component=mp.Ex,  # x方向极化
                center=mp.Vector3(0, 0, -0.5e-6),  # 位置对应原始代码
                size=mp.Vector3(5e-6, 5e-6, 0)  # 光源大小
            )
        ]
        
        # 创建仿真对象
        self.sim = mp.Simulation(
            cell_size=self.cell_size,
            geometry=self.geometry,
            sources=self.sources,
            boundary_layers=self.pml_layers,
            resolution=self.resolution,
            dimensions=3
        )
        
        # 添加透射监测器 (对应原始代码中的"T"监视器)
        tran_region = mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.8e-6),  # 透射面位置 (z=0.8 μm)
            size=mp.Vector3(1.04e-6, 1.04e-6, 0)  # 监测器大小 (1.04 μm × 1.04 μm)
        )
        
        # 添加相位监测器 (对应原始代码中的"phase"监视器)
        phase_monitor_center = mp.Vector3(0, 0, 0.8e-6)  # 相位监测位置
        
        # 添加监测器
        tran_mon = self.sim.add_flux(freq_center, freq_span/4, frequency_points, tran_region)
        
        # 为获取相位，我们需要使用DFT监测器来获取频域数据
        # 添加一个DFT监测器专门用于相位测量
        dft_ex = self.sim.add_dft_fields([mp.Ex], freq_center, freq_span/4, frequency_points, 
                                        center=phase_monitor_center, size=mp.Vector3(0,0,0))
        dft_ey = self.sim.add_dft_fields([mp.Ey], freq_center, freq_span/4, frequency_points, 
                                        center=phase_monitor_center, size=mp.Vector3(0,0,0))
        dft_ez = self.sim.add_dft_fields([mp.Ez], freq_center, freq_span/4, frequency_points, 
                                        center=phase_monitor_center, size=mp.Vector3(0,0,0))
        
        # 运行仿真
        self.sim.run(until=400)  # 增加运行时间以确保收敛
        
        # 获取透射通量
        self.Trans = mp.get_fluxes(tran_mon)[0] if mp.get_fluxes(tran_mon) else 0.0
        
        # 获取相位信息 - 使用DFT监测器
        try:
            # 从DFT监测器获取Ex场的复数值
            ex_dft = self.sim.get_dft_array(dft_ex, mp.Ex, 0)  # 获取第一个频率点的Ex值
            # 修正：如果ex_dft是标量而不是数组，直接获取相位
            if np.isscalar(ex_dft) or ex_dft.size == 1:
                self.phase = np.angle(ex_dft)
            else:
                self.phase = np.angle(ex_dft[0]) if len(ex_dft) > 0 else 0.0
        except Exception as e:
            print(f"获取相位时出错: {e}")
            # 备选方法：使用get_field_point
            try:
                ex_val = self.sim.get_field_point(mp.Ex, phase_monitor_center)
                self.phase = np.angle(ex_val) if ex_val else 0.0
            except:
                self.phase = 0.0  # 默认相位
        
        return self.Trans, self.phase