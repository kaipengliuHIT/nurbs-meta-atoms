"""
测试NURBS超表面单元仿真脚本
使用Meep进行FDTD仿真，计算相位和透射率，并可视化电场分布
"""
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from nurbs_atoms_data import Simulation

def create_test_control_points():
    """
    创建一个合理的四段NURBS曲线控制点
    
    四段NURBS曲线由8个控制点定义，每段使用3个控制点（相邻段共享端点）
    控制点分布在单元格内，形成一个类似圆角矩形的形状
    
    单元格大小: 0.5μm x 0.5μm
    控制点范围: 约 ±0.18μm (确保结构在单元格内)
    """
    # 定义8个控制点，形成一个略微不对称的形状以产生有趣的相位响应
    # 控制点按逆时针排列
    control_points = np.array([
        (0.16, 0.02),    # 点0: 右侧中部偏上
        (0.14, 0.14),    # 点1: 右上角
        (0.02, 0.16),    # 点2: 上侧中部偏右
        (-0.14, 0.14),   # 点3: 左上角
        (-0.16, -0.02),  # 点4: 左侧中部偏下
        (-0.14, -0.14),  # 点5: 左下角
        (-0.02, -0.16),  # 点6: 下侧中部偏左
        (0.14, -0.14)    # 点7: 右下角
    ])
    
    return control_points


def run_simulation_and_visualize(control_points, wavelength_nm=550):
    """
    运行仿真并可视化结果
    
    参数:
        control_points: (8,2) 数组，NURBS控制点坐标 (单位: μm)
        wavelength_nm: 波长 (单位: nm)
    
    返回:
        transmittance: 透射率
        phase: 相位 (弧度)
    """
    print("=" * 60)
    print("NURBS超表面单元仿真测试")
    print("=" * 60)
    
    # 1. 显示控制点信息
    print("\n1. 控制点坐标 (单位: μm):")
    for i, pt in enumerate(control_points):
        print(f"   点{i}: ({pt[0]:.3f}, {pt[1]:.3f})")
    
    # 2. 创建仿真对象
    print(f"\n2. 创建仿真对象...")
    sim_obj = Simulation(control_points=control_points)
    
    # 显示仿真参数
    print(f"   单元格大小: {sim_obj.cell_size}")
    print(f"   分辨率: {sim_obj.resolution} pixels/μm")
    print(f"   TiO2介电常数: {sim_obj.TiO2_material.epsilon_diag.x}")
    print(f"   波长: {wavelength_nm} nm")
    
    # 3. 可视化NURBS曲线形状
    print(f"\n3. 生成NURBS曲线...")
    nurbs_points = sim_obj.generate_complete_nurbs_curve(control_points)
    print(f"   生成了 {len(nurbs_points)} 个曲线采样点")
    
    # 4. 运行仿真
    print(f"\n4. 运行FDTD仿真...")
    print(f"   首先运行参考仿真（无结构）进行归一化...")
    wavelength_m = wavelength_nm * 1e-9
    transmittance, phase = sim_obj.run_forward(
        wavelength_start=wavelength_m, 
        wavelength_stop=wavelength_m,
        normalize=True  # 启用归一化
    )
    
    print(f"\n5. 仿真结果:")
    print(f"   归一化透射率: {transmittance:.4f} ({transmittance*100:.2f}%)")
    print(f"   相对相位: {phase:.4f} rad ({np.degrees(phase):.2f}°)")
    
    # 6. 获取电场分布数据
    print(f"\n6. 提取电场分布数据...")
    
    # 获取不同截面的电场
    # XZ平面 (y=0) - 侧视图
    xz_field = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0), 
        size=mp.Vector3(0.5, 0, 3.0), 
        component=mp.Ex
    )
    
    # XY平面 (z=0.3μm) - 穿过超原子的截面
    xy_field_meta = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0.3), 
        size=mp.Vector3(0.5, 0.5, 0), 
        component=mp.Ex
    )
    
    # XY平面 (z=0.8μm) - 超原子上方的截面
    xy_field_above = sim_obj.sim.get_array(
        center=mp.Vector3(0, 0, 0.8), 
        size=mp.Vector3(0.5, 0.5, 0), 
        component=mp.Ex
    )
    
    # 获取介电常数分布（用于显示结构）
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
    
    # 7. 创建可视化图表
    print(f"\n7. 生成可视化图表...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 子图1: NURBS曲线和控制点
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
    
    # 子图2: XZ截面电场分布
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
    
    # 子图3: XY截面电场分布（穿过超原子）
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
    
    # 子图4: XY截面电场分布（超原子上方）
    ax4 = fig.add_subplot(2, 3, 4)
    vmax_above = np.max(np.abs(xy_field_above)) if np.max(np.abs(xy_field_above)) > 0 else 1
    im4 = ax4.imshow(np.real(xy_field_above).T, cmap='RdBu', extent=extent_xy,
                     aspect='equal', origin='lower', vmin=-vmax_above, vmax=vmax_above)
    ax4.set_xlabel('X (um)')
    ax4.set_ylabel('Y (um)')
    ax4.set_title('Ex Field - XY Plane (z=0.8um, above meta-atom)')
    plt.colorbar(im4, ax=ax4, label='Ex')
    
    # 子图5: 介电常数分布
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(eps_xy.T, cmap='Blues', extent=extent_xy,
                     aspect='equal', origin='lower')
    ax5.set_xlabel('X (um)')
    ax5.set_ylabel('Y (um)')
    ax5.set_title('Dielectric Distribution (z=0.3um)')
    plt.colorbar(im5, ax=ax5, label='epsilon')
    
    # 子图6: 仿真结果文本
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
    """主函数"""
    # 创建测试控制点
    control_points = create_test_control_points()
    
    # 运行仿真并可视化
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

