#!/usr/bin/env python
"""
并行生成NURBS超表面训练数据
支持MPI并行计算，适用于多核工作站（如128核心）

使用方法:
    # 单机多进程模式（推荐用于工作站）
    mpirun -np 128 python generate_training_data_parallel.py --num_samples 50000 --output_dir ./training_data
    
    # 或者使用Python multiprocessing（不需要MPI）
    python generate_training_data_parallel.py --num_samples 50000 --num_workers 128 --mode multiprocessing
    
    # 测试模式（生成少量样本验证）
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

# 物理约束参数
CELL_SIZE = 0.5  # 单元格大小 (μm)
MIN_RADIUS = 0.05  # 最小半径 (μm)
MAX_RADIUS = 0.22  # 最大半径 (μm)，确保结构在单元格内
MIN_FEATURE_SIZE = 0.03  # 最小特征尺寸 (μm)，对应制造约束


def generate_random_control_points(seed=None):
    """
    生成符合物理约束的随机控制点
    
    物理约束:
    1. 所有点必须在单元格内 (±0.25μm)
    2. 结构必须有最小特征尺寸（可制造性）
    3. 控制点按逆时针排列
    4. 结构应该是凸的或近似凸的（避免自相交）
    5. 结构应该有合理的尺寸（不能太小或太大）
    
    返回:
        control_points: (8, 2) numpy数组
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 方法：使用极坐标生成，确保点按角度排序
    # 8个控制点对应8个角度方向
    base_angles = np.linspace(0, 2*np.pi, 9)[:-1]  # [0, π/4, π/2, ..., 7π/4]
    
    # 添加角度扰动（保持顺序）
    angle_perturbation = np.random.uniform(-np.pi/12, np.pi/12, 8)  # ±15度
    angles = base_angles + angle_perturbation
    
    # 确保角度单调递增
    for i in range(1, 8):
        if angles[i] <= angles[i-1]:
            angles[i] = angles[i-1] + 0.05
    
    # 生成半径（带有变化以产生不同形状）
    # 基础半径
    base_radius = np.random.uniform(MIN_RADIUS + 0.02, MAX_RADIUS - 0.02)
    
    # 添加半径变化（产生椭圆、方形等形状）
    # 使用傅里叶模式产生平滑变化
    n_modes = np.random.randint(1, 4)  # 1-3个傅里叶模式
    radius_variation = np.zeros(8)
    
    for mode in range(1, n_modes + 1):
        amplitude = np.random.uniform(0, 0.05) / mode  # 高阶模式振幅更小
        phase = np.random.uniform(0, 2*np.pi)
        radius_variation += amplitude * np.cos(mode * base_angles + phase)
    
    radii = base_radius + radius_variation
    
    # 确保半径在有效范围内
    radii = np.clip(radii, MIN_RADIUS, MAX_RADIUS)
    
    # 转换为笛卡尔坐标
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # 添加小的随机偏移（增加多样性）
    x += np.random.uniform(-0.01, 0.01, 8)
    y += np.random.uniform(-0.01, 0.01, 8)
    
    # 确保所有点在单元格内
    max_coord = CELL_SIZE / 2 - 0.02  # 留一点边距
    x = np.clip(x, -max_coord, max_coord)
    y = np.clip(y, -max_coord, max_coord)
    
    control_points = np.column_stack([x, y])
    
    # 验证生成的控制点
    if not validate_control_points(control_points):
        # 如果验证失败，使用更保守的参数重新生成
        return generate_conservative_control_points(seed)
    
    return control_points


def generate_conservative_control_points(seed=None):
    """
    生成保守的控制点（用于验证失败时的后备方案）
    """
    if seed is not None:
        np.random.seed(seed + 1000000)
    
    # 使用简单的椭圆形状
    base_angles = np.linspace(0, 2*np.pi, 9)[:-1]
    
    # 椭圆参数
    a = np.random.uniform(0.08, 0.16)  # 长轴
    b = np.random.uniform(0.08, 0.16)  # 短轴
    rotation = np.random.uniform(0, np.pi)  # 旋转角度
    
    # 生成椭圆上的点
    x = a * np.cos(base_angles)
    y = b * np.sin(base_angles)
    
    # 旋转
    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    
    return np.column_stack([x_rot, y_rot])


def validate_control_points(control_points):
    """
    验证控制点是否满足物理约束
    
    返回:
        bool: 是否有效
    """
    # 检查1: 所有点在单元格内
    max_coord = CELL_SIZE / 2
    if np.any(np.abs(control_points) > max_coord):
        return False
    
    # 检查2: 结构不能太小
    centroid = np.mean(control_points, axis=0)
    distances = np.linalg.norm(control_points - centroid, axis=1)
    if np.mean(distances) < MIN_RADIUS:
        return False
    
    # 检查3: 相邻点之间的距离不能太小（最小特征尺寸）
    for i in range(8):
        next_i = (i + 1) % 8
        dist = np.linalg.norm(control_points[i] - control_points[next_i])
        if dist < MIN_FEATURE_SIZE:
            return False
    
    # 检查4: 检查是否有自相交（简化检查：确保点按角度排序）
    angles = np.arctan2(control_points[:, 1] - centroid[1], 
                        control_points[:, 0] - centroid[0])
    # 将角度转换到 [0, 2π) 范围
    angles = np.mod(angles, 2 * np.pi)
    
    # 检查角度是否大致单调（允许一些误差）
    sorted_indices = np.argsort(angles)
    expected_order = np.arange(8)
    
    # 找到起始点（角度最小的点应该是第一个）
    start_idx = sorted_indices[0]
    rotated_expected = np.roll(expected_order, -start_idx)
    
    # 允许一定的顺序偏差
    order_diff = np.abs(sorted_indices - np.roll(rotated_expected, start_idx))
    if np.max(order_diff) > 2:  # 允许最多2个位置的偏差
        return False
    
    return True


def simulate_single_sample(args):
    """
    仿真单个样本（用于multiprocessing）
    
    参数:
        args: (sample_id, control_points, wavelength_nm, output_dir)
    
    返回:
        dict: 仿真结果
    """
    sample_id, control_points, wavelength_nm, output_dir = args
    
    try:
        # 导入meep（在子进程中导入以避免MPI冲突）
        import meep as mp
        from nurbs_atoms_data import Simulation
        
        # 创建仿真对象
        sim = Simulation(control_points=control_points)
        
        # 运行仿真
        wavelength_m = wavelength_nm * 1e-9
        transmittance, phase = sim.run_forward(
            wavelength_start=wavelength_m,
            wavelength_stop=wavelength_m,
            normalize=True
        )
        
        # 重置仿真释放内存
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
    使用Python multiprocessing进行并行计算
    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    
    if num_workers <= 0:
        num_workers = cpu_count()
    
    print(f"=" * 60)
    print(f"NURBS超表面训练数据生成 (Multiprocessing模式)")
    print(f"=" * 60)
    print(f"样本数量: {num_samples}")
    print(f"工作进程数: {num_workers}")
    print(f"波长: {wavelength_nm} nm")
    print(f"输出目录: {output_dir}")
    print(f"测试模式: {test_mode}")
    print(f"=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有控制点
    print("\n生成随机控制点...")
    all_control_points = []
    for i in tqdm(range(num_samples), desc="生成控制点"):
        cp = generate_random_control_points(seed=i)
        all_control_points.append(cp)
    
    # 准备任务参数
    tasks = [
        (i, all_control_points[i], wavelength_nm, output_dir)
        for i in range(num_samples)
    ]
    
    # 运行并行仿真
    print(f"\n开始并行仿真 ({num_workers} 个进程)...")
    start_time = time.time()
    
    results = []
    failed_count = 0
    
    # 使用进程池
    with Pool(processes=num_workers) as pool:
        # 使用imap_unordered获取结果（更高效）
        for result in tqdm(pool.imap_unordered(simulate_single_sample, tasks), 
                          total=num_samples, desc="仿真进度"):
            results.append(result)
            if not result['success']:
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    successful_results = [r for r in results if r['success']]
    
    print(f"\n" + "=" * 60)
    print(f"仿真完成!")
    print(f"=" * 60)
    print(f"总样本数: {num_samples}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {failed_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每样本: {elapsed_time/num_samples:.2f} 秒")
    print(f"吞吐量: {num_samples/elapsed_time:.2f} 样本/秒")
    
    # 保存结果
    save_results(successful_results, output_dir, wavelength_nm)
    
    return successful_results


def run_mpi(num_samples, wavelength_nm, output_dir, test_mode=False):
    """
    使用MPI进行并行计算（适用于集群和大型工作站）
    """
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"=" * 60)
        print(f"NURBS超表面训练数据生成 (MPI模式)")
        print(f"=" * 60)
        print(f"样本数量: {num_samples}")
        print(f"MPI进程数: {size}")
        print(f"波长: {wavelength_nm} nm")
        print(f"输出目录: {output_dir}")
        print(f"测试模式: {test_mode}")
        print(f"=" * 60)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    # 同步所有进程
    comm.Barrier()
    
    # 计算每个进程负责的样本范围
    samples_per_proc = num_samples // size
    remainder = num_samples % size
    
    if rank < remainder:
        local_start = rank * (samples_per_proc + 1)
        local_count = samples_per_proc + 1
    else:
        local_start = rank * samples_per_proc + remainder
        local_count = samples_per_proc
    
    if rank == 0:
        print(f"\n每个进程处理约 {samples_per_proc} 个样本")
    
    # 导入meep
    import meep as mp
    from nurbs_atoms_data import Simulation
    
    # 本地结果
    local_results = []
    
    start_time = time.time()
    
    for i in range(local_count):
        global_idx = local_start + i
        
        # 生成控制点
        control_points = generate_random_control_points(seed=global_idx)
        
        try:
            # 创建仿真
            sim = Simulation(control_points=control_points)
            
            # 运行仿真
            wavelength_m = wavelength_nm * 1e-9
            transmittance, phase = sim.run_forward(
                wavelength_start=wavelength_m,
                wavelength_stop=wavelength_m,
                normalize=True
            )
            
            # 重置仿真
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
        
        # 定期打印进度
        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (local_count - i - 1)
            print(f"进程0进度: {i+1}/{local_count}, 已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
    
    # 收集所有结果到rank 0
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # 合并结果
        merged_results = []
        for proc_results in all_results:
            merged_results.extend(proc_results)
        
        elapsed_time = time.time() - start_time
        
        # 统计
        successful_results = [r for r in merged_results if r['success']]
        failed_count = len(merged_results) - len(successful_results)
        
        print(f"\n" + "=" * 60)
        print(f"仿真完成!")
        print(f"=" * 60)
        print(f"总样本数: {num_samples}")
        print(f"成功: {len(successful_results)}")
        print(f"失败: {failed_count}")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每样本: {elapsed_time/num_samples:.2f} 秒")
        print(f"吞吐量: {num_samples/elapsed_time:.2f} 样本/秒")
        
        # 保存结果
        save_results(successful_results, output_dir, wavelength_nm)
        
        return successful_results
    
    return None


def save_results(results, output_dir, wavelength_nm):
    """
    保存仿真结果
    """
    if not results:
        print("警告: 没有成功的结果可保存")
        return
    
    # 按sample_id排序
    results = sorted(results, key=lambda x: x['sample_id'])
    
    # 提取数据
    control_points = np.array([r['control_points'] for r in results])
    transmittances = np.array([r['transmittance'] for r in results])
    phases = np.array([r['phase'] for r in results])
    sample_ids = np.array([r['sample_id'] for r in results])
    
    # 保存为numpy格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    np.save(os.path.join(output_dir, f'control_points_{timestamp}.npy'), control_points)
    np.save(os.path.join(output_dir, f'transmittances_{timestamp}.npy'), transmittances)
    np.save(os.path.join(output_dir, f'phases_{timestamp}.npy'), phases)
    np.save(os.path.join(output_dir, f'sample_ids_{timestamp}.npy'), sample_ids)
    
    # 保存合并的数据文件
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
    
    # 保存元数据
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
    
    print(f"\n数据已保存到: {output_dir}")
    print(f"  - control_points_{timestamp}.npy: 控制点数据 {control_points.shape}")
    print(f"  - transmittances_{timestamp}.npy: 透射率数据 {transmittances.shape}")
    print(f"  - phases_{timestamp}.npy: 相位数据 {phases.shape}")
    print(f"  - training_data_{timestamp}.npz: 合并数据文件")
    print(f"  - metadata_{timestamp}.json: 元数据")
    
    # 打印统计信息
    print(f"\n数据统计:")
    print(f"  透射率: {np.mean(transmittances):.4f} ± {np.std(transmittances):.4f} "
          f"(范围: {np.min(transmittances):.4f} - {np.max(transmittances):.4f})")
    print(f"  相位: {np.mean(phases):.4f} ± {np.std(phases):.4f} rad "
          f"(范围: {np.min(phases):.4f} - {np.max(phases):.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='并行生成NURBS超表面训练数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用multiprocessing模式（推荐用于单机）
  python generate_training_data_parallel.py --num_samples 50000 --num_workers 128 --mode multiprocessing
  
  # 使用MPI模式（需要mpi4py）
  mpirun -np 128 python generate_training_data_parallel.py --num_samples 50000 --mode mpi
  
  # 测试模式
  python generate_training_data_parallel.py --num_samples 100 --num_workers 4 --test
        """
    )
    
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='要生成的样本数量 (默认: 50000)')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='工作进程数，-1表示使用所有CPU核心 (默认: -1)')
    parser.add_argument('--wavelength', type=float, default=550,
                        help='仿真波长 (nm) (默认: 550)')
    parser.add_argument('--output_dir', type=str, default='./training_data',
                        help='输出目录 (默认: ./training_data)')
    parser.add_argument('--mode', type=str, choices=['multiprocessing', 'mpi', 'auto'],
                        default='auto',
                        help='并行模式: multiprocessing, mpi, 或 auto (默认: auto)')
    parser.add_argument('--test', action='store_true',
                        help='测试模式，使用较少样本验证')
    
    args = parser.parse_args()
    
    # 测试模式下减少样本数
    if args.test:
        args.num_samples = min(args.num_samples, 100)
        print("测试模式: 样本数限制为", args.num_samples)
    
    # 自动选择模式
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
    
    # 运行
    if args.mode == 'mpi':
        run_mpi(args.num_samples, args.wavelength, args.output_dir, args.test)
    else:
        run_multiprocessing(args.num_samples, args.num_workers, args.wavelength, 
                           args.output_dir, args.test)


if __name__ == '__main__':
    main()

