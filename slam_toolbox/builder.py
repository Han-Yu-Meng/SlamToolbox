import os
import shutil
from datetime import datetime
import numpy as np
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console

from .extractor import _read_pcd, _write_pcd

console = Console()

BATCH_SIZE = 15  # 每批帧数，控制内存峰值


def _timestamped_output_dir(map_path, method_name):
    root = os.path.join(map_path, "runs", method_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, stamp)
    suffix = 1
    while os.path.exists(out):
        out = os.path.join(root, f"{stamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(out, exist_ok=True)
    return out


def _voxel_downsample(xyz, voxel_size, intensity=None):
    """体素下采样，对 xyz 和 intensity 做均值聚合。"""

    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    # 用 structured array 做去重，无坐标范围限制
    dtype = np.dtype([('i', np.int64), ('j', np.int64), ('k', np.int64)])
    structured = np.empty(len(xyz), dtype=dtype)
    structured['i'] = voxel_indices[:, 0]
    structured['j'] = voxel_indices[:, 1]
    structured['k'] = voxel_indices[:, 2]

    _, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    unique_count = counts.size

    # 平均 xyz
    sum_xyz = np.zeros((unique_count, 3), dtype=np.float64)
    np.add.at(sum_xyz[:, 0], inverse, xyz[:, 0].astype(np.float64))
    np.add.at(sum_xyz[:, 1], inverse, xyz[:, 1].astype(np.float64))
    np.add.at(sum_xyz[:, 2], inverse, xyz[:, 2].astype(np.float64))
    avg_xyz = (sum_xyz / counts[:, None]).astype(np.float32)

    if intensity is not None:
        sum_intensity = np.zeros(unique_count, dtype=np.float64)
        np.add.at(sum_intensity, inverse, intensity.astype(np.float64))
        avg_intensity = (sum_intensity / counts).astype(np.float32)
    else:
        avg_intensity = None

    return avg_xyz, avg_intensity


def start_building(map_path):
    frame_dir = os.path.join(map_path, "frame")
    map_dir = os.path.join(map_path, "map")
    os.makedirs(map_dir, exist_ok=True)
    output_dir = _timestamped_output_dir(map_path, "map_builder")

    if not os.path.exists(frame_dir):
        console.print(f"[red]帧目录 {frame_dir} 不存在。请先运行 Frame Extractor 功能。[/red]")
        return

    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        console.print("[red]未在帧目录中找到 .pcd 文件。[/red]")
        return

    voxel_str = questionary.text("请输入体素下采样大小 (米):", default="0.05").ask()
    try:
        voxel_size = float(voxel_str)
    except ValueError:
        voxel_size = 0.05

    # 检查是否有 intensity 数据
    sample_xyz, sample_i = _read_pcd(os.path.join(frame_dir, files[0]))
    has_intensity = sample_i is not None
    console.print(f"正在分批建图（每 {BATCH_SIZE} 帧体素下采样, "
          f"voxel={voxel_size}m, intensity={'✓' if has_intensity else '✗'}）")

    # 累积器（处理过下采样的中间结果）
    acc_xyz = None           # (M, 3)
    acc_intensity = None     # (M,) 或 None
    total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("处理点云帧...", total=len(files))

        batch_xyz_list = []
        batch_intensity_list = []

        for i, file in enumerate(files):
            pcd_path = os.path.join(frame_dir, file)
            odom_path = pcd_path.replace(".pcd", ".odom")

            xyz, intensity = _read_pcd(pcd_path)

            if xyz is None or len(xyz) == 0:
                progress.update(task, advance=1)
                continue

            # 应用 odom 位姿
            if os.path.exists(odom_path):
                try:
                    pose = np.loadtxt(odom_path)
                    if pose.shape == (4, 4):
                        n = len(xyz)
                        pts_h = np.ones((n, 4), dtype=np.float64)
                        pts_h[:, :3] = xyz
                        xyz = (pose @ pts_h.T).T[:, :3].astype(np.float32)
                except Exception:
                    pass

            batch_xyz_list.append(xyz)
            if intensity is not None:
                batch_intensity_list.append(intensity)

            # 批次满：下采样后合并到累积器
            if (i + 1) % BATCH_SIZE == 0:
                batch_xyz = np.vstack(batch_xyz_list)
                batch_i = (np.hstack(batch_intensity_list)
                           if batch_intensity_list else None)

                ds_xyz, ds_i = _voxel_downsample(batch_xyz, voxel_size, batch_i)

                if acc_xyz is None:
                    acc_xyz, acc_intensity = ds_xyz, ds_i
                else:
                    # 合并到累积器再下采样，消除批次间重叠
                    acc_xyz = np.vstack([acc_xyz, ds_xyz])
                    acc_intensity = (np.hstack([acc_intensity, ds_i])
                                     if acc_intensity is not None and ds_i is not None
                                     else None)
                    acc_xyz, acc_intensity = _voxel_downsample(
                        acc_xyz, voxel_size, acc_intensity)

                batch_xyz_list = []
                batch_intensity_list = []

                batch_num = (i + 1) // BATCH_SIZE
                progress.update(task, advance=BATCH_SIZE,
                                description=f"处理点云帧... (批次 {batch_num}/{total_batches})")

        # 处理剩余不足一个批次的帧
        remaining = len(files) % BATCH_SIZE
        if batch_xyz_list:
            batch_xyz = np.vstack(batch_xyz_list)
            batch_i = (np.hstack(batch_intensity_list)
                       if batch_intensity_list else None)
            ds_xyz, ds_i = _voxel_downsample(batch_xyz, voxel_size, batch_i)

            if acc_xyz is None:
                acc_xyz, acc_intensity = ds_xyz, ds_i
            else:
                acc_xyz = np.vstack([acc_xyz, ds_xyz])
                if acc_intensity is not None and ds_i is not None:
                    acc_intensity = np.hstack([acc_intensity, ds_i])
                acc_xyz, acc_intensity = _voxel_downsample(
                    acc_xyz, voxel_size, acc_intensity)

            progress.update(task, advance=remaining)

    # 最终全局下采样
    console.print("正在最终全局去重...")
    final_xyz, final_intensity = _voxel_downsample(acc_xyz, voxel_size, acc_intensity)

    output_pcd_path = os.path.join(output_dir, "map.pcd")
    _write_pcd(output_pcd_path, final_xyz, final_intensity)
    latest_pcd_path = os.path.join(map_dir, "map.pcd")
    shutil.copy2(output_pcd_path, latest_pcd_path)
    console.print(f"全局地图拼接完成 → {output_pcd_path}（共 {len(final_xyz):,} 点）")
    console.print(f"最新地图快捷路径 → {latest_pcd_path}")

    # ---- 生成可视化 ----
    _generate_visualizations(output_dir, frame_dir, files, final_xyz)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _generate_visualizations(output_dir, frame_dir, files, map_xyz):
    """生成鸟瞰图（Z 轴染色）及轨迹叠加图。"""
    viz_dir = os.path.join(output_dir, "visualize")
    os.makedirs(viz_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]⚠ matplotlib 未安装，跳过可视化生成。[/yellow]")
        return

    # ---- 提取轨迹 ----
    traj_x, traj_y = _extract_trajectory(frame_dir, files)

    # ---- 对地图点云随机降采样（避免图像过大） ----
    viz_max_pts = 500_000
    if len(map_xyz) > viz_max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(map_xyz), viz_max_pts, replace=False)
        viz_xyz = map_xyz[idx]
    else:
        viz_xyz = map_xyz

    x, y, z = viz_xyz[:, 0], viz_xyz[:, 1], viz_xyz[:, 2]

    # ---- 计算 Z 轴颜色范围 ----
    z_min, z_max = np.percentile(z, [1, 99])

    # ================================================================
    # 图 1: 鸟瞰图（按 Z 轴染色）
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("black")

    scatter = ax.scatter(x, y, c=z, s=0.3, cmap="viridis",
                         vmin=z_min, vmax=z_max, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Z (m)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Map Bird's-Eye View (colored by Z)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="white")
    ax.tick_params(colors="white")

    fig.tight_layout()
    out_bev = os.path.join(viz_dir, "birdview_z.png")
    fig.savefig(out_bev, dpi=200, facecolor="black")
    plt.close(fig)
    console.print(f"[green]✓ 鸟瞰图 (Z 染色) → {out_bev}[/green]")

    # ================================================================
    # 图 2: 鸟瞰图 + 轨迹叠加
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("black")

    # 地图底色（灰色，弱对比）
    ax.scatter(x, y, c="#333333", s=0.2, rasterized=True, alpha=0.6)

    # 轨迹（橙色）
    if len(traj_x) > 0:
        ax.plot(traj_x, traj_y, color="#FF8C00", linewidth=0.8, alpha=0.9,
                label=f"Trajectory ({len(traj_x)} frames)")
        # 起点标记（绿色圆点）
        ax.scatter(traj_x[0], traj_y[0], c="#00FF00", s=40, marker="o",
                   edgecolors="white", linewidths=0.5, zorder=5, label="Start")
        # 终点标记（红色叉）
        ax.scatter(traj_x[-1], traj_y[-1], c="#FF0000", s=50, marker="x",
                   linewidths=1.5, zorder=5, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Map + Trajectory Overlay")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", framealpha=0.3, facecolor="black",
              edgecolor="white", labelcolor="white")

    fig.tight_layout()
    out_traj = os.path.join(viz_dir, "birdview_trajectory.png")
    fig.savefig(out_traj, dpi=200, facecolor="black")
    plt.close(fig)
    console.print(f"[green]✓ 鸟瞰图 + 轨迹 → {out_traj}[/green]")


def _extract_trajectory(frame_dir, files):
    """从 .odom 文件中提取轨迹 (x, y) 坐标序列。"""
    traj_x, traj_y = [], []
    for f in files:
        odom_path = os.path.join(frame_dir, f.replace(".pcd", ".odom"))
        if not os.path.exists(odom_path):
            continue
        try:
            T = np.loadtxt(odom_path)
            if T.shape == (4, 4):
                traj_x.append(T[0, 3])
                traj_y.append(T[1, 3])
        except Exception:
            pass
    return np.array(traj_x), np.array(traj_y)
