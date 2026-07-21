"""
ERASOR2 动态障碍物去除模块

流程:
  1. 将 frame/ 中的 PCD + .odom 转换为 KITTI 格式
  2. 生成 ERASOR2 YAML 配置
  3. 通过 Docker 运行 ERASOR2，输出去除动态障碍物后的静态地图 PCD
"""

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .extractor import _read_pcd, _write_pcd, _invert_transform

# ---------------------------------------------------------------------------
# ERASOR2 SemanticKITTILoader 补偿矩阵（来自上游 convert_ros2bag_to_erasor2_kitti.py）
# ---------------------------------------------------------------------------

TF_ORIGIN = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

KITTI_CAM2LIDAR = np.array(
    [
        [-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
        [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
        [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TF_ORIGIN_INV = np.linalg.inv(TF_ORIGIN)
KITTI_CAM2LIDAR_INV = np.linalg.inv(KITTI_CAM2LIDAR)


def _mat3x4_line(mat):
    """将 4×4 矩阵转为 12 个空格分隔的 float（ERASOR2 3×4 行主序格式）"""
    return " ".join(f"{v:.9f}" for v in mat[:3, :4].reshape(-1))


# ---------------------------------------------------------------------------
# 帧 → KITTI 格式转换
# ---------------------------------------------------------------------------

def convert_frames_to_kitti(map_path):
    """将 frame/ 中的 PCD + .odom 转为 KITTI 格式，输出到 map_path/erasor2_dataset/。

    Returns:
        (kitti_root, frame_count) — kitti_root 是 dataset 根目录路径
    """
    frame_dir = os.path.join(map_path, "frame")
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    labels_dir = os.path.join(seq_dir, "labels")

    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        raise FileNotFoundError(f"frame/ 中没有 .pcd 文件: {frame_dir}")

    true_pose_lines = []
    compensated_pose_lines = []
    time_lines = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("转换为 KITTI 格式...", total=len(files))

        for i, file in enumerate(files):
            stem = f"{i:06d}"
            pcd_path = os.path.join(frame_dir, file)
            odom_path = pcd_path.replace(".pcd", ".odom")

            # 读取点云
            xyz, intensity = _read_pcd(pcd_path)

            # 写入 .bin (float32 x y z intensity)
            bin_data = np.column_stack([xyz, intensity]).astype(np.float32) if intensity is not None else np.column_stack([xyz, np.ones(len(xyz), dtype=np.float32)]).astype(np.float32)
            bin_data.tofile(os.path.join(velodyne_dir, f"{stem}.bin"))

            # 写入 .label (全零)
            np.zeros(len(xyz), dtype=np.uint32).tofile(os.path.join(labels_dir, f"{stem}.label"))

            # 读写姿态
            if os.path.exists(odom_path):
                try:
                    T_odom_base = np.loadtxt(odom_path, dtype=np.float64)
                    if T_odom_base.shape != (4, 4):
                        T_odom_base = np.eye(4, dtype=np.float64)
                except Exception:
                    T_odom_base = np.eye(4, dtype=np.float64)
            else:
                T_odom_base = np.eye(4, dtype=np.float64)

            compensated = TF_ORIGIN_INV @ T_odom_base @ KITTI_CAM2LIDAR_INV
            compensated_pose_lines.append(_mat3x4_line(compensated))
            true_pose_lines.append(_mat3x4_line(T_odom_base))
            time_lines.append(f"{i * 0.1:.9f}")  # 用帧序号估算时间戳

            progress.update(task, advance=1)

    # 写入文本文件
    (Path(seq_dir) / "poses_suma_optim.txt").write_text("\n".join(compensated_pose_lines) + "\n")
    (Path(seq_dir) / "poses_odom_base.txt").write_text("\n".join(true_pose_lines) + "\n")
    (Path(seq_dir) / "times.txt").write_text("\n".join(time_lines) + "\n")
    (Path(seq_dir) / "conversion_notes.txt").write_text(
        f"converted from: {frame_dir}\n"
        f"frames_written: {len(files)}\n"
        "cloud_frame: base_link (extracted frames)\n"
        "poses_suma_optim.txt is compensated for ERASOR2 SemanticKITTILoader.\n"
        "poses_odom_base.txt contains the true odom -> base_link matrices.\n"
        "labels/*.label are zero placeholders for size compatibility.\n"
    )

    print(f"KITTI 格式转换完成: {len(files)} 帧 → {seq_dir}")
    return kitti_root, len(files)


# ---------------------------------------------------------------------------
# ERASOR2 YAML 配置生成
# ---------------------------------------------------------------------------

def generate_erasor2_config(kitti_root, output_dir, frame_count, min_z, max_z):
    """生成 ERASOR2 的 YAML 配置文件。"""
    seq_dir = os.path.join(kitti_root, "dataset", "sequences")

    yaml_content = f"""\
start_frame: 0
end_frame: {frame_count - 1}
viz_interval: 100
is_large_scale: true
num_omp_cores: 4

dataloader:
    run_traj_clustering: false
    dataset_name: "SemanticKITTI"
    abs_data_dir: "{seq_dir}"
    cloud_dir: ""
    cloud_format: ""
    pose_path: ""
    sequence: "00"
    abs_save_dir: "{output_dir}"
    instance_seg_method: "hdbscan"

    accum_interval: 1
    voxel_size: 0.2
    map_voxel_size: 0.2

    expansion_range: 0

erasor2:
    grid_resolution: 1.0
    egocentric_grid_resolution: 0.6
    range_of_interest: 80.0
    min_z_voi: {min_z}
    max_z_voi: {max_z}
    min_z_diff_thr: 0.4
    scan_ratio_threshold: 0.2
    log_odds:
        increment_gain: 2.0
        increment: 0.15
    region_proposal_thr: 0.8
    kernel_size: 1

    ratio_num_pts: 0.95
    minimum_num_pts: 5

    moving_object_detection:
        negative_log_odds: -2.0
        obj_score_soft_thr: 4.6
        obj_score_hard_thr: 14.0
        hard_thr_radius: 10.0

    over_segmentation:
        minimum_area_thr: 56
        ratio_of_unknown_prior: 0.25

    volumetric_outlier_removal:
        window_size: 1
        use_adaptive_voxel_size: true
        vor_cand_score_thr: 4.6
        dist_thr_gain: 1.732

    viz_flag:
        set_scan_and_pose: false
        set_submap: false
        update: false
        detect: false
        over_seg: false

    save_map: true

stop_for_each_frame: false

extrinsic:
    robot_body_size: 2.7
    sensor_height: 1.73
    rotation: [ 1, 0, 0,
                0, 1, 0,
                0, 0, 1 ]
    translation: [ 0.0, 0.0, 0.0 ]

rerun:
    enabled: false
    spawn: false
    save_path: ""
"""

    config_path = os.path.join(output_dir, "erasor2_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    Path(config_path).write_text(yaml_content)
    return config_path


# ---------------------------------------------------------------------------
# Docker 运行
# ---------------------------------------------------------------------------

def run_erasor2_docker(kitti_root, output_dir, config_path, frame_count):
    """通过 Docker 运行 ERASOR2。"""

    erasor2_root = os.path.expanduser("~/ERASOR2_RemoverT_Workspace/ERASOR2")
    if not os.path.isdir(erasor2_root):
        raise RuntimeError(
            f"ERASOR2 源码目录不存在: {erasor2_root}\n"
            "请确认 ~/ERASOR2_RemoverT_Workspace/ERASOR2 已 clone 并编译。"
        )

    # 检查/拉取镜像
    image = "stevenmhy/erasor2:ubuntu22"
    local_check = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if local_check.returncode != 0:
        # 也检查不带 registry 前缀的本地 tag
        local_fallback = subprocess.run(
            ["docker", "image", "inspect", "erasor2:ubuntu22"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if local_fallback.returncode == 0:
            image = "erasor2:ubuntu22"
            print(f"使用本地镜像: {image}")
        else:
            print(f"本地未找到镜像，正在从 Docker Hub 拉取 {image}...")
            subprocess.run(["docker", "pull", image], check=True)
    else:
        print(f"本地已有镜像: {image}")

    # 检查 build_safe 二进制
    mapgen_bin = os.path.join(erasor2_root, "build_safe", "mapgen")
    erasor2_bin = os.path.join(erasor2_root, "build_safe", "run_erasor2")
    kitti_clustering = os.path.join(erasor2_root, "scripts", "kitti_clustering.py")

    for p in [mapgen_bin, erasor2_bin, kitti_clustering]:
        if not os.path.exists(p):
            raise RuntimeError(f"ERASOR2 缺少文件: {p}")

    docker_cmd = [
        "docker", "run", "--rm",
        "--memory=10g",
        "--cpus=4",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{erasor2_root}:{erasor2_root}",
        "-v", f"{kitti_root}:{kitti_root}",
        "-v", f"{output_dir}:{output_dir}",
        "-w", erasor2_root,
        image,
        "bash", "-lc",
        "set -euo pipefail; "
        f"python3 scripts/kitti_clustering.py "
        f"  --kitti_dir {kitti_root} "
        f"  --seq 00 "
        f"  --init_stamp 0 "
        f"  --end_stamp {frame_count - 1} "
        f"  --save-instance-labels "
        f"  --save-ground-labels; "
        f"{erasor2_root}/build_safe/mapgen {config_path}; "
        f"{erasor2_root}/build_safe/run_erasor2 {config_path}",
    ]

    print("正在 Docker 容器中运行 ERASOR2（可能需要数分钟）...")
    print(f"输出目录: {output_dir}")

    result = subprocess.run(docker_cmd, check=False)
    if result.returncode != 0:
        print(f"[yellow]Docker 返回非零退出码: {result.returncode}，请检查上方日志[/yellow]")

    return result.returncode


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def start_erasor2(map_path):
    """ERASOR2 动态障碍物去除主流程。"""

    frame_dir = os.path.join(map_path, "frame")
    if not os.path.isdir(frame_dir):
        print(f"帧目录 {frame_dir} 不存在。")
        run_extractor = questionary.confirm(
            "是否先运行 Frame Extractor 提取点云帧？",
            default=True
        ).ask()
        if run_extractor:
            from .extractor import start_extraction
            start_extraction(map_path)
        else:
            return

        # 再次检查
        if not os.path.isdir(frame_dir):
            print("Frame Extractor 未能生成帧目录，退出。")
            return

    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        print("frame/ 中没有 .pcd 文件。")
        return

    print(f"检测到 {len(files)} 个帧，准备运行 ERASOR2 动态障碍物去除。\n")

    # 用户配置 Z 范围
    min_z_str = questionary.text(
        "请输入 ERASOR2 高度范围下限 Z_min (米):",
        default="-4.5"
    ).ask()
    max_z_str = questionary.text(
        "请输入 ERASOR2 高度范围上限 Z_max (米):",
        default="1.5"
    ).ask()
    try:
        min_z = float(min_z_str)
    except ValueError:
        min_z = -4.5
    try:
        max_z = float(max_z_str)
    except ValueError:
        max_z = 1.5

    # Step 1: 转换帧 → KITTI
    print()
    kitti_root, frame_count = convert_frames_to_kitti(map_path)

    # Step 2: 生成配置
    output_dir = os.path.join(map_path, "erasor2_output")
    config_path = generate_erasor2_config(kitti_root, output_dir, frame_count, min_z, max_z)
    print(f"配置文件已生成: {config_path}")

    # Step 3: 运行 ERASOR2
    print()
    try:
        run_erasor2_docker(kitti_root, output_dir, config_path, frame_count)
    except RuntimeError as e:
        print(f"[red]错误: {e}[/red]")
        return

    # Step 4: 复制静态地图结果
    import glob
    import shutil

    map_dir = os.path.join(map_path, "map")
    os.makedirs(map_dir, exist_ok=True)

    # ERASOR2 输出的三个 PCD:
    #   *_original.pcd   → 原始全量地图（去除前）
    #   *_voxel_*.pcd    → Mapgen 体素化后的地图
    #   *_estimated.pcd  → 静态地图（去除动态障碍物后）★ 这个是最有用的
    output_before = os.path.join(map_dir, "map_erasor2_before.pcd")
    output_after  = os.path.join(map_dir, "map_erasor2_static.pcd")

    before_candidates = sorted(glob.glob(os.path.join(output_dir, "*_original.pcd")))
    after_candidates  = sorted(glob.glob(os.path.join(output_dir, "*_estimated.pcd")))

    if before_candidates:
        shutil.copy2(before_candidates[0], output_before)
    if after_candidates:
        shutil.copy2(after_candidates[0], output_after)

    if after_candidates:
        print(f"\n[bold green]ERASOR2 处理完成！[/bold green]")
        print(f"  原始地图（去除前）: {output_before}")
        print(f"  静态地图（去除后）: {output_after}")
        print(f"  完整输出目录: {output_dir}/")
    elif before_candidates:
        print(f"\n[yellow]ERASOR2 仅生成了原始地图，未找到 estimated 结果。[/yellow]")
        print(f"  原始地图: {output_before}")
        print(f"  输出目录: {output_dir}/")
