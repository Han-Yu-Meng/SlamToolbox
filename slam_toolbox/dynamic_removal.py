"""
ERASOR2 + Removert 动态障碍物去除模块
"""

import os
import sys
import subprocess
import tempfile
import textwrap
import shutil
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .extractor import (
    _read_pcd,
    _write_pcd,
    _invert_transform,
    _transform_to_matrix,
    lookup_transform,
    parse_pc2_msg,
    rosbag2_py,
    deserialize_message,
    get_message,
)

# ---------------------------------------------------------------------------
# Docker 镜像（自包含，无需挂载主机文件）
# ---------------------------------------------------------------------------

_ERASOR2_IMAGE = "stevenmhy/slamtoolbox-erasor2:latest"
_REMOVERT_IMAGE = "stevenmhy/slamtoolbox-removert:latest"

# 容器内固定路径
_ERASOR2_BIN_DIR = "/opt/erasor2/bin"
_ERASOR2_SCRIPTS_DIR = "/opt/erasor2/scripts"
_REMOVERT_WS = "/opt/removert_ws"

_PKG_DIR = Path(__file__).resolve().parent  # slam_toolbox/

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


def _timestamped_output_dir(map_path, method_name):
    """Create a timestamped run directory for method outputs."""
    root = os.path.join(map_path, "runs", method_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, stamp)
    suffix = 1
    while os.path.exists(out):
        out = os.path.join(root, f"{stamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(out, exist_ok=True)
    return out


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


def _find_bag_storage(map_path):
    bag_dir = os.path.join(map_path, "bag")
    for root, _, files in os.walk(bag_dir):
        for name in files:
            if name.endswith(".db3"):
                return bag_dir, "sqlite3", os.path.join(root, name)
            if name.endswith(".mcap"):
                return bag_dir, "mcap", os.path.join(root, name)
    raise FileNotFoundError(f"未在 {bag_dir} 下找到 .db3 或 .mcap")


def _collect_bag_metadata(bag_dir, storage_id, pointcloud_topic, config):
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if pointcloud_topic not in type_map:
        raise ValueError(f"bag 中没有点云话题: {pointcloud_topic}")

    tf_type = next((typ for name, typ in type_map.items() if name in ("/tf", "/tf_static")), None)
    tf_msg_cls = get_message(tf_type) if tf_type else None
    dynamic_tf = {}
    static_tf = {}
    total_cloud_msgs = 0

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == pointcloud_topic:
            total_cloud_msgs += 1
        elif tf_msg_cls and topic in ("/tf", "/tf_static"):
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                sec = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                matrix = _transform_to_matrix(transform)
                key = (parent, child)
                if topic == "/tf_static":
                    static_tf[key] = (sec, matrix)
                else:
                    dynamic_tf.setdefault(key, []).append((sec, matrix))

    from .config import build_fixed_transforms

    for key, value in build_fixed_transforms(config).items():
        if key not in static_tf:
            static_tf[key] = value

    for key in dynamic_tf:
        dynamic_tf[key].sort(key=lambda x: x[0])

    if total_cloud_msgs == 0:
        raise ValueError(f"bag 中没有点云消息: {pointcloud_topic}")

    return storage_options, converter_options, type_map, {"dynamic": dynamic_tf, "static": static_tf}, total_cloud_msgs


def _lookup_or_identity(tf_buffer, parent, child, timestamp, warn_set):
    if parent == child:
        return np.eye(4, dtype=np.float64)

    transform = lookup_transform(tf_buffer, parent, child, timestamp)
    if transform is not None:
        return transform

    tag = (parent, child)
    if tag not in warn_set:
        warn_set.add(tag)
        print(f"[yellow]警告: 缺少 TF {parent} -> {child}，使用单位阵代替。[/yellow]")
    return np.eye(4, dtype=np.float64)


def convert_bag_to_kitti(map_path, config):
    """从原始 bag 生成 KITTI 数据集。

    写出的 velodyne/*.bin 必须是 base_link 局部帧；如果输入点云是 /cloud_registered
    这类 odom/global 点云，会先转到 fixed_frame，再用 odom->base_link 的逆变换转回局部帧。
    """
    if rosbag2_py is None:
        raise RuntimeError("无法导入 rosbag2_py。请在 ROS2 环境中运行 ERASOR2 转换。")

    cfg = config["config"]
    fixed_frame = cfg["fixed_frame"]
    base_link_frame = cfg["base_link_frame"]
    pointcloud_topic = cfg["pointcloud_topic"]

    bag_dir, storage_id, db_file = _find_bag_storage(map_path)
    print(f"从 bag 逐帧转换为 KITTI: {db_file}")
    print(
        f"  fixed_frame={fixed_frame}, base_link_frame={base_link_frame}, "
        f"pointcloud_topic={pointcloud_topic}"
    )

    storage_options, converter_options, type_map, tf_buffer, total_cloud_msgs = _collect_bag_metadata(
        bag_dir, storage_id, pointcloud_topic, config
    )

    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    labels_dir = os.path.join(seq_dir, "labels")

    if os.path.isdir(seq_dir):
        shutil.rmtree(seq_dir)
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    true_pose_lines = []
    compensated_pose_lines = []
    time_lines = []
    warn_set = set()
    frame_count = 0
    first_cloud_frame = None
    max_points = 0

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    cloud_msg_type = get_message(type_map[pointcloud_topic])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("从 bag 逐帧转换为 KITTI...", total=total_cloud_msgs)

        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != pointcloud_topic:
                continue

            msg = deserialize_message(data, cloud_msg_type)
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cloud_frame = msg.header.frame_id if msg.header.frame_id else base_link_frame
            if first_cloud_frame is None:
                first_cloud_frame = cloud_frame

            xyz, intensity = parse_pc2_msg(msg)
            if len(xyz) == 0:
                progress.update(task, advance=1)
                continue
            if intensity is None:
                intensity = np.ones(len(xyz), dtype=np.float32)

            T_odom_base = _lookup_or_identity(
                tf_buffer, fixed_frame, base_link_frame, sec, warn_set
            )

            T_cloud_to_fixed = _lookup_or_identity(
                tf_buffer, fixed_frame, cloud_frame, sec, warn_set
            )
            T_fixed_to_base = np.linalg.inv(T_odom_base)
            T_cloud_to_base = T_fixed_to_base @ T_cloud_to_fixed
            if not np.allclose(T_cloud_to_base, np.eye(4)):
                pts_h = np.ones((len(xyz), 4), dtype=np.float64)
                pts_h[:, :3] = xyz
                xyz = (T_cloud_to_base @ pts_h.T).T[:, :3].astype(np.float32)

            stem = f"{frame_count:06d}"
            bin_data = np.column_stack([xyz, intensity]).astype(np.float32)
            bin_data.tofile(os.path.join(velodyne_dir, f"{stem}.bin"))
            np.zeros(len(xyz), dtype=np.uint32).tofile(os.path.join(labels_dir, f"{stem}.label"))

            compensated = TF_ORIGIN_INV @ T_odom_base @ KITTI_CAM2LIDAR_INV
            compensated_pose_lines.append(_mat3x4_line(compensated))
            true_pose_lines.append(_mat3x4_line(T_odom_base))
            time_lines.append(f"{sec:.9f}")
            max_points = max(max_points, len(xyz))
            frame_count += 1
            progress.update(task, advance=1)

    if frame_count == 0:
        raise RuntimeError("没有成功转换任何点云帧。")

    (Path(seq_dir) / "poses_suma_optim.txt").write_text("\n".join(compensated_pose_lines) + "\n")
    (Path(seq_dir) / "poses_odom_base.txt").write_text("\n".join(true_pose_lines) + "\n")
    (Path(seq_dir) / "times.txt").write_text("\n".join(time_lines) + "\n")
    (Path(seq_dir) / "conversion_notes.txt").write_text(
        f"source_bag: {bag_dir}\n"
        f"cloud_topic: {pointcloud_topic}\n"
        f"tf_edge: {fixed_frame} -> {base_link_frame}\n"
        f"cloud_frame_written: {base_link_frame}\n"
        f"source_cloud_frame: {first_cloud_frame}\n"
        f"point_transform: {first_cloud_frame} -> {fixed_frame} -> {base_link_frame}\n"
        f"frames_written: {frame_count}\n"
        f"max_points_per_frame: {max_points}\n"
        "poses_suma_optim.txt is compensated for ERASOR2 SemanticKITTILoader.\n"
        "poses_odom_base.txt contains the true odom -> base_link matrices.\n"
        "labels/*.label are zero placeholders for size compatibility, not ground truth.\n"
    )

    print(f"KITTI 格式转换完成: {frame_count} 帧 → {seq_dir}")
    return kitti_root, frame_count


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
    """通过 Docker 运行 ERASOR2（二进制和脚本均在镜像内）。"""

    image = _ensure_or_pull_image(_ERASOR2_IMAGE)

    docker_cmd = [
        "docker", "run", "--rm",
        "--memory=10g",
        "--cpus=4",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{kitti_root}:{kitti_root}",
        "-v", f"{output_dir}:{output_dir}",
        "-w", _ERASOR2_SCRIPTS_DIR,
        image,
        "bash", "-lc",
        "set -euo pipefail; "
        f"python3 {_ERASOR2_SCRIPTS_DIR}/kitti_clustering.py "
        f"  --kitti_dir {kitti_root} "
        f"  --seq 00 "
        f"  --init_stamp 0 "
        f"  --end_stamp {frame_count - 1} "
        f"  --save-instance-labels "
        f"  --save-ground-labels; "
        f"{_ERASOR2_BIN_DIR}/mapgen {config_path}; "
        f"{_ERASOR2_BIN_DIR}/run_erasor2 {config_path}",
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
    config_path = os.path.join(map_path, "config.yaml")
    try:
        from .config import DEFAULT_CONFIG, load_config
        config = load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG
    except Exception as e:
        print(f"[red]读取 config.yaml 失败: {e}[/red]")
        return

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        print(f"bag 目录 {bag_dir} 不存在。ERASOR2 需要从原始点云话题逐帧转换。")
        return

    print("ERASOR2 将从 bag 的原始点云话题逐帧生成 KITTI 数据集，不再使用 frame/ 聚合 PCD。\n")

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

    # Step 1: 原始 bag 点云逐帧 → KITTI
    print()
    try:
        kitti_root, frame_count = convert_bag_to_kitti(map_path, config)
    except Exception as e:
        print(f"[red]KITTI 转换失败: {e}[/red]")
        return

    # Step 2: 生成配置
    output_dir = _timestamped_output_dir(map_path, "erasor2")
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


# ---------------------------------------------------------------------------
# Removert 动态障碍物去除
# ---------------------------------------------------------------------------

def _load_map_config(map_path):
    config_path = os.path.join(map_path, "config.yaml")
    try:
        from .config import DEFAULT_CONFIG, load_config
        return load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG
    except Exception as e:
        raise RuntimeError(f"读取 config.yaml 失败: {e}") from e


def _has_current_bag_local_transform(seq_dir):
    notes_path = os.path.join(seq_dir, "conversion_notes.txt")
    if not os.path.exists(notes_path):
        return False
    notes = Path(notes_path).read_text(errors="replace")
    return "source_bag:" in notes and "point_transform:" in notes and "cloud_frame_written: base_link" in notes


def _ensure_kitti_dataset(map_path):
    """确保 KITTI 数据集存在：优先复用新版 bag local 数据，否则从 bag 重新转换。"""
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")

    if os.path.isdir(velodyne_dir) and os.listdir(velodyne_dir):
        bin_files = [f for f in os.listdir(velodyne_dir) if f.endswith(".bin")]
        if _has_current_bag_local_transform(seq_dir):
            frame_count = len(bin_files)
            print(f"复用已有 bag local KITTI 数据集: {velodyne_dir} ({frame_count} 帧)")
            return kitti_root, frame_count
        print("检测到旧版或 frame 版 KITTI 数据集，将舍弃并从 bag 重新生成。")

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        raise RuntimeError(f"bag 目录不存在，无法生成 KITTI 数据集: {bag_dir}")

    print("KITTI 数据集不存在或不是新版 bag local 格式，先从 bag 转换...")
    return convert_bag_to_kitti(map_path, _load_map_config(map_path))


def _ensure_local_kitti_dataset(map_path):
    """确保存在适合 local hash voxel / raycasting 的逐帧 local KITTI 数据集。"""
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    pose_path = os.path.join(seq_dir, "poses_odom_base.txt")

    if os.path.isdir(velodyne_dir) and os.path.exists(pose_path):
        bin_files = [f for f in os.listdir(velodyne_dir) if f.endswith(".bin")]
        if bin_files and _has_current_bag_local_transform(seq_dir):
            _require_sensor_trajectory(seq_dir)
            print(f"复用已有逐帧 KITTI 数据集: {velodyne_dir} ({len(bin_files)} 帧)")
            return kitti_root, len(bin_files)
        if bin_files:
            print("检测到旧版、frame 版或未校正的 KITTI 数据集，将舍弃并从 bag 重新生成逐帧 local KITTI。")

    print("逐帧 KITTI 数据集不存在或缺少 poses_odom_base.txt，先从 bag 转换...")
    kitti_root, frame_count = convert_bag_to_kitti(map_path, _load_map_config(map_path))
    _require_sensor_trajectory(seq_dir)
    return kitti_root, frame_count


def _require_sensor_trajectory(seq_dir):
    pose_path = os.path.join(seq_dir, "poses_odom_base.txt")
    identity_path = os.path.join(seq_dir, "poses_identity.txt")

    if not os.path.exists(pose_path):
        if os.path.exists(identity_path):
            raise RuntimeError(
                "当前数据集只有 poses_identity.txt，没有真实传感器轨迹。"
                "local hash voxel 和 raycasting 需要逐帧传感器位姿，YunJingFull 这类数据暂不支持。"
            )
        raise RuntimeError(f"缺少真实传感器轨迹文件: {pose_path}")

    translations = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 12:
                translations.append((vals[3], vals[7], vals[11]))
            elif len(vals) == 16:
                translations.append((vals[3], vals[7], vals[11]))
            else:
                raise RuntimeError(f"不支持的 pose 格式: {pose_path}")

    if len(translations) < 2:
        raise RuntimeError("真实传感器轨迹少于 2 帧，无法进行 local hash voxel/raycasting。")

    arr = np.asarray(translations, dtype=np.float64)
    movement = np.linalg.norm(arr - arr[0], axis=1).max()
    if movement < 1e-3:
        raise RuntimeError(
            "检测到传感器轨迹几乎全为同一位姿，无法进行 local hash voxel/raycasting。"
            "YunJingFull 这类无真实传感器轨迹的数据请先跳过。"
        )


def _run_python_script(script_name, args):
    script_path = _PKG_DIR / "algorithms" / script_name
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, check=False).returncode


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")
_READ_PROGRESS_RE = re.compile(r"\[progress\]\s+readValidScans\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MERGE_PROGRESS_RE = re.compile(r"\[progress\]\s+mergeScansWithinGlobalCoord\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MAP_SIDE_PROGRESS_RE = re.compile(r"\[progress\]\s+map-side scan loop\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MAP2RANGE_PROGRESS_RE = re.compile(r"\[progress\]\s+map2RangeImg\b.*?(\d+)%")
_REMOVE_ITER_RE = re.compile(r"\[progress\]\s+remove iteration\s+(\d+)/(\d+)")
_SAVE_PCD_RE = re.compile(
    r"(removert_(?:after|dynamic)(?:_local)?\.pcd|"
    r"(?:Dynamic|Static)MapMapside(?:Global|Local)ResX[0-9.]+\.pcd)"
)


def _strip_ansi(text):
    return _ANSI_RE.sub("", text)


def _run_removert_with_progress(cmd, log_path, output_dir, frame_count):
    """Run Removert while converting its stable progress log markers to stage bars."""
    saved_files = set()
    recent_errors = []
    remove_has_progress = False

    def _get_task(progress, task_id):
        return next(task for task in progress.tasks if task.id == task_id)

    def _remember_error(line):
        lower = line.lower()
        if any(token in lower for token in ("error", "failed", "exception", "abort", "terminate")):
            recent_errors.append(line.strip())
            del recent_errors[:-8]

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            read_task = progress.add_task("读取点云", total=frame_count)
            merge_task = progress.add_task("构建地图", total=frame_count)
            remove_task = progress.add_task("动态清除", total=frame_count)
            write_task = progress.add_task("写出结果", total=4)

            assert process.stdout is not None
            for raw_line in process.stdout:
                log_file.write(raw_line)
                log_file.flush()

                line = _strip_ansi(raw_line).strip()
                if not line:
                    continue
                _remember_error(line)

                match = _READ_PROGRESS_RE.search(line)
                if match:
                    done, total, _ = match.groups()
                    progress.update(read_task, completed=int(done), total=int(total))
                    continue

                match = _MERGE_PROGRESS_RE.search(line)
                if match:
                    done, total, _ = match.groups()
                    progress.update(merge_task, completed=int(done), total=int(total))
                    continue

                match = _MAP_SIDE_PROGRESS_RE.search(line)
                if match:
                    done, total, _ = match.groups()
                    remove_has_progress = True
                    progress.update(remove_task, completed=int(done), total=int(total))
                    continue

                match = _MAP2RANGE_PROGRESS_RE.search(line)
                if match and not remove_has_progress:
                    pct = max(0, min(100, int(match.group(1))))
                    progress.update(remove_task, completed=round(frame_count * pct / 100))
                    continue

                match = _REMOVE_ITER_RE.search(line)
                if match and not remove_has_progress:
                    done, total = (int(v) for v in match.groups())
                    remove_has_progress = True
                    progress.update(remove_task, completed=done, total=total)
                    continue

                for filename in _SAVE_PCD_RE.findall(line):
                    saved_files.add(filename)
                if saved_files:
                    progress.update(write_task, completed=min(len(saved_files), 4))

            returncode = process.wait()

            for filename in (
                "removert_after.pcd",
                "removert_dynamic.pcd",
                "removert_after_local.pcd",
                "removert_dynamic_local.pcd",
            ):
                if os.path.exists(os.path.join(output_dir, filename)):
                    saved_files.add(filename)
            progress.update(write_task, completed=min(len(saved_files), 4))

            if returncode == 0:
                for task_id in (read_task, merge_task, remove_task):
                    task = _get_task(progress, task_id)
                    if task.total is not None and task.completed < task.total:
                        progress.update(task_id, completed=task.total)

        if returncode != 0 and recent_errors:
            print("[yellow]Removert 关键错误日志:[/yellow]")
            for line in recent_errors:
                print(f"  {line}")

        return returncode


def _copy_cleanup_result(out_dir, result_name, map_path, output_name):
    src = os.path.join(out_dir, result_name)
    if not os.path.exists(src):
        print(f"[yellow]未找到输出文件: {src}[/yellow]")
        return
    map_dir = os.path.join(map_path, "map")
    os.makedirs(map_dir, exist_ok=True)
    dst = os.path.join(map_dir, output_name)
    shutil.copy2(src, dst)
    print(f"  已复制结果: {dst}")


def start_local_hash_voxel(map_path):
    """Local hash voxel 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        print(f"[red]Local Hash Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Local Hash Voxel 动态障碍物清除。\n")

    voxel_size = questionary.text("Hash voxel 尺寸 (米):", default="0.4").ask() or "0.4"
    max_range = questionary.text("最大水平距离 (米):", default="30.0").ask() or "30.0"
    local_z_min = questionary.text("局部 Z 下限 (米):", default="-2.5").ask() or "-2.5"
    local_z_max = questionary.text("局部 Z 上限 (米):", default="3.0").ask() or "3.0"
    ground_z = questionary.text(
        "地面保护局部 Z 上限 (留空关闭，常用 0.0 或 -0.1):",
        default="0.0",
    ).ask()
    unknown_policy = questionary.select(
        "unknown 体素处理策略:",
        choices=["keep", "drop"],
        default="keep",
    ).ask() or "keep"

    output_dir = _timestamped_output_dir(map_path, "local_hash_voxel")

    args = [
        "--dataset", kitti_root,
        "--out", output_dir,
        "--seq", "00",
        "--voxel-size", voxel_size,
        "--max-range", max_range,
        "--local-z-min", local_z_min,
        "--local-z-max", local_z_max,
        "--unknown-policy", unknown_policy,
        "--deduplicate", "quantized",
        "--interleave-after",
    ]
    if ground_z:
        args.extend(["--ground-protect-local-z-max", ground_z])

    ret = _run_python_script("local_hash_voxel_filter.py", args)
    if ret != 0:
        print(f"[yellow]Local Hash Voxel 返回非零退出码: {ret}[/yellow]")
        return

    print("\n[bold green]Local Hash Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    _copy_cleanup_result(output_dir, "local_hash_voxel_after.pcd", map_path, "map_local_hash_voxel_static.pcd")
    _copy_cleanup_result(output_dir, "local_hash_voxel_dynamic.pcd", map_path, "map_local_hash_voxel_dynamic.pcd")


def start_raycast_voxel(map_path):
    """Raycast voxel cleanup 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        print(f"[red]Raycast Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Raycast Voxel Cleanup。\n")

    voxel_size = questionary.text("Raycast voxel 尺寸 (米):", default="0.30").ask() or "0.30"
    max_range = questionary.text("最大水平距离 (米):", default="35.0").ask() or "35.0"
    body_radius = questionary.text("车体半径过滤 (米):", default="0.8").ask() or "0.8"
    ray_stride = questionary.text("Ray point stride:", default="8").ask() or "8"
    ground_z = questionary.text(
        "地面保护局部 Z 上限 (留空关闭，常用 0.0 或 -0.1):",
        default="0.0",
    ).ask()

    output_dir = _timestamped_output_dir(map_path, "raycast_voxel")

    args = [
        "--dataset", kitti_root,
        "--out", output_dir,
        "--seq", "00",
        "--voxel-size", voxel_size,
        "--max-range", max_range,
        "--body-radius", body_radius,
        "--ray-point-stride", ray_stride,
        "--deduplicate", "quantized",
        "--interleave-after",
    ]
    if ground_z:
        args.extend(["--ground-protect-local-z-max", ground_z])

    ret = _run_python_script("raycast_voxel_cleanup.py", args)
    if ret != 0:
        print(f"[yellow]Raycast Voxel 返回非零退出码: {ret}[/yellow]")
        return

    print("\n[bold green]Raycast Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    _copy_cleanup_result(output_dir, "raycast_after.pcd", map_path, "map_raycast_voxel_static.pcd")
    _copy_cleanup_result(output_dir, "raycast_removed.pcd", map_path, "map_raycast_voxel_removed.pcd")


def _ensure_or_pull_image(image, fallback=None):
    """检查 Docker 镜像是否存在，否则拉取。返回实际的 image tag。"""
    local = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if local.returncode == 0:
        print(f"本地已有镜像: {image}")
        return image

    if fallback:
        local2 = subprocess.run(
            ["docker", "image", "inspect", fallback],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if local2.returncode == 0:
            print(f"使用本地镜像: {fallback}")
            return fallback

    print(f"本地未找到镜像，正在从 Docker Hub 拉取 {image}...")
    subprocess.run(["docker", "pull", image], check=True)
    return image


def start_removert(map_path):
    """Removert 动态障碍物去除主流程。"""

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        print(f"bag 目录 {bag_dir} 不存在。Removert 需要从原始点云话题逐帧转换，不再使用 frame/。")
        return

    try:
        kitti_root, frame_count = _ensure_kitti_dataset(map_path)
    except Exception as e:
        print(f"[red]KITTI 转换失败: {e}[/red]")
        return

    # 配置
    scan_dir = os.path.join(kitti_root, "dataset", "sequences", "00", "velodyne")
    pose_path = os.path.join(kitti_root, "dataset", "sequences", "00", "poses_odom_base.txt")
    if not os.path.exists(pose_path):
        pose_path = os.path.join(kitti_root, "dataset", "sequences", "00", "poses_suma_optim.txt")

    print(f"\n检测到 {frame_count} 帧，准备运行 Removert 动态障碍物去除。\n")

    vfov_str = questionary.text("垂直 FOV (度):", default="50").ask()
    hfov_str = questionary.text("水平 FOV (度):", default="360").ask()
    batch_str = questionary.text("批处理大小:", default="150").ask()
    omp_str = questionary.text("OpenMP 核心数:", default="4").ask()

    try:
        vfov = float(vfov_str)
    except ValueError:
        vfov = 50
    try:
        hfov = float(hfov_str)
    except ValueError:
        hfov = 360
    try:
        batch_size = int(batch_str)
    except ValueError:
        batch_size = 150
    try:
        omp_cores = int(omp_str)
    except ValueError:
        omp_cores = 4

    # 输出目录
    output_dir = _timestamped_output_dir(map_path, "removert")

    # 生成配置文件
    params_text = f"""removert:
  isScanFileKITTIFormat: true

  saveMapPCD: true
  saveCleanScansPCD: false
  save_pcd_directory: "{output_dir}"

  sequence_scan_dir: "{scan_dir}"
  sequence_pose_path: "{pose_path}"

  sequence_vfov: {vfov}
  sequence_hfov: {hfov}

  ExtrinsicLiDARtoPoseBase: [1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0]

  use_keyframe_gap: true
  keyframe_gap: 1

  start_idx: 0
  end_idx: {frame_count - 1}

  clean_for_all_scan: false
  batch_size: {batch_size}
  valid_ratio_to_save: 0.75

  remove_resolution_list: [2.5, 2.0, 1.5]
  revert_resolution_list: [1.0, 0.9, 0.8, 0.7]

  downsample_voxel_size: 0.0

  num_nn_points_within: 2
  dist_nn_points_within: 0.1

  num_omp_cores: {omp_cores}

  rimg_color_min: 0.0
  rimg_color_max: 20.0
"""
    params_path = os.path.join(output_dir, "removert_params.yaml")
    Path(params_path).write_text(params_text)
    print(f"配置文件已生成: {params_path}")

    # ---- Docker 运行（workspace 已预编译在镜像内）----
    image = _ensure_or_pull_image(_REMOVERT_IMAGE)

    print("正在 Docker 容器中运行 Removert（可能需要数分钟）...")
    print(f"输出目录: {output_dir}")
    log_path = os.path.join(output_dir, "removert_docker.log")
    print(f"详细日志: {log_path}")

    docker_cmd = [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=4",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{kitti_root}:{kitti_root}:ro",
        "-v", f"{output_dir}:{output_dir}",
        "-w", _REMOVERT_WS,
        image,
        "bash", "-lc",
        "set -euo pipefail; "
        "source /opt/ros/noetic/setup.bash; "
        "source /opt/removert_ws/devel/setup.bash; "
        "roscore >/tmp/roscore.log 2>&1 & "
        "ROSCORE_PID=$!; "
        "trap 'kill $ROSCORE_PID 2>/dev/null' EXIT; "
        "for i in $(seq 1 30); do "
        "  if rosparam list >/dev/null 2>&1; then break; fi; "
        "  sleep 1; "
        "done; "
        f"rosparam load {params_path}; "
        "rosrun removert removert_removert",
    ]

    returncode = _run_removert_with_progress(
        docker_cmd,
        log_path,
        output_dir,
        frame_count,
    )
    if returncode != 0:
        print(f"[yellow]Docker 返回非零退出码: {returncode}，请检查日志: {log_path}[/yellow]")

    # ---- 复制结果 ----
    import glob
    import shutil

    map_dir_local = os.path.join(map_path, "map")
    os.makedirs(map_dir_local, exist_ok=True)

    # Removert outputs:
    #   final maps: removert_after.pcd / _local.pcd, removert_dynamic.pcd / _local.pcd
    #   original maps may also be generated as removert_before.pcd / _local.pcd
    after_pcd = os.path.join(output_dir, "removert_after.pcd")
    after_local_pcd = os.path.join(output_dir, "removert_after_local.pcd")
    before_pcd = os.path.join(output_dir, "removert_before.pcd")
    dynamic_pcd = os.path.join(output_dir, "removert_dynamic.pcd")

    copied = []
    if os.path.exists(after_pcd):
        shutil.copy2(after_pcd, os.path.join(map_dir_local, "map_removert_static.pcd"))
        copied.append("removert_after (全局静态地图)")
    if os.path.exists(after_local_pcd):
        shutil.copy2(after_local_pcd, os.path.join(map_dir_local, "map_removert_static_local.pcd"))
        copied.append("removert_after_local (局部静态地图)")
    if os.path.exists(before_pcd):
        shutil.copy2(before_pcd, os.path.join(map_dir_local, "map_removert_before.pcd"))
        copied.append("removert_before (原始地图)")
    if os.path.exists(dynamic_pcd):
        shutil.copy2(dynamic_pcd, os.path.join(map_dir_local, "map_removert_dynamic.pcd"))
        copied.append("removert_dynamic (动态点云)")

    if copied:
        print(f"\n[bold green]Removert 处理完成！[/bold green]")
        for name in copied:
            print(f"  ✓ {name}")
        print(f"  完整输出目录: {output_dir}/")
    else:
        print(f"\n[yellow]未找到 Removert 输出文件，请检查 Docker 日志。[/yellow]")
        print(f"  输出目录: {output_dir}/")
