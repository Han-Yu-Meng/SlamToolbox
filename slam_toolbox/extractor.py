import os
import struct
import re
import warnings
import questionary
import numpy as np
import open3d as o3d
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rosbag2_py = None
    deserialize_message = None
    get_message = None


# ---------------------------------------------------------------------------
# TF 工具
# ---------------------------------------------------------------------------

def _quaternion_to_rotation_matrix(qx, qy, qz, qw):
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[0, 1] = 2.0 * (qx * qy - qz * qw)
    R[0, 2] = 2.0 * (qx * qz + qy * qw)
    R[1, 0] = 2.0 * (qx * qy + qz * qw)
    R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[1, 2] = 2.0 * (qy * qz - qx * qw)
    R[2, 0] = 2.0 * (qx * qz - qy * qw)
    R[2, 1] = 2.0 * (qy * qz + qx * qw)
    R[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return R


def _transform_to_matrix(transform_msg):
    t = transform_msg.transform.translation
    r = transform_msg.transform.rotation
    R = _quaternion_to_rotation_matrix(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def _invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def _find_transform_at(entries, timestamp):
    if not entries:
        return None
    lo, hi = 0, len(entries) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= timestamp:
            best = entries[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def lookup_transform(tf_buffer, parent, child, timestamp):
    key = (parent, child)
    if key in tf_buffer['dynamic']:
        result = _find_transform_at(tf_buffer['dynamic'][key], timestamp)
        if result is not None:
            return result
    if key in tf_buffer['static']:
        return tf_buffer['static'][key][1]
    # 尝试反向查找
    rev_key = (child, parent)
    if rev_key in tf_buffer['dynamic']:
        result = _find_transform_at(tf_buffer['dynamic'][rev_key], timestamp)
        if result is not None:
            return _invert_transform(result)
    if rev_key in tf_buffer['static']:
        return _invert_transform(tf_buffer['static'][rev_key][1])
    return None


def _lookup_or_eye(tf_buffer, parent, child, timestamp, warn_set, warn_tag):
    """查找 TF 变换；若缺失则发出一次性警告并返回单位阵。"""
    T = lookup_transform(tf_buffer, parent, child, timestamp)
    if T is not None:
        return T

    # 发出一次性警告
    if warn_tag not in warn_set:
        warn_set.add(warn_tag)
        warnings.warn(
            f"[TF 链断裂] 无法找到 {parent} → {child} 的 TF 变换，"
            f"已使用单位阵代替。请检查 bag 中是否录制了 /tf 与 /tf_static，"
            f"或确认 config.yaml 中 fixed_frame / base_link_frame 是否正确。"
        )
    return np.eye(4)


# ---------------------------------------------------------------------------
# 坐标变换核心（使用 config 中的 frame 名称）
# ---------------------------------------------------------------------------

def _compute_cloud_to_baselink_ref(
    tf_buffer, cloud_frame, msg_sec, window_start,
    fixed_frame, base_link_frame, warn_set,
):
    """
    计算点云从 cloud_frame(t) → base_link_frame(t0) 的累积变换。

    变换链:
      cloud_frame(t) → base_link_frame(t) → fixed_frame → base_link_frame(t0)

    其中 fixed_frame 是世界坐标系（如 "odom"），
    base_link_frame 是机器人底盘坐标系，
    fixed_frame → base_link_frame 表示机器人在世界中的位姿。
    """
    # Step 1: cloud_frame → base_link_frame (at time t)
    if cloud_frame == base_link_frame:
        T_cloud_to_baselink = np.eye(4)
    elif cloud_frame == fixed_frame:
        # cloud 已经在世界系，需要 base_link(t) → fixed_frame 的逆
        tag = f"tf_miss:{fixed_frame}->{base_link_frame}@t"
        T_fixed_to_base_t = _lookup_or_eye(
            tf_buffer, fixed_frame, base_link_frame, msg_sec,
            warn_set, tag,
        )
        T_cloud_to_baselink = _invert_transform(T_fixed_to_base_t)
    else:
        # 一般情况：先尝试直接查找 base_link_frame → cloud_frame
        T_cloud_to_baselink = lookup_transform(
            tf_buffer, base_link_frame, cloud_frame, msg_sec,
        )

        if T_cloud_to_baselink is None:
            # 直接查找失败 → 尝试通过 fixed_frame 做多跳组合:
            #   cloud_frame → fixed_frame → base_link_frame
            T_fixed_to_cloud = lookup_transform(
                tf_buffer, fixed_frame, cloud_frame, msg_sec,
            )
            T_fixed_to_base = lookup_transform(
                tf_buffer, fixed_frame, base_link_frame, msg_sec,
            )

            if T_fixed_to_cloud is not None and T_fixed_to_base is not None:
                # 拼合: cloud→fixed 再 inv(fixed→base) = cloud→base_link
                T_cloud_to_baselink = (
                    _invert_transform(T_fixed_to_base) @ T_fixed_to_cloud
                )
            else:
                # 所有路径都失败 → 警告并使用单位阵
                tag = f"tf_miss:{base_link_frame}->{cloud_frame}"
                T_cloud_to_baselink = _lookup_or_eye(
                    tf_buffer, base_link_frame, cloud_frame, msg_sec,
                    warn_set, tag,
                )

    # Step 2: base_link_frame(t) → fixed_frame (机器人当前位姿)
    tag_t = f"tf_miss:{fixed_frame}->{base_link_frame}@t"
    T_fixed_to_base_t = _lookup_or_eye(
        tf_buffer, fixed_frame, base_link_frame, msg_sec,
        warn_set, tag_t,
    )

    # Step 3: fixed_frame → base_link_frame(t0) (参考位姿)
    tag_ref = f"tf_miss:{fixed_frame}->{base_link_frame}@ref"
    T_fixed_to_base_ref = _lookup_or_eye(
        tf_buffer, fixed_frame, base_link_frame, window_start,
        warn_set, tag_ref,
    )

    # base_link_frame(t) → base_link_frame(t0)
    T_baselink_t_to_t0 = _invert_transform(T_fixed_to_base_ref) @ T_fixed_to_base_t

    # 完整变换: cloud_frame(t) → base_link_frame(t0)
    T_rel = T_baselink_t_to_t0 @ T_cloud_to_baselink

    return T_rel, T_fixed_to_base_ref


# ---------------------------------------------------------------------------
# 自定义 PCD 读写（保留 intensity 字段）
# ---------------------------------------------------------------------------

def _read_pcd(path):
    """读取 PCD 文件 → (xyz_float32, intensity_float32 或 None)"""
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if line.strip() == b'DATA binary':
                break
        raw = f.read()

    header_str = header.decode('ascii', errors='replace')
    fields = re.search(r'FIELDS\s+(.+)', header_str).group(1).split()
    points  = int(re.search(r'POINTS\s+(\d+)', header_str).group(1))

    data = np.frombuffer(raw, dtype=np.float32).reshape(points, len(fields))

    xi, yi, zi = fields.index('x'), fields.index('y'), fields.index('z')
    xyz = np.ascontiguousarray(data[:, [xi, yi, zi]])

    if 'intensity' in fields:
        ii = fields.index('intensity')
        intensity = np.ascontiguousarray(data[:, ii])
    else:
        intensity = None

    return xyz, intensity


def _write_pcd(path, xyz, intensity=None):
    """写入带 intensity 字段的 binary PCD 文件"""
    n = len(xyz)
    if intensity is not None:
        data = np.column_stack([xyz, intensity]).astype(np.float32)
        fields_line = "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
    else:
        data = xyz.astype(np.float32)
        fields_line = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
{fields_line}WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def start_extraction(map_path, config=None):
    """从 rosbag 中提取点云帧，按时间窗口累积并保存为 PCD。

    参数:
        map_path: 地图目录路径
        config:   配置字典，为 None 时使用默认值
    """
    if rosbag2_py is None:
        console.print("[red]错误: 无法导入 rosbag2_py。请确保是在激活的 ROS2 终端中运行。[/red]")
        return

    if config is None:
        config = {
            "config": {
                "fixed_frame": "odom",
                "base_link_frame": "base_link",
                "pointcloud_topic": "/cloud_registered",
            }
        }

    cfg = config["config"]
    fixed_frame = cfg["fixed_frame"]
    base_link_frame = cfg["base_link_frame"]
    pointcloud_topic = cfg["pointcloud_topic"]

    console.print(
        f"[dim]提取配置: fixed_frame={fixed_frame}, "
        f"base_link_frame={base_link_frame}, "
        f"pointcloud_topic={pointcloud_topic}[/dim]"
    )

    bag_dir = os.path.join(map_path, "bag")
    frame_dir = os.path.join(map_path, "frame")
    os.makedirs(frame_dir, exist_ok=True)

    db_file = ""
    for root, dirs, files in os.walk(bag_dir):
        for f in files:
            if f.endswith('.db3') or f.endswith('.mcap'):
                db_file = os.path.join(root, f)
                break

    if not db_file:
        console.print(f"[red]未在 {bag_dir} 下找到 .db3 或 .mcap。[/red]")
        return

    interval_str = questionary.text(
        "请输入点云累计保存时长间隔 (秒):", default="1.0"
    ).ask()
    try:
        interval = float(interval_str)
    except ValueError:
        interval = 1.0

    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    # 从 config 构建固定变换（补充 bag 中的 TF）
    from .config import build_fixed_transforms
    config_static_tf = build_fixed_transforms(config)

    # ====== 第一遍：统计 cloud + 收集 TF ======
    console.print(f"正在扫描 {db_file} …")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    tf_type = None
    for name, typ in type_map.items():
        if name in ('/tf', '/tf_static'):
            tf_type = typ
            break

    total_cloud_msgs = 0
    tf_msg_cls = None
    dynamic_tf = {}
    static_tf = {}
    if tf_type:
        tf_msg_cls = get_message(tf_type)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == pointcloud_topic:
            total_cloud_msgs += 1
        elif tf_msg_cls and topic in ('/tf', '/tf_static'):
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                sec = (transform.header.stamp.sec +
                       transform.header.stamp.nanosec * 1e-9)
                matrix = _transform_to_matrix(transform)
                key = (parent, child)
                if topic == '/tf_static':
                    static_tf[key] = (sec, matrix)
                else:
                    dynamic_tf.setdefault(key, []).append((sec, matrix))

    # 合并 config 中的 fixed_transform（优先级低于 bag 中的 /tf_static）
    for key, (sec, matrix) in config_static_tf.items():
        if key not in static_tf:
            static_tf[key] = (sec, matrix)

    for key in dynamic_tf:
        dynamic_tf[key].sort(key=lambda x: x[0])

    tf_buffer = {'dynamic': dynamic_tf, 'static': static_tf}

    if total_cloud_msgs == 0:
        console.print(
            f"[red]未在 bag 中找到话题 '{pointcloud_topic}' 的消息，退出。"
            f"请检查 config.yaml 中 pointcloud_topic 是否正确。[/red]"
        )
        return

    tf_frames = set()
    for (p, c) in list(dynamic_tf.keys()) + list(static_tf.keys()):
        tf_frames.add(p)
        tf_frames.add(c)
    console.print(f"  {pointcloud_topic} 消息数: {total_cloud_msgs}")
    console.print(f"  TF 帧: {sorted(tf_frames) if tf_frames else '(无)'}")

    # 预先检查 TF 链完整性
    _check_tf_chain(tf_buffer, fixed_frame, base_link_frame, pointcloud_topic)

    # ====== 第二遍：按时间窗口累积 ======
    reader.open(storage_options, converter_options)

    frame_idx = 0
    window_start = -1.0
    accumulated_points = []
    accumulated_intensities = []
    reference_pose = None
    cloud_msg_type = None
    first_cloud_frame = None
    has_intensity = False

    # TF 断链一次性警告集合
    warn_set = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("累积提取点云帧...", total=total_cloud_msgs)

        msg_idx = 0
        while reader.has_next():
            topic, data, t = reader.read_next()

            if topic == pointcloud_topic:
                if cloud_msg_type is None:
                    cloud_msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, cloud_msg_type)
                sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                cloud_frame = msg.header.frame_id if msg.header.frame_id else base_link_frame

                if first_cloud_frame is None:
                    first_cloud_frame = cloud_frame

                if window_start < 0:
                    window_start = sec

                # 核心变换: cloud_frame(t) → base_link_frame(t0)
                T_rel, T_fixed_to_base_ref = _compute_cloud_to_baselink_ref(
                    tf_buffer, cloud_frame, sec, window_start,
                    fixed_frame, base_link_frame, warn_set,
                )

                if reference_pose is None:
                    reference_pose = T_fixed_to_base_ref.copy()

                xyz, intensity = parse_pc2_msg(msg)
                if len(xyz) > 0:
                    if intensity is not None:
                        has_intensity = True
                    else:
                        intensity = np.ones(len(xyz), dtype=np.float32)

                    n = len(xyz)
                    pts_h = np.ones((n, 4), dtype=np.float64)
                    pts_h[:, :3] = xyz
                    transformed = (T_rel @ pts_h.T).T[:, :3].astype(np.float32)
                    accumulated_points.append(transformed)
                    accumulated_intensities.append(intensity)

                if sec - window_start >= interval:
                    _save_accumulated_frame(frame_dir, frame_idx,
                                            accumulated_points,
                                            accumulated_intensities,
                                            reference_pose,
                                            has_intensity)
                    frame_idx += 1
                    accumulated_points = []
                    accumulated_intensities = []
                    reference_pose = None
                    window_start = -1.0

                msg_idx += 1
                progress.update(task, completed=msg_idx)

    if accumulated_points:
        _save_accumulated_frame(frame_dir, frame_idx,
                                accumulated_points,
                                accumulated_intensities,
                                reference_pose,
                                has_intensity)
        frame_idx += 1

    # 汇总 TF 断链警告
    if warn_set:
        console.print(
            f"[yellow]⚠ 帧提取过程中出现 {len(warn_set)} 类 TF 链断裂，"
            f"已使用单位阵代替。详情请查看上方 Python 警告信息。[/yellow]"
        )

    console.print(
        f"提取完成！cloud frame_id = \"{first_cloud_frame}\"，"
        f"共 {frame_idx} 帧，intensity {'✓' if has_intensity else '✗'} → {frame_dir}"
    )


def _check_tf_chain(tf_buffer, fixed_frame, base_link_frame, pointcloud_topic):
    """预先检查 TF 链的完整性，对缺失的关键链路发出警告。"""
    all_keys = set()
    for (p, c) in list(tf_buffer['dynamic'].keys()) + list(tf_buffer['static'].keys()):
        all_keys.add((p, c))
        all_keys.add((c, p))  # 反向也算可用

    issues = []

    # 检查 fixed_frame → base_link_frame 是否有变换可用
    fwd = (fixed_frame, base_link_frame) in all_keys
    rev = (base_link_frame, fixed_frame) in all_keys
    if not fwd and not rev:
        issues.append(
            f"缺少 {fixed_frame} ↔ {base_link_frame} 的 TF 变换。"
            f"该变换是机器人的世界位姿，缺少将导致所有帧使用单位阵。"
        )

    if issues:
        for msg in issues:
            warnings.warn(f"[TF 链预检] {msg}")
        console.print(
            f"[yellow]⚠ TF 链预检发现问题 ({len(issues)} 项)，"
            f"详见上方警告。[/yellow]"
        )


def _save_accumulated_frame(frame_dir, frame_idx,
                            point_arrays, intensity_arrays,
                            reference_pose, has_intensity):
    if not point_arrays:
        return
    xyz = np.vstack(point_arrays)

    pcd_path = os.path.join(frame_dir, f"{frame_idx:06d}.pcd")
    if has_intensity and intensity_arrays:
        intensity = np.hstack(intensity_arrays)
        _write_pcd(pcd_path, xyz, intensity)
    else:
        _write_pcd(pcd_path, xyz)

    odom_path = os.path.join(frame_dir, f"{frame_idx:06d}.odom")
    if reference_pose is None:
        reference_pose = np.eye(4)
    np.savetxt(odom_path, reference_pose, fmt="%.6f")


# ---------------------------------------------------------------------------
# 点云解析
# ---------------------------------------------------------------------------

def parse_pc2_msg(msg):
    """解析 PointCloud2 → (xyz_Nx3, intensity_N 或 None)"""
    field_names = [f.name for f in msg.fields]
    has_intensity = 'intensity' in field_names

    try:
        from sensor_msgs_py import point_cloud2
        read_fields = ["x", "y", "z"]
        if has_intensity:
            read_fields.append("intensity")

        pts = list(point_cloud2.read_points(msg, field_names=read_fields, skip_nans=True))
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), None

        xyz = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        if has_intensity:
            intensity = np.array([p[3] for p in pts], dtype=np.float32)
        else:
            intensity = None

        return xyz, intensity

    except Exception:
        return _parse_pc2_manual(msg)


def _parse_pc2_manual(msg):
    """手动解析 PointCloud2 二进制数据"""
    field_names = [f.name for f in msg.fields]
    offsets = {f.name: f.offset for f in msg.fields}
    has_intensity = 'intensity' in field_names

    data_len = len(msg.data)
    points_count = data_len // msg.point_step

    xyz = np.zeros((points_count, 3), dtype=np.float32)
    intensity = np.zeros(points_count, dtype=np.float32) if has_intensity else None

    valid = 0
    for i in range(points_count):
        off = i * msg.point_step
        x = struct.unpack_from('f', msg.data, off + offsets['x'])[0]
        y = struct.unpack_from('f', msg.data, off + offsets['y'])[0]
        z = struct.unpack_from('f', msg.data, off + offsets['z'])[0]

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue

        xyz[valid, 0] = x
        xyz[valid, 1] = y
        xyz[valid, 2] = z

        if has_intensity:
            intensity[valid] = struct.unpack_from('f', msg.data, off + offsets['intensity'])[0]

        valid += 1

    xyz = xyz[:valid]
    if has_intensity:
        intensity = intensity[:valid]

    return xyz, intensity
