"""
位姿修正模块。

提供:
  - 强制平面约束：将 odom 位姿的 z / roll / pitch 置零，仅保留 x / y / yaw
  - Interactive SLAM：通过 Docker GUI 工具交互式修正位姿
"""

import os
import glob
import shutil
import subprocess
import questionary
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

# Docker 镜像
INTERACTIVE_SLAM_IMAGE = "stevenmhy/slamtoolbox-interactive_slam:latest"


# ---------------------------------------------------------------------------
# 强制平面约束
# ---------------------------------------------------------------------------

def _decompose_pose(T):
    """从 4x4 齐次矩阵中提取 x, y, z, yaw（假设 roll=pitch=0 时有效）。"""
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return x, y, z, yaw


def _make_planar_pose(x, y, yaw):
    """构造仅含 x, y, yaw 的 4x4 齐次矩阵（z=0, roll=0, pitch=0）。"""
    c = np.cos(yaw)
    s = np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = 0.0
    return T


def start_planar_constraint(map_path):
    """
    强制平面约束：读取 frame 目录下所有 .odom 文件，
    将位姿的 z / roll / pitch 强制置零，仅保留 x / y / yaw。
    """
    frame_dir = os.path.join(map_path, "frame")
    if not os.path.exists(frame_dir):
        console.print(f"[red]帧目录 {frame_dir} 不存在。请先运行帧构建。[/red]")
        return

    odom_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".odom")]
    )
    if not odom_files:
        console.print("[red]未在帧目录中找到 .odom 文件。[/red]")
        return

    # 第一遍：统计原始位姿的非平面分量
    max_abs_z = 0.0
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    matrices = {}

    for fname in odom_files:
        path = os.path.join(frame_dir, fname)
        T = np.loadtxt(path)
        if T.shape != (4, 4):
            continue
        matrices[fname] = T

        z = T[2, 3]
        R = T[:3, :3]
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

        max_abs_z = max(max_abs_z, abs(z))
        max_abs_roll = max(max_abs_roll, abs(roll))
        max_abs_pitch = max(max_abs_pitch, abs(pitch))

    console.print(
        f"原始位姿非平面分量统计 "
        f"（共 {len(matrices)} 个）:"
    )
    console.print(f"  max |z|     = {max_abs_z:.4f} m")
    console.print(
        f"  max |roll|  = {max_abs_roll:.4f} rad ({np.degrees(max_abs_roll):.2f}°)"
    )
    console.print(
        f"  max |pitch| = {max_abs_pitch:.4f} rad ({np.degrees(max_abs_pitch):.2f}°)"
    )

    # 第二遍：施加平面约束
    modified = 0
    for fname, T in matrices.items():
        path = os.path.join(frame_dir, fname)

        x, y, z, yaw = _decompose_pose(T)
        T_new = _make_planar_pose(x, y, yaw)

        if not np.allclose(T, T_new):
            np.savetxt(path, T_new, fmt="%.6f")
            modified += 1

    if modified > 0:
        console.print(
            f"[green]✓ 平面约束完成：{modified}/{len(matrices)} 个位姿被修正。[/green]"
        )
    else:
        console.print(
            f"[dim]✓ 所有 {len(matrices)} 个位姿已经是平面的，无需修正。[/dim]"
        )


# ---------------------------------------------------------------------------
# Interactive SLAM
# ---------------------------------------------------------------------------

def _allow_docker_x11():
    """Allow the root user inside the local Docker container to open X11 windows."""
    try:
        subprocess.run(
            ["xhost", "+SI:localuser:root"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        console.print("[yellow]警告: 未找到 xhost，Docker GUI 可能无法连接 X11。[/yellow]")


def _docker_run_cmd(map_path, rosrun_cmd):
    """构建 docker run 命令，通过 --env 注入 MESA 变量，bash -c 执行 rosrun。"""
    map_abs = os.path.abspath(map_path)
    x11_socket = "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    display = os.environ.get("DISPLAY", ":0")
    xauthority = os.environ.get("XAUTHORITY") or os.path.expanduser("~/.Xauthority")

    bash_cmd = (
        "source /root/catkin_ws/devel/setup.bash && "
        f"{rosrun_cmd}"
    )

    cmd = [
        "docker", "run", "-it",
        "--net=host",
        "--env", f"DISPLAY={display}",
        "--env", "HOME=/root",
        "--env", "QT_X11_NO_MITSHM=1",
        "--env", "MESA_GL_VERSION_OVERRIDE=4.5",
        "--env", "MESA_GLSL_VERSION_OVERRIDE=450",
        "--volume", f"{x11_socket}",
        "--volume", f"{map_abs}:/Map:rw",
        "--volume", f"{map_abs}:/root/Map:rw",
    ]
    if os.path.exists(xauthority):
        cmd.extend([
            "--env", "XAUTHORITY=/tmp/.docker.xauth",
            "--volume", f"{xauthority}:/tmp/.docker.xauth:ro",
        ])
    cmd.extend([
        INTERACTIVE_SLAM_IMAGE,
        "/bin/bash", "-c", bash_cmd,
    ])
    return cmd


def start_interactive_slam(map_path):
    """
    Interactive SLAM —— 三阶段交互式位姿修正。

    阶段 1: odometry2graph
        启动 Docker 容器，用户在 GUI 中操作，将 odom 数据保存为
        interactive_slam 图格式到 /root/Map/interactive_slam/original/

    阶段 2: interactive_slam
        再次启动容器，用户在 GUI 中进行 interactive SLAM 优化，
        结果输出到 /root/Map/interactive_slam/corrected/

    阶段 3: 插值回填
        将稀疏修正位姿通过 Slerp 插值还原为稠密位姿，
        直接覆盖 frame/ 目录下的 .odom 文件（自动备份）。
    """
    frame_dir = os.path.join(map_path, "frame")
    if not os.path.exists(frame_dir):
        console.print(f"[red]帧目录 {frame_dir} 不存在。请先运行帧构建。[/red]")
        return

    odom_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".odom")]
    )
    if not odom_files:
        console.print("[red]未在帧目录中找到 .odom 文件，请先运行帧构建。[/red]")
        return

    map_abs = os.path.abspath(map_path)
    original_dir = os.path.join(map_path, "interactive_slam", "original")
    corrected_dir = os.path.join(map_path, "interactive_slam", "corrected")

    # ==================== 阶段 1: odometry2graph ====================
    console.print(
        Panel.fit(
            "[bold]阶段 1/3: odometry2graph[/bold]\n\n"
            "即将自动启动 Interactive SLAM GUI (odometry2graph)。\n\n"
            "在 GUI 中操作:\n"
            f"  1. [bold cyan]File → Open → ROS[/bold cyan]\n"
            f"     选择 [bold yellow]/Map/frame/[/bold yellow] 目录加载 odometry 数据\n"
            f"  2. [bold cyan]File → Save[/bold cyan]\n"
            f"     保存到 [bold yellow]/Map/interactive_slam/original/[/bold yellow]\n\n"
            "关闭 GUI 窗口后容器将自动退出。",
            title="Interactive SLAM",
            border_style="cyan",
        )
    )

    if not questionary.confirm("准备启动 GUI (odometry2graph)，是否继续？").ask():
        return

    console.print("[dim]正在启动 Docker 容器...[/dim]")
    os.makedirs(original_dir, exist_ok=True)
    _allow_docker_x11()

    ret = subprocess.call(
        _docker_run_cmd(map_path, "rosrun interactive_slam odometry2graph")
    )
    if ret != 0:
        console.print(f"[yellow]⚠ Docker 容器退出码: {ret}[/yellow]")

    # 检查用户是否保存了数据
    if not os.listdir(original_dir):
        console.print(
            "[yellow]⚠ 未在 /Map/interactive_slam/original/ 中检测到文件，"
            "请确认已在 GUI 中保存。[/yellow]"
        )
        if not questionary.confirm("是否仍要继续阶段 2？").ask():
            return

    # ==================== 阶段 2: interactive_slam ====================
    console.print(
        Panel.fit(
            "[bold]阶段 2/3: interactive_slam[/bold]\n\n"
            "即将自动启动 Interactive SLAM GUI (interactive_slam)。\n\n"
            "在 GUI 中操作:\n"
            f"  1. [bold cyan]File → Open → New Map[/bold cyan]\n"
            f"     选择 [bold yellow]/Map/interactive_slam/original/[/bold yellow] 加载图数据\n"
            f"  2. [bold cyan]File → Save → Save map data[/bold cyan]\n"
            f"     优化结果保存到 [bold yellow]/Map/interactive_slam/corrected/[/bold yellow]\n\n"
            "关闭 GUI 窗口后容器将自动退出。",
            title="Interactive SLAM",
            border_style="cyan",
        )
    )

    if not questionary.confirm("准备再次启动 GUI (interactive_slam)，是否继续？").ask():
        return

    console.print("[dim]正在启动 Docker 容器...[/dim]")
    os.makedirs(corrected_dir, exist_ok=True)
    _allow_docker_x11()

    ret = subprocess.call(
        _docker_run_cmd(map_path, "rosrun interactive_slam interactive_slam")
    )
    if ret != 0:
        console.print(f"[yellow]⚠ Docker 容器退出码: {ret}[/yellow]")

    # 检查是否有 corrected 输出
    if not os.listdir(corrected_dir):
        console.print(
            "[red]✗ 未在 /Map/interactive_slam/corrected/ 中检测到优化结果，"
            "无法继续阶段 3。[/red]"
        )
        return

    # ==================== 阶段 3: 插值回填 ====================
    console.print(
        Panel.fit(
            "[bold]阶段 3/3: 插值回填[/bold]\n\n"
            "将 interactive_slam 生成的稀疏修正位姿通过 Slerp 插值还原为\n"
            "稠密位姿，并直接覆盖 frame/ 目录下的 .odom 文件。\n\n"
            "原始 .odom 文件会自动备份到 frame_backup/。",
            title="Interactive SLAM",
            border_style="cyan",
        )
    )

    if not questionary.confirm("是否执行插值回填？").ask():
        return

    _interpolate_and_apply(map_path, frame_dir, corrected_dir, odom_files)


# ---------------------------------------------------------------------------
# 插值回填核心逻辑
# ---------------------------------------------------------------------------

def _parse_data_file(file_path):
    """
    解析 interactive_slam 生成的 data 文件。

    返回:
        dict: {'stamp_id': int, 'estimate': 4x4 np.array, 'odom': 4x4 np.array}
        或 None（解析失败）
    """
    data_info = {"stamp_id": None, "estimate": None, "odom": None}

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("stamp"):
                    parts = line.split()
                    if len(parts) >= 2:
                        data_info["stamp_id"] = int(parts[1])
                elif line.startswith("estimate"):
                    matrix = []
                    for j in range(1, 5):
                        if i + j < len(lines):
                            matrix.append(
                                [float(x) for x in lines[i + j].split()]
                            )
                    if len(matrix) == 4:
                        data_info["estimate"] = np.array(matrix)
                elif line.startswith("odom"):
                    matrix = []
                    for j in range(1, 5):
                        if i + j < len(lines):
                            matrix.append(
                                [float(x) for x in lines[i + j].split()]
                            )
                    if len(matrix) == 4:
                        data_info["odom"] = np.array(matrix)
    except Exception as e:
        console.print(f"[yellow]解析 {file_path} 失败: {e}[/yellow]")
        return None

    if data_info["stamp_id"] is None or data_info["estimate"] is None:
        return None
    return data_info


def _interpolate_and_apply(map_path, frame_dir, corrected_dir, odom_files):
    """读取 corrected 稀疏位姿，插值并覆盖 frame/ 中的 .odom 文件。"""
    try:
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
    except ImportError:
        console.print(
            "[red]错误: 需要 scipy 库。请执行: pip install scipy[/red]"
        )
        return

    # ---- 获取所有原始稠密帧 ID ----
    raw_ids = []
    for f in odom_files:
        try:
            raw_ids.append(int(os.path.basename(f).split(".")[0]))
        except ValueError:
            continue
    raw_ids.sort()

    if not raw_ids:
        console.print("[red]无法从 .odom 文件名中提取帧 ID。[/red]")
        return
    console.print(f"原始稠密帧: {len(raw_ids)} 个")

    # ---- 读取优化后的关键帧 ----
    subdirs = sorted(
        [
            d
            for d in os.listdir(corrected_dir)
            if os.path.isdir(os.path.join(corrected_dir, d))
        ]
    )

    kf_data_list = []
    seen_ids = set()
    for subdir in subdirs:
        data_file = os.path.join(corrected_dir, subdir, "data")
        if not os.path.exists(data_file):
            continue

        data = _parse_data_file(data_file)
        if data is None:
            continue

        stamp_id = data["stamp_id"]
        # 防止重复帧导致 Slerp 崩溃
        if stamp_id not in seen_ids:
            seen_ids.add(stamp_id)
            kf_data_list.append(data)

    if len(kf_data_list) < 2:
        console.print(
            "[red]错误: 优化后的关键帧不足 2 个，无法进行插值。[/red]"
        )
        return

    kf_data_list.sort(key=lambda x: x["stamp_id"])
    console.print(f"优化后稀疏关键帧: {len(kf_data_list)} 个")

    # ---- 计算关键帧上的误差漂移 Delta_T ----
    kf_ids = []
    delta_translations = []
    delta_quats = []

    console.print("计算 SE(3) 漂移修正量...")
    for data in kf_data_list:
        stamp_id = data["stamp_id"]
        stamp_str = f"{stamp_id:06d}"
        T_opt = data["estimate"]

        raw_odom_path = os.path.join(frame_dir, f"{stamp_str}.odom")
        if not os.path.exists(raw_odom_path):
            console.print(
                f"[yellow]警告: 关键帧 {stamp_id} 对应的原始 .odom 不存在。[/yellow]"
            )
            continue

        T_raw = np.loadtxt(raw_odom_path)
        if T_raw.shape != (4, 4):
            continue

        # Delta_T = T_opt @ inv(T_raw)
        T_raw_inv = np.linalg.inv(T_raw)
        Delta_T = T_opt @ T_raw_inv

        kf_ids.append(stamp_id)
        delta_translations.append(Delta_T[:3, 3])
        r = R.from_matrix(Delta_T[:3, :3])
        delta_quats.append(r.as_quat())

    if len(kf_ids) == 0:
        console.print(
            "[red]错误: 没有关键帧能匹配到原始 .odom 文件。[/red]"
        )
        return

    kf_ids = np.array(kf_ids)
    delta_translations = np.array(delta_translations)

    # 构建球面线性插值器 (Slerp)
    rotations = R.from_quat(delta_quats)
    slerp = Slerp(kf_ids, rotations)

    # ---- 备份原始 frame/ 目录 ----
    backup_dir = os.path.join(map_path, "frame_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        for f in odom_files:
            src = os.path.join(frame_dir, f)
            dst = os.path.join(backup_dir, f)
            shutil.copy2(src, dst)
        console.print(f"[dim]原始 .odom 已备份到 {backup_dir}[/dim]")

    # ---- 对所有稠密帧插值并应用修正 ----
    console.print("插值并应用修正到所有稠密帧...")
    processed = 0

    for i in raw_ids:
        i_str = f"{i:06d}"

        # 边界处理：早于第一个 / 晚于最后一个关键帧，维持常量漂移
        if i <= kf_ids[0]:
            t = delta_translations[0]
            r = rotations[0]
        elif i >= kf_ids[-1]:
            t = delta_translations[-1]
            r = rotations[-1]
        else:
            idx_right = np.searchsorted(kf_ids, i)
            idx_left = idx_right - 1

            id_L = kf_ids[idx_left]
            id_R = kf_ids[idx_right]

            alpha = (i - id_L) / float(id_R - id_L)

            # 平移：线性插值
            t_L = delta_translations[idx_left]
            t_R = delta_translations[idx_right]
            t = (1.0 - alpha) * t_L + alpha * t_R

            # 旋转：球面插值
            r = slerp(i)

        # 重构当前帧的误差修正矩阵 Delta_T
        Delta_T = np.eye(4)
        Delta_T[:3, :3] = r.as_matrix()
        Delta_T[:3, 3] = t

        # 读取原始位姿，施加修正
        raw_odom_path = os.path.join(frame_dir, f"{i_str}.odom")
        T_raw = np.loadtxt(raw_odom_path)
        T_final = Delta_T @ T_raw

        # 写回 frame 目录
        np.savetxt(raw_odom_path, T_final, fmt="%.10f")
        processed += 1

    console.print(
        f"[green]✓ 插值回填完成！已修正 {processed}/{len(raw_ids)} 个位姿。[/green]"
    )
    console.print(f"[dim]  修正后位姿已写回: {frame_dir}[/dim]")
    console.print(f"[dim]  原始备份位于:   {backup_dir}[/dim]")
