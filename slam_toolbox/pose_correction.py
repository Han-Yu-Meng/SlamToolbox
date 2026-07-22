"""
位姿修正模块。

提供:
  - 强制平面约束：将 odom 位姿的 z / roll / pitch 置零，仅保留 x / y / yaw
  - Interactive SLAM：占位
"""

import os
import numpy as np
from rich.console import Console

console = Console()


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
    console.print(
        f"  max |z|     = {max_abs_z:.4f} m"
    )
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


def start_interactive_slam(map_path):
    """
    Interactive SLAM —— 占位。
    """
    console.print(
        "[yellow]Interactive SLAM 功能尚未实现，敬请期待。[/yellow]"
    )
