"""
SLAM Toolbox 配置文件管理模块。

提供 config.yaml 的生成、加载与校验功能。
"""

import os
import yaml
import numpy as np
from rich.console import Console

console = Console()

# 默认配置（最小可用模板）
DEFAULT_CONFIG = {
    "config": {
        "fixed_frame": "odom",
        "base_link_frame": "base_link",
        "pointcloud_topic": "/cloud_registered",
    }
}


def _quaternion_to_matrix(qx, qy, qz, qw):
    """四元数 → 4x4 齐次变换矩阵"""
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
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (qx, qy, qz)  # fallback — overwritten by caller
    return T


def generate_config(path, force=False):
    """
    生成默认 config.yaml。

    参数:
        path:  目标 YAML 文件路径
        force: 若为 True，直接覆盖已存在的文件；否则提示用户确认。

    返回:
        bool: 是否成功生成（用户确认覆盖 / 原来不存在）
    """
    if os.path.exists(path) and not force:
        console.print(
            f"[yellow]⚠ config.yaml 已存在于 {path}，是否覆盖？[/yellow]"
        )
        answer = input("  输入 y 覆盖，其他键跳过: ").strip().lower()
        if answer != "y":
            console.print("[dim]  跳过 config 生成。[/dim]")
            return False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    console.print(f"[green]✓ 已生成默认 config.yaml → {path}[/green]")
    return True


def load_config(path):
    """
    加载并校验 config.yaml。

    返回:
        dict: 配置字典（顶层键为 "config"），包含:
              - fixed_frame (str)
              - base_link_frame (str)
              - pointcloud_topic (str)
              - fixed_transform (dict, 可选): {child_name: {parent, position, rotation}}

    异常:
        FileNotFoundError: 文件不存在
        ValueError:      格式错误或缺少必要字段
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml 未找到: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("config.yaml 为空。")

    if "config" not in cfg:
        raise ValueError("config.yaml 缺少顶层 'config' 键。")

    c = cfg["config"]

    # 必要字段校验
    for key in ("fixed_frame", "base_link_frame", "pointcloud_topic"):
        if key not in c:
            raise ValueError(f"config.yaml 中 'config' 缺少必要字段: {key}")

    # 校验 fixed_transform（可选）
    if "fixed_transform" in c:
        ft = c["fixed_transform"]
        if not isinstance(ft, dict):
            raise ValueError("config.fixed_transform 必须是一个字典。")
        for child_name, entry in ft.items():
            if "parent" not in entry:
                raise ValueError(
                    f"fixed_transform.{child_name} 缺少 'parent' 字段。"
                )
            for arr_key in ("position", "rotation"):
                arr = entry.get(arr_key, [])
                if len(arr) not in (3, 4):
                    raise ValueError(
                        f"fixed_transform.{child_name}.{arr_key} "
                        f"长度必须为 3 (position) 或 4 (rotation)，实际: {len(arr)}"
                    )

    return cfg


def build_fixed_transforms(config):
    """
    将 config 中的 fixed_transform 转换为 TF buffer 可用的静态变换字典。

    返回:
        dict: {(parent, child): (timestamp_sec, 4x4_matrix)}
    """
    static = {}
    c = config.get("config", config)
    ft = c.get("fixed_transform", {})
    if not ft:
        return static

    for child_name, entry in ft.items():
        parent = entry["parent"]
        pos = np.array(entry["position"], dtype=np.float64)
        rot = np.array(entry["rotation"], dtype=np.float64)

        # 四元数: xyzw
        T = _quaternion_to_matrix(rot[0], rot[1], rot[2], rot[3])
        T[:3, 3] = pos[:3]

        key = (parent, child_name)
        static[key] = (0.0, T)

    return static
