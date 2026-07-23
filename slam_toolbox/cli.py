import os
import sys
import questionary
from questionary import Choice
from rich.console import Console

from .config import generate_config, load_config

console = Console()

MAP_BASE_DIR = os.path.expanduser("~/Map")


def get_map_directories():
    """扫描 ~/Map 目录下所有包含或可能用于存放地图的文件夹"""
    os.makedirs(MAP_BASE_DIR, exist_ok=True)
    dirs = [
        d for d in os.listdir(MAP_BASE_DIR)
        if os.path.isdir(os.path.join(MAP_BASE_DIR, d))
    ]
    # 过滤掉隐藏文件夹
    return [d for d in dirs if not d.startswith(".")]


def _ensure_config(map_path):
    """确保 map_path 下存在 config.yaml；若不存在则引导用户生成。"""
    config_path = os.path.join(map_path, "config.yaml")

    if os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            console.print(
                f"[dim]已加载 config.yaml: "
                f"fixed_frame={cfg['config']['fixed_frame']}, "
                f"base_link_frame={cfg['config']['base_link_frame']}, "
                f"pointcloud_topic={cfg['config']['pointcloud_topic']}[/dim]"
            )
            return cfg
        except Exception as e:
            console.print(f"[red]加载 config.yaml 失败: {e}[/red]")
            # 继续询问是否覆盖
            pass

    console.print("[yellow]当前地图目录未找到有效的 config.yaml。[/yellow]")
    return _generate_and_load_config(config_path)


def _generate_and_load_config(config_path):
    """交互式生成 config.yaml 并加载返回。"""
    create = questionary.confirm(
        "是否生成默认 config.yaml？",
        default=True
    ).ask()

    if not create:
        console.print("[dim]跳过 config 生成，各模块将使用内置默认值。[/dim]")
        return {
            "config": {
                "fixed_frame": "odom",
                "base_link_frame": "base_link",
                "pointcloud_topic": "/cloud_registered",
            }
        }

    # 让用户确认/修改关键参数
    fixed_frame = questionary.text(
        "固定坐标系 (fixed_frame):",
        default="odom"
    ).ask()

    base_link_frame = questionary.text(
        "机器人底盘坐标系 (base_link_frame):",
        default="base_link"
    ).ask()

    pointcloud_topic = questionary.text(
        "点云话题 (pointcloud_topic):",
        default="/cloud_registered"
    ).ask()

    config = {
        "config": {
            "fixed_frame": fixed_frame,
            "base_link_frame": base_link_frame,
            "pointcloud_topic": pointcloud_topic,
        }
    }

    import yaml
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"[green]✓ config.yaml 已保存 → {config_path}[/green]")
    return config


def main():
    try:
        import rclpy
    except ImportError:
        console.print(
            "[red]错误: 未检测到 ROS2 环境。"
            "请先 source 您的 ROS2 工作空间再运行此工具。[/red]"
        )
        sys.exit(1)

    console.print("[bold green]欢迎使用 SLAM Toolbox CLI[/bold green]\n")

    NEW_MAP_TOKEN = "新建地图"

    while True:
        # 1. 选择工作地图目录
        dirs = get_map_directories()
        choices = dirs + [NEW_MAP_TOKEN] if dirs else [NEW_MAP_TOKEN]
        map_name = questionary.select(
            "请选择需要操作的地图目录:",
            choices=choices
        ).ask()

        if not map_name:
            return

        if map_name == NEW_MAP_TOKEN:
            new_name = questionary.text("请输入新地图名称:").ask()
            if not new_name:
                continue
            os.makedirs(os.path.join(MAP_BASE_DIR, new_name, "bag"), exist_ok=True)
            os.makedirs(os.path.join(MAP_BASE_DIR, new_name, "map"), exist_ok=True)
            dirs = get_map_directories()  # 刷新列表
            map_name = new_name

        # 设置常用路径
        map_path = os.path.abspath(os.path.join(MAP_BASE_DIR, map_name))

        # 2. 加载或生成 config.yaml
        config = _ensure_config(map_path)

        # 3. 主功能循环
        change_map = False
        while True:
            action = questionary.select(
                f"当前地图: {map_name} | 请选择操作类型:",
                choices=[
                    "1. 3D Map",
                    "2. 2D Map",
                    "3. 更换地图",
                    "退出"
                ]
            ).ask()

            if action == "1. 3D Map":
                sub_action = questionary.select(
                    "3D Map — 请选择操作:",
                    choices=[
                        Choice(
                            title=[("bold ansiyellow", "数据获取")],
                            disabled="section",
                        ),
                        "   1. 录制 rosbag",
                        Choice(
                            title=[("bold ansiyellow", "帧构建")],
                            disabled="section",
                        ),
                        "   2. Frame Extractor",
                        Choice(
                            title=[("bold ansiyellow", "位姿修正")],
                            disabled="section",
                        ),
                        "   3. 强制平面约束",
                        "   4. Interactive SLAM",
                        Choice(
                            title=[("bold ansiyellow", "动态障碍物清除")],
                            disabled="section",
                        ),
                        "   5. ERASOR2",
                        "   6. Removert",
                        "   7. Local Hash Voxel",
                        "   8. Raycast Voxel",
                        Choice(
                            title=[("bold ansiyellow", "构建全局地图")],
                            disabled="section",
                        ),
                        "   9. Map Builder",
                        Choice(
                            title=[("", "")],
                            disabled="section",
                        ),
                        "   返回上一级",
                    ]
                ).ask()

                if sub_action is None or "返回上一级" in sub_action:
                    continue

                if "录制 rosbag" in sub_action:
                    from .recorder import start_recording
                    start_recording(map_path, config)
                elif "Frame Extractor" in sub_action:
                    from .extractor import start_extraction
                    start_extraction(map_path, config)
                elif "强制平面约束" in sub_action:
                    from .pose_correction import start_planar_constraint
                    start_planar_constraint(map_path)
                elif "Interactive SLAM" in sub_action:
                    from .pose_correction import start_interactive_slam
                    start_interactive_slam(map_path)
                elif "ERASOR2" in sub_action:
                    from .dynamic_removal import start_erasor2
                    start_erasor2(map_path)
                elif "Removert" in sub_action:
                    from .dynamic_removal import start_removert
                    start_removert(map_path)
                elif "Local Hash Voxel" in sub_action:
                    from .dynamic_removal import start_local_hash_voxel
                    start_local_hash_voxel(map_path)
                elif "Raycast Voxel" in sub_action:
                    from .dynamic_removal import start_raycast_voxel
                    start_raycast_voxel(map_path)
                elif "Map Builder" in sub_action:
                    from .builder import start_building
                    start_building(map_path)

            elif action == "2. 2D Map":
                sub_action = questionary.select(
                    "2D Map 子功能列表:",
                    choices=[
                        "1. PGM Generator (生成 2D 栅格地图)",
                        "返回上一级"
                    ]
                ).ask()

                if sub_action and "1. PGM Generator" in sub_action:
                    from .pgm_generator import start_generation
                    start_generation(map_path)

            elif action == "3. 更换地图":
                change_map = True
                break

            else:
                return

        if change_map:
            continue


if __name__ == "__main__":
    main()
