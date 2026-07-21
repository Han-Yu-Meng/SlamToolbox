import os
import sys
import questionary
from rich.console import Console

console = Console()

MAP_BASE_DIR = os.path.expanduser("~/Map")

def get_map_directories():
    """扫描 ~/Map 目录下所有包含或可能用于存放地图的文件夹"""
    os.makedirs(MAP_BASE_DIR, exist_ok=True)
    dirs = [d for d in os.listdir(MAP_BASE_DIR) if os.path.isdir(os.path.join(MAP_BASE_DIR, d))]
    # 过滤掉隐藏文件夹
    return [d for d in dirs if not d.startswith('.')]

def main():
    try:
        import rclpy
    except ImportError:
        console.print("[red]错误: 未检测到 ROS2 环境。请先 source 您的 ROS2 工作空间再运行此工具。[/red]")
        sys.exit(1)

    console.print("[bold green]欢迎使用 SLAM Toolbox CLI[/bold green]\n")
    
    # 1. 选择工作地图目录
    dirs = get_map_directories()

    NEW_MAP_TOKEN = "新建地图"

    while True:
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
            break
        else:
            break

    # 设置常用路径
    map_path = os.path.abspath(os.path.join(MAP_BASE_DIR, map_name))
    
    # 2. 主功能循环
    while True:
        action = questionary.select(
            f"当前地图: {map_name} | 请选择操作类型:",
            choices=[
                "1. 3D Map",
                "2. 2D Map",
                "退出"
            ]
        ).ask()

        if action == "1. 3D Map":
            sub_action = questionary.select(
                "3D Map 子功能列表:",
                choices=[
                    "1. Bag Recorder (录制 Bag 包)",
                    "2. Frame Extractor (帧提取)",
                    "3. Map Builder (点云地图构建)",
                    "4. ERASOR2 (动态障碍物去除)",
                    "返回上一级"
                ]
            ).ask()

            if "1. Bag Recorder" in sub_action:
                from .recorder import start_recording
                start_recording(map_path)
            elif "2. Frame Extractor" in sub_action:
                from .extractor import start_extraction
                start_extraction(map_path)
            elif "3. Map Builder" in sub_action:
                from .builder import start_building
                start_building(map_path)
            elif "4. ERASOR2" in sub_action:
                from .erasor2 import start_erasor2
                start_erasor2(map_path)

        elif action == "2. 2D Map":
            sub_action = questionary.select(
                "2D Map 子功能列表:",
                choices=[
                    "1. PGM Generator (生成 2D 栅格地图)",
                    "返回上一级"
                ]
            ).ask()

            if "1. PGM Generator" in sub_action:
                from .pgm_generator import start_generation
                start_generation(map_path)

        else:
            break

if __name__ == "__main__":
    main()
