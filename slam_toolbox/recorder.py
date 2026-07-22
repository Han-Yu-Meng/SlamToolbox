import os
import subprocess
import signal
import time
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rich.live import Live
from rich.table import Table
from rich.console import Console

console = Console()


class RecorderMonitor(Node):
    """实时监控录制状态的 ROS2 节点。

    订阅配置的点云话题并监听 fixed_frame → base_link_frame 的 TF 变换。
    """

    def __init__(self, pointcloud_topic, fixed_frame, base_link_frame):
        super().__init__("recorder_monitor")
        self.cloud_count = 0
        self.x = 0.0
        self.y = 0.0
        self._fixed_frame = fixed_frame
        self._base_link_frame = base_link_frame

        self.subscription = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.cloud_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.update_pose)

    def cloud_callback(self, msg):
        self.cloud_count += 1

    def update_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self._fixed_frame, self._base_link_frame, now
            )
            self.x = trans.transform.translation.x
            self.y = trans.transform.translation.y
        except TransformException:
            pass


def ros_spin_thread(node):
    rclpy.spin(node)


def generate_status_table(count, x, y, elapsed, fixed_frame, base_link_frame, pointcloud_topic):
    table = Table(title="[bold green]ROS2 Bag 录制状态监控[/bold green]")
    table.add_column("监控指标", justify="left", style="cyan")
    table.add_column("实时数值", justify="right", style="magenta")

    table.add_row("已运行时间 (秒)", f"{elapsed:.1f}")
    table.add_row(f"已接收点云帧数 ({pointcloud_topic})", str(count))
    table.add_row(
        f"当前坐标 X ({fixed_frame} → {base_link_frame})", f"{x:.3f} m"
    )
    table.add_row(
        f"当前坐标 Y ({fixed_frame} → {base_link_frame})", f"{y:.3f} m"
    )
    return table


def start_recording(map_path, config=None):
    """启动 ros2 bag 录制。

    参数:
        map_path: 地图目录路径
        config:   配置字典（cli._ensure_config 的返回值），为 None 时使用默认值
    """
    if config is None:
        config = {
            "config": {
                "fixed_frame": "odom",
                "base_link_frame": "base_link",
                "pointcloud_topic": "/cloud_registered",
            }
        }

    cfg = config["config"]
    pointcloud_topic = cfg["pointcloud_topic"]
    fixed_frame = cfg["fixed_frame"]
    base_link_frame = cfg["base_link_frame"]

    bag_dir = os.path.join(map_path, "bag")
    output_bag = os.path.join(bag_dir, "bag")

    # 1. 启动监控节点
    rclpy.init()
    monitor_node = RecorderMonitor(pointcloud_topic, fixed_frame, base_link_frame)
    spin_thread = threading.Thread(
        target=ros_spin_thread, args=(monitor_node,), daemon=True
    )
    spin_thread.start()

    # 2. 启动 ros2 bag 录制子进程 — 话题来自 config
    topics = [pointcloud_topic, "/Odometry", "/tf", "/tf_static"]
    cmd = ["ros2", "bag", "record", "-o", output_bag] + topics

    console.print(f"[dim]执行命令: {' '.join(cmd)}[/dim]")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    start_time = time.time()
    console.print("开始录制... 按 Enter 键或 Ctrl+C 停止录制。")

    try:
        with Live(
            generate_status_table(
                0, 0.0, 0.0, 0.0, fixed_frame, base_link_frame, pointcloud_topic
            ),
            refresh_per_second=4,
        ) as live:
            while proc.poll() is None:
                elapsed = time.time() - start_time
                live.update(
                    generate_status_table(
                        monitor_node.cloud_count,
                        monitor_node.x,
                        monitor_node.y,
                        elapsed,
                        fixed_frame,
                        base_link_frame,
                        pointcloud_topic,
                    )
                )
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        # 优雅终止 ros2 bag 录制
        proc.send_signal(signal.SIGINT)
        proc.wait()

        # 清理节点
        monitor_node.destroy_node()
        rclpy.shutdown()
        console.print("\n录制已结束，数据已保存。")
