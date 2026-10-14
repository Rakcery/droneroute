"""ROS mesajı kurma yardımcıları (node'ların ortak kısmı)."""
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image

from drone_dstar_lite import COLS, ROWS
from droneroute_ros import codec

FRAME_ID = "map"


def make_grid(stamp, data):
    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = FRAME_ID
    msg.info.map_load_time = stamp
    msg.info.resolution = float(codec.RESOLUTION)
    msg.info.width, msg.info.height = COLS, ROWS
    msg.info.origin.orientation.w = 1.0
    msg.data = data
    return msg


def make_pose(stamp, cell, alt=0.0):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = FRAME_ID
    msg.pose.position.x, msg.pose.position.y = codec.cell_to_xy(cell)
    msg.pose.position.z = float(alt)
    msg.pose.orientation.w = 1.0
    return msg


def make_image(stamp, img, frame_id="camera"):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    if img.dtype == np.float32:
        msg.encoding, msg.step = "32FC1", img.shape[1] * 4
    elif img.ndim == 2:
        msg.encoding, msg.step = "mono8", img.shape[1]
    else:
        msg.encoding, msg.step = "bgr8", img.shape[1] * 3
    msg.data = img.tobytes()
    return msg


def read_image(msg):
    return codec.image_to_array(msg.height, msg.width, msg.encoding, msg.data)


def spin(node_cls):
    """Node'u çalıştır, Ctrl+C'de sessizce kapan."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
