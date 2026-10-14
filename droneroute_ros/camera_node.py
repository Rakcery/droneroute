"""
Kamera node'u: sentetik şantiye sahnesi, webcam ya da video → /camera/image_raw (sensor_msgs/Image, bgr8)

Simülasyonda boş zemin görüntüsünü de bir kere /camera/background'a yayınlar (latched), algılama node'u
renk kalibrasyonunu ondan yapar. Gerçek kamerada bu yok, algılama ilk kareden ya da servisle kalibre olur.

Simülasyonda ayrıca /camera/height (32FC1): piksel başına zeminden yükseklik (metre). Gerçek drone'da bu
bir derinlik kamerasından gelir; 3D planlama için. Görüntüden önce, aynı zaman damgasıyla yayınlanır.

Parametreler: kaynak (sim | kamera | video), kamera_id, video (dosya yolu), seed (-1 = rastgele), fps
"""
import argparse

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

from drone_vision import Source
from droneroute_ros.ros_util import spin, make_image

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera")
        kaynak = self.declare_parameter("kaynak", "sim").value
        video = self.declare_parameter("video", "").value
        seed = self.declare_parameter("seed", -1).value
        args = argparse.Namespace(kamera=kaynak == "kamera", kamera_id=self.declare_parameter("kamera_id", 0).value,
                                  video=video if kaynak == "video" and video else None,
                                  seed=None if seed < 0 else seed)
        self.src = Source(args)
        self.pub = self.create_publisher(Image, "/camera/image_raw", 5)
        self.pub_h = self.create_publisher(Image, "/camera/height", 5)
        if self.src.sim:
            bg = self.create_publisher(Image, "/camera/background", LATCHED)
            bg.publish(make_image(self.get_clock().now().to_msg(), self.src.sim.background()))
        fps = self.declare_parameter("fps", 30.0).value
        self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f"kaynak={self.src.name} fps={fps:g} → /camera/image_raw")

    def tick(self):
        frame, _ = self.src.read()
        if frame is None:
            self.get_logger().warn("kare alınamadı", throttle_duration_sec=2.0)
            return
        stamp = self.get_clock().now().to_msg()
        height = self.src.height()
        if height is not None:
            self.pub_h.publish(make_image(stamp, height))
        self.pub.publish(make_image(stamp, frame))


def main():
    spin(CameraNode)
