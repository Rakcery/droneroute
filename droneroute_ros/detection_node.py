"""
Algılama node'u: /camera/image_raw → engel hücreleri + hareketli engel tahmini

  /obstacles/grid  (nav_msgs/OccupancyGrid)  zaman filtresinden geçmiş şu anki engeller (100 / 0)
  /obstacles/tti   (nav_msgs/OccupancyGrid)  hareketli engel o hücreye kaç kare sonra geliyor (codec.py)
  /obstacles/mask  (sensor_msgs/Image mono8) piksel bazında engel maskesi (görsel için)
  /obstacles/height, /obstacles/tti_height  engellerin boyu (3D planlama için, codec.py)

/camera/height (derinlik) gelirse görüntüyle damgadan eşleştirilip boylar ölçülür; gelmezse (webcam) her engel "boyu bilinmiyor" = sonsuz
yüksek, yani 3D planlayıcı üstünden uçmaz.

Üçü de kameranın zaman damgasını taşır; planlama node'u grid ile tti'yi damgadan eşleştirir.
Güvenlik payı, duvar/ceza kararı burada değil, planlamada: bu node sadece "ne nerede, ne zaman" der.

Kalibrasyon (renk yöntemi): simülasyonda /camera/background'dan. Gerçek kamerada önce boş zemini göster,
sonra:  ros2 service call /detection/calibrate std_srvs/srv/Trigger
Parametreler: yontem (renk | kenar | yolo | hibrit), bekle (arka plan için kaç kare beklensin, sonra ilk kare zemin sayılır)
YOLO (drone_yolo.py): model (boş = models/droneroute_yoloe.pt), guven, perspektif (eğik kamera: tespitleri zemine
indir), irtifa, egim, fov, vfov (0 = kare piksel), hucre_m, kalibrasyon (JSON yolu, boş = kameradan hesapla).
Eğik kamerada /obstacles/* hâlâ kuşbakışı grid; kamera grid'in alt kenarının ortasında, ileri = yukarı.
"""
import time

import cv2
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from drone_vision import FRAME_H, FRAME_W, ObstacleDetector, Perception
from droneroute_ros import codec
from droneroute_ros.ros_util import spin, make_grid, make_image, read_image

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def fit(img):
    return img if img.shape[:2] == (FRAME_H, FRAME_W) else cv2.resize(img, (FRAME_W, FRAME_H))


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection")
        method = self.declare_parameter("yontem", "renk").value
        self.wait = self.declare_parameter("bekle", 15).value
        self.det = ObstacleDetector(method)
        if method in ("yolo", "hibrit"):
            self.det.yolo = self.make_yolo()
        self.perc = Perception(self.det)
        self.calibrated = method not in ("renk", "hibrit")
        self.calib_next = False
        self.waited = 0
        self.pub_grid = self.create_publisher(OccupancyGrid, "/obstacles/grid", 5)
        self.pub_tti = self.create_publisher(OccupancyGrid, "/obstacles/tti", 5)
        self.pub_mask = self.create_publisher(Image, "/obstacles/mask", 2)
        self.pub_h = self.create_publisher(OccupancyGrid, "/obstacles/height", 5)
        self.pub_th = self.create_publisher(OccupancyGrid, "/obstacles/tti_height", 5)
        # Derinlik varsa görüntüyle damgadan eşleştir (ikisi birlikte işlenir). Hiç gelmiyorsa (webcam)
        # görüntü tek başına işlenir, boylar "bilinmiyor" olur.
        self.last_pair = 0.0
        self.sync = TimeSynchronizer([Subscriber(self, Image, "/camera/image_raw"),
                                      Subscriber(self, Image, "/camera/height")], 10)
        self.sync.registerCallback(self.on_pair)
        self.create_subscription(Image, "/camera/background", self.on_background, LATCHED)
        self.create_subscription(Image, "/camera/image_raw", self.on_image_only, 5)
        self.create_service(Trigger, "/detection/calibrate", self.on_calibrate)
        self.get_logger().info(f"yontem={method} /camera/image_raw → /obstacles/grid + /obstacles/tti")

    def make_yolo(self):
        import drone_yolo as Y
        dp = lambda name, default: self.declare_parameter(name, default).value
        model, conf = dp("model", ""), dp("guven", 0.25)
        persp, calib = dp("perspektif", False), dp("kalibrasyon", "")
        cam = Y.Camera(dp("irtifa", 25.0), dp("egim", 35.0), dp("fov", 82.0), dp("vfov", 0.0) or None)
        cell_m = dp("hucre_m", Y.CELL_M)
        proj = None
        if persp:
            proj = Y.GroundProjector.from_file(calib, cam, cell_m) if calib else Y.GroundProjector.from_camera(cam, cell_m)
            self.get_logger().info(f"eğik kamera: irtifa={cam.alt} m eğim={cam.pitch}°, hücre={cell_m} m")
        return Y.YoloDetector(Y.UltralyticsBackend(model or None, conf), proj)

    def on_background(self, msg):
        self.det.calibrate(fit(read_image(msg)))
        self.calibrated = True
        self.get_logger().info("zemin /camera/background'dan kalibre edildi")

    def on_pair(self, img_msg, h_msg):
        self.last_pair = time.monotonic()
        h = read_image(h_msg)
        if h.shape != (FRAME_H, FRAME_W):
            h = cv2.resize(h, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        self.on_image(img_msg, h)

    def on_image_only(self, msg):
        if time.monotonic() - self.last_pair > 1.0:    # derinlik kamerası yok: sadece görüntü
            self.on_image(msg, None)

    def on_calibrate(self, _req, res):
        self.calib_next = True
        res.success, res.message = True, "sıradaki kare zemin olarak öğrenilecek"
        return res

    def on_image(self, msg, height):
        frame = fit(read_image(msg))
        if self.calib_next or not self.calibrated:
            if not self.calib_next and self.waited < self.wait:     # arka plan yolda olabilir, biraz bekle
                self.waited += 1
                return
            self.det.calibrate(frame)
            self.calibrated, self.calib_next = True, False
            self.get_logger().info("zemin bu kareden kalibre edildi (boş zemin değilse: /detection/calibrate)")
        stamp = msg.header.stamp
        raw, moving = self.perc.process(frame, height)
        self.pub_grid.publish(make_grid(stamp, codec.flags_to_data(raw)))
        self.pub_tti.publish(make_grid(stamp, codec.tti_to_data(moving)))
        self.pub_h.publish(make_grid(stamp, codec.height_to_data(self.perc.height)))
        self.pub_th.publish(make_grid(stamp, codec.height_to_data(self.perc.moving_height)))
        if self.pub_mask.get_subscription_count():
            self.pub_mask.publish(make_image(stamp, self.perc.last_mask))


def main():
    spin(DetectionNode)
