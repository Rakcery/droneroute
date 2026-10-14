"""
Görsel node'u: kamera görüntüsünün üstüne engelleri, payı, tahmini, rotayı ve drone'u çizer.

  /viz/image (sensor_msgs/Image)  çizilmiş görüntü (rqt_image_view ile de bakılabilir)
  goster:=true ise OpenCV penceresi açar (WSL2'de WSLg ile Windows'ta pencere olarak görünür):
    sol tık  → /goal_pose     (yeni hedef)
    sağ tık  → /initialpose   (drone'u ışınla)

Sadece dinler ve çizer; hiçbir node buna bağımlı değil, kapatılsa da sistem çalışır.
"""
import os
from types import SimpleNamespace

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import drone_tracker as tracking
import drone_vision as V
from drone_dstar3d import ALTS, CLEARANCE
from drone_dstar_lite import COLS, ROWS
from droneroute_ros import codec
from droneroute_ros.ros_util import make_image, make_pose, read_image, spin

P = V.CELL_PX
WIN = "DroneRoute ROS 2 - kamera > algilama > planlama"


def fill_cells(view, cells, color, alpha=0.45):
    if not cells.any():
        return view
    overlay = view.copy()
    for y, x in zip(*np.nonzero(cells)):
        cv2.rectangle(overlay, (x * P, y * P), ((x + 1) * P - 1, (y + 1) * P - 1), color, -1)
    return cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)


class VizNode(Node):
    def __init__(self):
        super().__init__("viz")
        self.show = self.declare_parameter("goster", True).value
        if self.show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.get_logger().warn("ekran yok (DISPLAY boş), pencere açmıyorum; /viz/image'a bakabilirsin")
            self.show = False
        self.frame = self.mask = None
        self.raw = self.occ = np.zeros((ROWS, COLS), bool)
        self.tti = np.full((ROWS, COLS), np.inf)
        self.path, self.drone, self.goal, self.status = [], None, None, "planlayici bekleniyor"
        self.alt = 0.0                                     # drone irtifası; 0 ise planlayıcı 2D
        self.height = np.zeros((ROWS, COLS))
        self.count = 0

        def sub(msg_type, topic, fn):
            self.create_subscription(msg_type, topic, fn, 2)
        sub(Image, "/camera/image_raw", lambda m: setattr(self, "frame", read_image(m)))
        sub(Image, "/obstacles/mask", lambda m: setattr(self, "mask", read_image(m)))
        sub(OccupancyGrid, "/obstacles/grid", lambda m: setattr(self, "raw", codec.data_to_flags(m.data)))
        sub(OccupancyGrid, "/obstacles/tti", lambda m: setattr(self, "tti", codec.data_to_tti(m.data)))
        sub(OccupancyGrid, "/obstacles/height", lambda m: setattr(self, "height", codec.data_to_height(m.data)))
        sub(OccupancyGrid, "/plan/costmap", self.on_cost)
        sub(Path, "/plan/path", lambda m: setattr(self, "path", [(*self._cell(p.pose), self._level(p.pose))
                                                                    for p in m.poses]))
        sub(PoseStamped, "/drone/pose", self.on_drone)
        sub(PoseStamped, "/plan/goal", lambda m: setattr(self, "goal", self._cell(m.pose)))
        sub(String, "/plan/status", lambda m: setattr(self, "status", m.data))
        self.pub_img = self.create_publisher(Image, "/viz/image", 2)
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 2)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 2)
        if self.show:
            cv2.namedWindow(WIN)
            cv2.setMouseCallback(WIN, self.on_mouse)
        self.create_timer(1.0 / 20, self.tick)

    @staticmethod
    def _cell(pose):
        return codec.xy_to_cell(pose.position.x, pose.position.y)

    @staticmethod
    def _level(pose):
        return int(np.argmin([abs(a - pose.position.z) for a in ALTS]))

    def on_drone(self, msg):
        self.drone, self.alt = self._cell(msg.pose), msg.pose.position.z

    def on_cost(self, msg):
        self.occ, _ = codec.data_to_cost(msg.data)
        self.count += 1

    def on_mouse(self, event, x, y, *_):
        if x >= V.FRAME_W or y >= V.FRAME_H:
            return
        stamp, c = self.get_clock().now().to_msg(), (x // P, y // P)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pub_goal.publish(make_pose(stamp, c))
        elif event == cv2.EVENT_RBUTTONDOWN:
            m = PoseWithCovarianceStamped()
            pose = make_pose(stamp, c)
            m.header, m.pose.pose = pose.header, pose.pose
            self.pub_init.publish(m)

    def draw(self):
        view = cv2.resize(self.frame, (V.FRAME_W, V.FRAME_H))
        is3d = self.alt > 0
        low = self.raw & (self.alt >= self.height + CLEARANCE) if is3d else np.zeros_like(self.raw)
        view = fill_cells(view, self.raw & ~low, V.C_RAW)
        view = fill_cells(view, low, V.C_LOW)
        view = fill_cells(view, self.occ & ~self.raw, V.C_INFL)
        view = fill_cells(view, (self.tti > 0) & (self.tti <= tracking.HORIZON), V.C_PRED)
        for x in range(0, V.FRAME_W, P):
            cv2.line(view, (x, 0), (x, V.FRAME_H), (60, 60, 60), 1)
        for y in range(0, V.FRAME_H, P):
            cv2.line(view, (0, y), (V.FRAME_W, y), (60, 60, 60), 1)
        mid = lambda c: (c[0] * P + P // 2, c[1] * P + P // 2)
        for a, b in zip(self.path, self.path[1:]):
            if not is3d:
                cv2.line(view, mid(a), mid(b), V.C_PATH, 3, cv2.LINE_AA)
            elif a[2] == b[2]:
                cv2.line(view, mid(a), mid(b), V.C_LEVELS[a[2]], 3, cv2.LINE_AA)
            else:
                cv2.circle(view, mid(a), P // 2 - 3, V.C_LEVELS[b[2]], 2, cv2.LINE_AA)
        if self.goal:
            gx, gy = self.goal
            cv2.rectangle(view, (gx * P + 3, gy * P + 3), ((gx + 1) * P - 4, (gy + 1) * P - 4), V.C_GOAL, -1)
        if self.drone:
            dx, dy = self.drone
            cv2.circle(view, (dx * P + P // 2, dy * P + P // 2), P // 2 - 2, V.C_DRONE, -1, cv2.LINE_AA)
            if is3d:
                for col, th in (((0, 0, 0), 3), (V.C_TEXT, 1)):
                    cv2.putText(view, f"{self.alt:.0f}m", (dx * P + P, dy * P + 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, col, th, cv2.LINE_AA)

        side = np.zeros((V.FRAME_H, V.FRAME_W // 3, 3), np.uint8)
        if self.mask is not None:
            small = cv2.resize(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), (V.FRAME_W // 3, V.FRAME_H // 3))
            side[:small.shape[0]] = small
        lines = ["camera > detection > planner > viz", f"planlama mesaji: {self.count}",
                 f"planlama: {'3D, irtifa ' + format(self.alt, '.0f') + ' m' if is3d else '2D'}", "",
                 "kirmizi: engel  turuncu: pay", "mavi: bu irtifada ustunden gecilir",
                 "mor: hareketli engel gelecek", "sol tik: hedef  sag tik: drone", "Q: pencereyi kapat"]
        for i, t in enumerate(lines):
            cv2.putText(side, t, (8, V.FRAME_H // 3 + 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, V.C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(side, "maske (/obstacles/mask)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if is3d and self.drone:
            z = self._level(SimpleNamespace(position=SimpleNamespace(z=self.alt)))
            V.draw_profile(side, SimpleNamespace(path=self.path, height=self.height, drone=self.drone, z=z))
        out = np.hstack([view, side])
        cv2.rectangle(out, (0, 0), (V.FRAME_W, 26), (0, 0, 0), -1)
        cv2.putText(out, self.status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, V.C_TEXT, 1, cv2.LINE_AA)
        return out

    def tick(self):
        if self.frame is None:
            return
        out = self.draw()
        self.pub_img.publish(make_image(self.get_clock().now().to_msg(), out))
        if self.show:
            cv2.imshow(WIN, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                self.show = False


def main():
    spin(VizNode)
