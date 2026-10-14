"""
Planlama node'u: engel hücreleri + hareketli engel tahmini (+ boylar) → D* Lite rota

Girdi:
  /obstacles/grid, /obstacles/tti              (OccupancyGrid, aynı zaman damgasıyla eşleştirilir)
  /obstacles/height, /obstacles/tti_height     (sadece 3D'de; boylar, metre)
  /goal_pose    (geometry_msgs/PoseStamped)               yeni hedef  (RViz: "2D Goal Pose")
  /initialpose  (geometry_msgs/PoseWithCovarianceStamped)  drone'u ışınla (RViz: "2D Pose Estimate")
Çıktı:
  /plan/path    (nav_msgs/Path)            drone'dan hedefe rota (3D'de pose.z = uçuş irtifası, metre)
  /plan/costmap (nav_msgs/OccupancyGrid)   planlayıcının gördüğü dünya: 100 duvar, 1..99 ceza
                                           (3D'de drone'un bulunduğu irtifa katmanı)
  /drone/pose, /plan/goal (PoseStamped), /plan/status (std_msgs/String)

Algoritma ROS'suz modüllerde: uc_boyut:=true ise drone_plan3d.GridPlanner3D (irtifalı D* Lite), değilse
drone_vision.GridPlanner (2D). Diğer node'lar ikisinde de aynı.
Drone şimdilik simüle: her `adim_her` karede rotada bir düğüm ilerler. Gazebo/PX4'te bu kısım gerçek
konumla (odometri) değişecek.

Parametreler: uc_boyut, pay (güvenlik payı, hücre), tahmin (true/false), adim_her, otomatik_hedef, seed
"""
import random

import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from std_msgs.msg import String

import drone_vision as V
from drone_dstar3d import ALTS
from drone_plan3d import GridPlanner3D
from droneroute_ros import codec
from droneroute_ros.ros_util import FRAME_ID, make_grid, make_pose, spin


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner")
        self.is3d = self.declare_parameter("uc_boyut", True).value
        predict = self.declare_parameter("tahmin", True).value
        self.gp = GridPlanner3D(predict=predict) if self.is3d else V.GridPlanner(predict=predict)
        self.gp.margin = self.declare_parameter("pay", V.MARGIN).value
        self.move_every = self.declare_parameter("adim_her", V.MOVE_EVERY).value
        self.auto_goal = self.declare_parameter("otomatik_hedef", True).value
        seed = self.declare_parameter("seed", -1).value
        self.rng = random.Random(None if seed < 0 else seed)
        self.frames = 0

        self.pub_path = self.create_publisher(Path, "/plan/path", 5)
        self.pub_cost = self.create_publisher(OccupancyGrid, "/plan/costmap", 5)
        self.pub_drone = self.create_publisher(PoseStamped, "/drone/pose", 5)
        self.pub_goal = self.create_publisher(PoseStamped, "/plan/goal", 5)
        self.pub_status = self.create_publisher(String, "/plan/status", 5)
        topics = ["/obstacles/grid", "/obstacles/tti"]
        if self.is3d:
            topics += ["/obstacles/height", "/obstacles/tti_height"]
        self.sync = TimeSynchronizer([Subscriber(self, OccupancyGrid, t) for t in topics], 10)
        self.sync.registerCallback(self.on_obstacles)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 5)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 5)
        self.get_logger().info(f"{'3D' if self.is3d else '2D'} pay={self.gp.margin} tahmin={predict} "
                               f"adim_her={self.move_every}")

    def _cell(self, pose):
        return codec.xy_to_cell(pose.position.x, pose.position.y)

    def on_goal(self, msg):
        c = self._cell(msg.pose)
        if c != self.gp.drone:
            self.gp.set_goal(c)
            self.get_logger().info(f"yeni hedef {c}")

    def on_initialpose(self, msg):
        c = self._cell(msg.pose.pose)
        if c != self.gp.goal:
            self.gp.set_drone(c)
            self.get_logger().info(f"drone {c} hücresine taşındı")

    def on_obstacles(self, grid_msg, tti_msg, height_msg=None, tti_h_msg=None):
        raw = codec.data_to_flags(grid_msg.data)
        tti = codec.data_to_tti(tti_msg.data)
        if self.is3d:
            self.gp.update(raw, codec.data_to_height(height_msg.data), tti, codec.data_to_height(tti_h_msg.data))
        else:
            self.gp.update(raw, tti)
        self.frames += 1
        if self.frames % self.move_every == 0 and self.gp.step_drone() and self.auto_goal:
            nxt = V.random_free_cell(self.gp, self.rng, far_from=self.gp.drone)   # hedefe vardı, yenisini seç
            if nxt:
                self.gp.set_goal(nxt)
        self.publish(grid_msg.header.stamp)

    def publish(self, stamp):
        gp = self.gp
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, FRAME_ID
        if self.is3d:
            path.poses = [make_pose(stamp, (x, y), ALTS[z]) for x, y, z in (gp.path or [])]
            occ, pen = gp.occ3[gp.z], np.array(gp.penalty[gp.z])
            drone_alt, goal_alt = ALTS[gp.z], ALTS[gp.gz]
        else:
            path.poses = [make_pose(stamp, c) for c in (gp.path or [])]
            occ, pen = gp.occ, np.array(gp.penalty)
            drone_alt = goal_alt = 0.0
        self.pub_path.publish(path)
        self.pub_cost.publish(make_grid(stamp, codec.cost_to_data(occ, pen)))
        self.pub_drone.publish(make_pose(stamp, gp.drone, drone_alt))
        self.pub_goal.publish(make_pose(stamp, gp.goal, goal_alt))
        self.pub_status.publish(String(data=gp.status))


def main():
    spin(PlannerNode)
