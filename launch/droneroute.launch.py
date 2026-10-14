"""
Hepsini birden başlat:
  ros2 launch droneroute droneroute.launch.py                      # sentetik şantiye sahnesi
  ros2 launch droneroute droneroute.launch.py kaynak:=kamera       # webcam
  ros2 launch droneroute droneroute.launch.py kaynak:=video video:=/yol/dosya.mp4
  ros2 launch droneroute droneroute.launch.py tahmin:=false        # hareketli engel tahmini kapalı
  ros2 launch droneroute droneroute.launch.py goster:=false        # pencere yok (/viz/image yine yayınlanır)
  ros2 launch droneroute droneroute.launch.py uc_boyut:=false      # irtifasız (2D) planlama
  ros2 launch droneroute droneroute.launch.py kaynak:=video video:=/yol/ucus.mp4 yontem:=yolo perspektif:=true irtifa:=40.0 egim:=30.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as L
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = [  # (ad, varsayılan, tip, açıklama)
    ("kaynak", "sim", str, "sim | kamera | video"),
    ("video", "", str, "kaynak:=video ise dosya yolu"),
    ("kamera_id", "0", int, "webcam numarası"),
    ("seed", "-1", int, "sahne + hedef tohumu (-1 = rastgele)"),
    ("yontem", "renk", str, "algılama: renk | kenar | yolo | hibrit"),
    ("model", "", str, "YOLO ağırlıkları (boş = models/droneroute_yoloe.pt)"),
    ("guven", "0.25", float, "YOLO en düşük güven"),
    ("perspektif", "false", bool, "eğik kamera: tespitleri zemine indir (yontem:=yolo)"),
    ("irtifa", "25.0", float, "kamera yüksekliği (m)"),
    ("egim", "35.0", float, "kamera aşağı eğimi (derece, 90 = tepeden)"),
    ("fov", "82.0", float, "yatay görüş açısı (derece)"),
    ("hucre_m", "2.0", float, "eğik kamerada hücre boyu (m)"),
    ("kalibrasyon", "", str, "zemin kalibrasyonu JSON (boş = kameradan)"),
    ("uc_boyut", "true", bool, "irtifalı (3D) planlama"),
    ("pay", "1", int, "güvenlik payı (hücre)"),
    ("tahmin", "true", bool, "hareketli engel tahmini"),
    ("goster", "true", bool, "OpenCV penceresi"),
]


def p(name):
    typ = next(t for n, _, t, _ in ARGS if n == name)
    return {name: ParameterValue(L(name), value_type=typ)}


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, _, desc in ARGS] + [
            Node(package="droneroute", executable="camera_node", name="camera", output="screen",
                 parameters=[p("kaynak"), p("video"), p("kamera_id"), p("seed")]),
            Node(package="droneroute", executable="detection_node", name="detection", output="screen",
                 parameters=[p("yontem"), p("model"), p("guven"), p("perspektif"), p("irtifa"), p("egim"),
                             p("fov"), p("hucre_m"), p("kalibrasyon")]),
            Node(package="droneroute", executable="planner_node", name="planner", output="screen",
                 parameters=[p("uc_boyut"), p("pay"), p("tahmin"), p("seed")]),
            Node(package="droneroute", executable="viz_node", name="viz", output="screen",
                 parameters=[p("goster")]),
        ])
