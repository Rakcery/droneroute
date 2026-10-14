"""
ROS 2 paketi (colcon build). Klasörün kendisi paket: algoritma modülleri (drone_*.py) ROS'suz kalır,
droneroute_ros/ içindeki node'lar onları import eder. ROS'suz kullanım (python drone_vision.py) aynen çalışır.
"""
from glob import glob

from setuptools import setup

PKG = "droneroute"

setup(
    name=PKG,
    version="0.4.0",
    packages=["droneroute_ros"],
    py_modules=["drone_dstar_lite", "drone_dstar3d", "drone_plan3d", "drone_vision", "drone_tracker", "drone_yolo"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
        (f"share/{PKG}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="DroneRoute: kamera → algılama → D* Lite planlama ROS 2 node'ları",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = droneroute_ros.camera_node:main",
            "detection_node = droneroute_ros.detection_node:main",
            "planner_node = droneroute_ros.planner_node:main",
            "viz_node = droneroute_ros.viz_node:main",
        ],
    },
)
