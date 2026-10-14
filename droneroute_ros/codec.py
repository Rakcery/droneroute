"""
Grid <-> mesaj verisi dönüşümleri. ROS import etmez (numpy yeter), o yüzden ROS'suz test edilebilir.

Koordinatlar: bizim grid'de hücre (x, y), y aşağı doğru (görüntü gibi). ROS'ta harita çerçevesi
("map") y yukarı. nav_msgs/OccupancyGrid verisi satır satır, ilk satır haritanın en altı (origin).
Bu yüzden grid'i ters satır sırasıyla yazıyoruz: kamera görüntüsündeki üst satır haritada en üstte.
Bir hücre RESOLUTION metre. 40x28 hücre = 20 m x 14 m'lik alan.

Konular (topic) ve içerikleri:
  /obstacles/grid  OccupancyGrid  100 = şu an engel, 0 = boş            (algılama → planlama)
  /obstacles/tti   OccupancyGrid  -1 = hareketli engel gelmiyor,          (algılama → planlama)
                                  0..100 = TTI_SCALE * "kaç kare sonra geliyor"
  /plan/costmap    OccupancyGrid  100 = duvar (pay + yakında gelecek), 1..99 = ceza, 0 = boş   (planlama → görsel)
                                  3D'de drone'un o anki irtifa katmanı
  /obstacles/height      OccupancyGrid  engelin boyu (metre, yukarı yuvarlanır), 0 = boş, -1 = bilinmiyor (3D)
  /obstacles/tti_height  OccupancyGrid  /obstacles/tti'deki hücreye gelecek engelin boyu, aynı kodlama  (3D)
  /camera/height   Image 32FC1    piksel başına zeminden yükseklik (metre), derinlik kamerası yerine (sim)
"""
import math

import numpy as np

from drone_dstar_lite import COLS, ROWS

RESOLUTION = 0.5     # metre / hücre
TTI_SCALE = 5        # tti 0.2 kare hassasiyetle; 20 kare ufuk → 0..100
COST_SCALE = 20      # ceza (0..4) → 0..80


def flags_to_data(arr):
    """ROWS x COLS bool → OccupancyGrid.data (100 dolu, 0 boş)."""
    return np.where(arr[::-1], 100, 0).astype(np.int8).ravel().tolist()


def data_to_flags(data, width=COLS, height=ROWS):
    return (np.asarray(data, np.int8).reshape(height, width)[::-1] >= 50).copy()


def tti_to_data(tti):
    """
    ROWS x COLS float (inf = gelmiyor) → int8 listesi. Aşağı yuvarlıyoruz: kodlama yüzünden engel
    en fazla 0.2 kare *erken* gelmiş sayılır, asla geç değil (temkinli taraf).
    """
    v = np.where(np.isfinite(tti), np.floor(np.minimum(tti, 100 / TTI_SCALE) * TTI_SCALE), -1)
    return v[::-1].astype(np.int8).ravel().tolist()


def data_to_tti(data, width=COLS, height=ROWS):
    v = np.asarray(data, np.int8).reshape(height, width)[::-1].astype(np.float64)
    return np.where(v < 0, np.inf, v / TTI_SCALE)


def cost_to_data(occ, pen):
    """Planlayıcının gördüğü dünya: duvar 100, cezalı hücre 1..99, boş 0 (RViz'de 'costmap' renk şemasıyla bakılabilir)."""
    v = np.where(occ, 100, np.where(pen > 0, np.clip(np.ceil(pen * COST_SCALE), 1, 99), 0))
    return v[::-1].astype(np.int8).ravel().tolist()


def data_to_cost(data, width=COLS, height=ROWS):
    v = np.asarray(data, np.int8).reshape(height, width)[::-1]
    return v >= 100, np.where(v >= 100, 0.0, v.astype(np.float64) / COST_SCALE)


def height_to_data(h):
    """ROWS x COLS metre (inf = bilinmiyor) → int8. Yukarı yuvarla: engel en fazla 1 m uzun sayılır, asla kısa değil."""
    v = np.where(np.isfinite(h), np.ceil(np.clip(h, 0, 100)), -1)
    return v[::-1].astype(np.int8).ravel().tolist()


def data_to_height(data, width=COLS, height=ROWS):
    v = np.asarray(data, np.int8).reshape(height, width)[::-1].astype(np.float64)
    return np.where(v < 0, np.inf, v)


def cell_to_xy(cell, res=RESOLUTION):
    """Hücre merkezinin harita koordinatı (metre)."""
    x, y = cell
    return (x + 0.5) * res, (ROWS - 1 - y + 0.5) * res


def xy_to_cell(wx, wy, res=RESOLUTION):
    """Harita koordinatı → hücre (kenara kırpılır)."""
    x = min(max(int(math.floor(wx / res)), 0), COLS - 1)
    y = ROWS - 1 - min(max(int(math.floor(wy / res)), 0), ROWS - 1)
    return x, y


def image_to_array(height, width, encoding, data):
    """sensor_msgs/Image → BGR numpy (cv_bridge gerekmesin diye elle)."""
    buf = np.frombuffer(bytes(data), np.uint8)
    if encoding in ("bgr8", "rgb8"):
        img = buf.reshape(height, width, 3)
        return img[:, :, ::-1].copy() if encoding == "rgb8" else img
    if encoding == "mono8":
        return buf.reshape(height, width)
    if encoding == "32FC1":
        return buf.view(np.float32).reshape(height, width)
    raise ValueError(f"desteklenmeyen görüntü kodlaması: {encoding} (bgr8, rgb8, mono8, 32FC1 olmalı)")
