"""drone_nofly.py: sanal yasak bölgeler (E-2, S-2). ROS'suz, saf Python."""
import numpy as np                            # sayı tabloları
from drone_dstar_lite import COLS, ROWS       # ızgara boyu: 40 sütun, 28 satır


def nofly_mask(zones, margin=0):
    """zones: [((x, y), yarıçap), ...] -> 28x40 bool tablo, True = yasak hücre."""
    yy, xx = np.mgrid[0:ROWS, 0:COLS]         # her hücrenin satır ve sütun numarası
    mask = np.zeros((ROWS, COLS), bool)       # başta hiçbir yer yasak değil
    for (cx, cy), r in zones:                 # her daire için
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + margin) ** 2   # daireyi ekle
    return mask                               # True = yasak hücre

NOFLY_PEN = 50.0                              # drone bölgenin içindeyse o bölgenin hücre cezası (birimsiz çarpan)

def nofly_split(zones, drone, margin=0):
    """Drone'un içinde olduğu daireler pahalı, diğerleri duvar -> (duvar, pahali) tabloları."""
    dx, dy = drone                            # drone'un hücresi (x, y)
    duvar = np.zeros((ROWS, COLS), bool)      # girilmez hücreler
    pahali = np.zeros((ROWS, COLS), bool)     # cezalı hücreler (drone'un içinde olduğu daireler)
    for (cx, cy), r in zones:                 # her daireye tek tek bak
        tek = nofly_mask([((cx, cy), r)], margin)   # sadece bu dairenin tablosu
        if tek[dy, dx]:                       # drone bu dairenin içinde mi?
            pahali |= tek                     # evet: duvar yapma, pahalı yap (çıkabilsin)
        else:
            duvar |= tek                      # hayır: duvar
    return duvar, pahali                      # iki tablo