"""
DroneRoute — Adım 2: OpenCV ile engel algılama + D* Lite canlı yeniden planlama
Gereken: pip install -r requirements.txt   (opencv-python, numpy, pygame)

Akış (her karede):
  kamera karesi ──► engel maskesi ──► hücre doluluğu ──► zaman filtresi ──┐
                         │                                                  ├─► güvenlik payı ──► grid + ceza
                         └──► takip (drone_tracker.py) ──► "engel kaç kare ──┘                        │
                              sonra nerede" tahmini                                                    │
              D* Lite sadece değişen hücreleri alır, rotayı artımlı günceller  ◄──────────────────────┘

Hareketli engeller (Adım 2.5): kamyon, vinç kolu, kuş takip edilir; yakında gelecekleri hücreler duvar,
biraz daha sonra gelecekleri pahalı sayılır. Rotası kapanan drone olduğu yerde ezilmeyi beklemez,
kenara çekilir.

ROS 2 (Adım 3): Perception (algılama + takip) ve GridPlanner (planlama) ayrı sınıflar; ROS 2'de ayrı
node'lar olarak koşuyorlar (droneroute_ros/). VisionPlanner ikisini tek nesnede birleştirir (bu pencere için).

Kamera yukarıdan bakıyor (top-down) varsayımı var: görüntünün her pikseli zemindeki bir noktaya denk.
Engel = "zemine benzemeyen" şey. Klasik iki algılama yöntemi (M tuşu ile değişir; YOLO yöntemleri için drone_yolo.py):
  renk  : Boş zeminin renk histogramını öğren (kalibrasyon), her pikselin zemine benzerliğine bak
          (histogram back-projection). Zemine benzemeyen = engel. Simülasyonda ve masa üstünde iyi çalışır.
  kenar : Canny kenarları + kapatma + kontur doldurma. Kalibrasyon istemez ama dokulu zeminde gürültülü.

OpenCV sadece görüntü işleme ilkelleri için (blur, HSV, histogram, Canny, morfoloji, kontur).
Rota hesaplama, takip/tahmin, güvenlik payı ve zaman filtresi kendi kodumuz.

Kaynak:
  python drone_vision.py                     # sentetik kuşbakışı şantiye sahnesi (kamera gerekmez)
  python drone_vision.py --kamera            # webcam (açılmazsa simülasyona düşer)
  python drone_vision.py --video dosya.mp4   # kayıtlı video
  python drone_vision.py --headless 600      # pencere açmadan 600 kare koş, istatistik bas, PNG kaydet
  python drone_vision.py --3d                # irtifalı planlama (drone_plan3d.py, 3D D* Lite)
  python drone_vision.py --yontem yolo --model sim --perspektif   # Adım 5: YOLO + eğik kamera (drone_yolo.py)

Kontroller (pencere seçiliyken):
  Sol tık  : hedefi oraya taşı          Sağ tık : drone'u oraya ışınla
  M        : algılama yöntemi (renk/kenar/yolo/hibrit)
  C        : zemin kalibrasyonu (webcam'de önce boş zemini göster, sonra C)
  I        : güvenlik payını aç/kapat    + / -   : payı büyüt/küçült (hücre)
  T        : hareketli engel tahminini aç/kapat (kapalıyken kamyonun drone'u ezdiğini görürsün)
  SPACE    : duraklat   N : simülasyonda yeni sahne   Q / ESC : çık

Not: OpenCV'nin yazı fontu Türkçe karakter basamıyor, ekrandaki yazılar bu yüzden ş/ğ/ı'sız.
"""
import argparse
import math
import os
import random
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np

import drone_dstar_lite as dl
import drone_tracker as tracking
from drone_dstar_lite import COLS, ROWS, DStarLite, path_cost

# ---------------- Ayarlar ----------------
CELL_PX = 22                       # ekranda bir hücre kaç piksel
FRAME_W, FRAME_H = COLS * CELL_PX, ROWS * CELL_PX   # her kare bu boyuta getirilir (880x616)
OCC_THRESH = 0.20                  # hücrenin en az %20'si engelse hücre dolu
MIN_BLOB_AREA = 60                 # bundan küçük lekeler gürültü sayılır (piksel)
BACKPROJ_THRESH = 40               # renk yönteminde zemine benzerlik eşiği (0-255)
K_ON, K_OFF = 2, 4                 # hücre dolu sayılması için 2 kare üst üste, boşalması için 4 kare
MARGIN = 1                         # güvenlik payı (hücre). Drone vinç/ağacın dibinden geçmesin
MAX_MARGIN = 4
MOVE_EVERY = 4                     # drone kaç karede bir adım atsın
H_HARD = 8                         # hareketli engelin bu kadar kare içinde gireceği hücre: duvar
PEN_MAX = 4.0                      # H_HARD..HORIZON arası gelecek hücreler: en fazla 5 kat pahalı
EDGE_BAND, EDGE_PEN = 2, 2.0       # kadraj kenarındaki 2 hücre 3 kat pahalı: dışarıdan ne gireceğini göremeyiz
UNSEEN_PEN = 4.0                   # eğik kamera: görüş alanı dışı 5 kat pahalı (orada bina olabilir, bilmiyoruz)

# BGR renkler
C_RAW = (40, 40, 230)
C_INFL = (0, 150, 255)
C_PATH = (255, 220, 60)
C_DRONE = (40, 230, 255)
C_GOAL = (90, 230, 90)
C_TEXT = (240, 240, 240)
C_PRED = (200, 60, 200)            # hareketli engelin yakında geleceği hücreler (mor)
C_TRACK = (255, 255, 255)
C_LOW = (170, 120, 40)             # 3D: drone'un irtifasında üstünden geçilebilen engel (mavi)
C_LEVELS = [(255, 220, 60), (80, 230, 120), (40, 200, 255), (220, 90, 255)]   # 3D rota rengi: 5/10/15/20 m (mavi, yeşil, turuncu, pembe)


# ---------------- Sentetik sahne (kuşbakışı şantiye) ----------------
class SimScene:
    """
    Kamera olmadan test için kuşbakışı sahne: çimen + toprak yol (zemin), binalar, ağaçlar,
    dönen bir vinç kolu, yolda giden bir kamyon ve sahnede uçup kenarlardan seken bir kuş. Her kare hafif kamera gürültüsü eklenir.
    render() kareyle birlikte gerçek engel maskesini (ground truth) de döner; testler bunu kullanır.
    Her engelin bir yüksekliği var (metre); render() sonrası self.height_map derinlik kamerasının
    vereceği "zeminden yükseklik" görüntüsü (piksel başına metre, zemin 0). 3D planlama bunu kullanır.
    """
    H_TRUCK, H_BIRD, H_JIB, H_MAST = 4.0, 12.0, 35.0, 45.0   # metre

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.t = 0
        self._make_ground()
        self._make_static()
        # Yükseklikler ayrı rastgele üreteçten: eski sahneler (ve testleri) aynen kalsın
        hr = random.Random(None if seed is None else seed * 7919 + 1)
        self.building_h = [hr.uniform(8, 28) for _ in self.buildings]
        self.tree_h = [hr.uniform(6, 14) for _ in self.trees]
        self.static_height = np.zeros((FRAME_H, FRAME_W), np.float32)
        order = sorted([(h, "b", i) for i, h in enumerate(self.building_h)] + [(h, "t", i) for i, h in enumerate(self.tree_h)])
        for h, kind, i in order:                     # alçaktan yükseğe çiz: üst üste binen yerde yüksek olan kalır
            if kind == "b":
                x, y, w, hh, _ = self.buildings[i]
                cv2.rectangle(self.static_height, (x, y), (x + w, y + hh), float(h), -1)
            else:
                x, y, rad = self.trees[i]
                cv2.circle(self.static_height, (x, y), rad, float(h), -1)
        self.height_map = self.static_height.copy()
        r = self.rng
        self.crane_mast = (int(FRAME_W * r.uniform(0.6, 0.72)), int(FRAME_H * r.uniform(0.3, 0.7)))
        self.crane_len = r.randint(150, 210)
        self.crane_angle = r.uniform(0, 2 * math.pi)
        self.crane_speed = r.choice((-1, 1)) * r.uniform(0.008, 0.015)   # rad / kare
        self.truck_y = r.uniform(0, FRAME_H)
        self.truck_speed = r.uniform(2.0, 3.5)                            # piksel / kare
        # Kuş (ya da başka bir drone): sahnede düz uçar, kenara çarpınca seker
        self.bird = [r.uniform(60, FRAME_W - 60), r.uniform(60, FRAME_H - 60)]
        a, sp = r.uniform(0, 2 * math.pi), r.uniform(1.5, 2.5)
        self.bird_v = [sp * math.cos(a), sp * math.sin(a)]

    def _make_ground(self):
        noise = self.np_rng.normal(0, 1, (FRAME_H // 4, FRAME_W // 4)).astype(np.float32)
        noise = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.5), (FRAME_W, FRAME_H))
        grass = np.array([70, 150, 95], np.float32)
        g = np.clip(grass + noise[..., None] * np.array([8, 14, 8], np.float32), 0, 255)
        self.ground = g.astype(np.uint8)
        # Toprak yollar da zemin: drone üstlerinden uçabilir
        self.road_x = int(FRAME_W * self.rng.uniform(0.38, 0.5))
        cv2.rectangle(self.ground, (self.road_x - 22, 0), (self.road_x + 22, FRAME_H), (95, 135, 165), -1)
        ry = int(FRAME_H * self.rng.uniform(0.15, 0.3))
        cv2.rectangle(self.ground, (0, ry - 14), (FRAME_W, ry + 14), (95, 135, 165), -1)

    def _make_static(self):
        """Binalar ve ağaçlar; drone başlangıcı ve hedefin etrafını boş bırak."""
        r = self.rng
        keep_free = [(CELL_PX * 1.5, FRAME_H / 2), (FRAME_W - CELL_PX * 1.5, FRAME_H / 2)]
        self.buildings, self.trees = [], []

        def clear(x, y, rad):
            ok = all(math.hypot(x - fx, y - fy) > rad + 70 for fx, fy in keep_free)
            return ok and abs(x - self.road_x) > rad + 26  # yolun üstüne bina koyma

        while len(self.buildings) < 6:
            w, h = r.randint(50, 120), r.randint(40, 110)
            x, y = r.randint(0, FRAME_W - w), r.randint(0, FRAME_H - h)
            if clear(x + w / 2, y + h / 2, max(w, h) / 2):
                roof = r.choice([(85, 85, 90), (60, 70, 170), (120, 110, 100)])
                self.buildings.append((x, y, w, h, roof))
        while len(self.trees) < 10:
            rad = r.randint(12, 22)
            x, y = r.randint(rad, FRAME_W - rad), r.randint(rad, FRAME_H - rad)
            if clear(x, y, rad):
                self.trees.append((x, y, rad))

    def background(self):
        """Engelsiz zemin karesi: renk yönteminin kalibrasyonu için."""
        return self._noisy(self.ground.copy())

    def _noisy(self, img):
        # Kamera gürültüsü: her karede yeni gürültü üretmek yavaş, önceden 8 tane hazırlayıp seçiyoruz
        if not hasattr(self, "_noise"):
            self._noise = []
            for _ in range(8):
                n = self.np_rng.normal(0, 4, img.shape)
                self._noise.append((np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8)))
        pos, neg = self._noise[self.rng.randrange(8)]
        out = cv2.convertScaleAbs(img, alpha=self.rng.uniform(0.95, 1.05))   # ufak pozlama oynaması
        return cv2.subtract(cv2.add(out, pos), neg)

    def render(self):
        img = self.ground.copy()
        gt = np.zeros((FRAME_H, FRAME_W), np.uint8)
        for x, y, w, h, roof in self.buildings:
            for canvas, col in ((img, roof), (gt, 255)):
                cv2.rectangle(canvas, (x, y), (x + w, y + h), col, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 2)
            cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), (40, 40, 40), 1)   # çatı mahyası
        for x, y, rad in self.trees:
            cv2.circle(img, (x, y), rad, (30, 75, 35), -1)
            cv2.circle(img, (x - rad // 3, y - rad // 3), rad // 2, (40, 100, 45), -1)
            cv2.circle(gt, (x, y), rad, 255, -1)
        # Kamyon: yolda aşağı yukarı gidiyor
        tx, ty = self.road_x, int(self.truck_y)
        for canvas, col in ((img, (20, 120, 240)), (gt, 255)):
            cv2.rectangle(canvas, (tx - 13, ty - 26), (tx + 13, ty + 26), col, -1)
        cv2.rectangle(img, (tx - 13, ty + 8), (tx + 13, ty + 26), (200, 200, 210), -1)  # kasa
        # Vinç: kule + dönen kol + karşı ağırlık
        mx, my = self.crane_mast
        a = self.crane_angle
        tip = (int(mx + math.cos(a) * self.crane_len), int(my + math.sin(a) * self.crane_len))
        back = (int(mx - math.cos(a) * 55), int(my - math.sin(a) * 55))
        for canvas, col in ((img, (0, 200, 240)), (gt, 255)):
            cv2.line(canvas, back, tip, col, 12)
        cv2.rectangle(img, (back[0] - 12, back[1] - 12), (back[0] + 12, back[1] + 12), (60, 60, 60), -1)
        cv2.rectangle(gt, (back[0] - 12, back[1] - 12), (back[0] + 12, back[1] + 12), 255, -1)
        cv2.rectangle(img, (mx - 14, my - 14), (mx + 14, my + 14), (0, 160, 200), -1)
        cv2.rectangle(gt, (mx - 14, my - 14), (mx + 14, my + 14), 255, -1)
        # Kuş: gövde + kanatlar
        bx, by = int(self.bird[0]), int(self.bird[1])
        vx, vy = self.bird_v
        n = math.hypot(vx, vy)
        px, py = -vy / n, vx / n                                    # uçuş yönüne dik
        cv2.line(img, (int(bx - px * 11), int(by - py * 11)), (int(bx + px * 11), int(by + py * 11)), (90, 90, 100), 6)
        cv2.circle(img, (bx, by), 7, (40, 40, 45), -1)
        cv2.circle(gt, (bx, by), 12, 255, -1)
        # Yükseklik haritası (derinlik kamerası): sabitler hazır, hareketlileri üstüne en büyüğü al
        hm = self.static_height.copy()
        mov = np.zeros_like(hm)
        cv2.rectangle(mov, (tx - 13, ty - 26), (tx + 13, ty + 26), self.H_TRUCK, -1)
        np.maximum(hm, mov, out=hm)
        mov[:] = 0
        cv2.circle(mov, (bx, by), 12, self.H_BIRD, -1)
        np.maximum(hm, mov, out=hm)
        mov[:] = 0
        cv2.line(mov, back, tip, self.H_JIB, 12)
        cv2.rectangle(mov, (back[0] - 12, back[1] - 12), (back[0] + 12, back[1] + 12), self.H_JIB, -1)
        cv2.rectangle(mov, (mx - 14, my - 14), (mx + 14, my + 14), self.H_MAST, -1)
        np.maximum(hm, mov, out=hm)
        self.height_map = hm
        self.objs = self.objects()
        return self._noisy(img), gt

    def objects(self):
        """
        Sahnedeki nesneler (YOLO modunun sahte dedektörü drone_yolo.SimOracle ve eğik kamera görüntüsü için):
        her biri {"cls": sınıf, "parts": [(taban poligonu [piksel], alt boy, üst boy [m], BGR renk), ...]}.
        """
        rect = lambda x1, y1, x2, y2: np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
        out = [dict(cls="bina", parts=[(rect(x, y, x + w, y + h), 0.0, bh, roof)])
               for (x, y, w, h, roof), bh in zip(self.buildings, self.building_h)]
        out += [dict(cls="agac", parts=[(cv2.ellipse2Poly((x, y), (rad, rad), 0, 0, 360, 30), 0.0, th, (30, 75, 35))])
                for (x, y, rad), th in zip(self.trees, self.tree_h)]
        tx, ty = self.road_x, int(self.truck_y)
        out.append(dict(cls="arac", parts=[(rect(tx - 13, ty - 26, tx + 13, ty + 26), 0.0, self.H_TRUCK, (20, 120, 240))]))
        mx, my = self.crane_mast
        a = self.crane_angle
        tip = (mx + math.cos(a) * self.crane_len, my + math.sin(a) * self.crane_len)
        back = (mx - math.cos(a) * 55, my - math.sin(a) * 55)
        jib = cv2.boxPoints((((tip[0] + back[0]) / 2, (tip[1] + back[1]) / 2), (self.crane_len + 55, 12), math.degrees(a)))
        bx, by = int(back[0]), int(back[1])
        out.append(dict(cls="vinc", parts=[
            (rect(mx - 14, my - 14, mx + 14, my + 14), 0.0, self.H_MAST, (0, 160, 200)),
            (np.round(jib).astype(np.int32), self.H_JIB - 2, self.H_JIB, (0, 200, 240)),
            (rect(bx - 12, by - 12, bx + 12, by + 12), self.H_JIB - 3, self.H_JIB, (60, 60, 60))]))
        out.append(dict(cls="kus", parts=[(cv2.ellipse2Poly((int(self.bird[0]), int(self.bird[1])), (12, 12), 0, 0, 360, 30),
                                           self.H_BIRD - 1, self.H_BIRD + 1, (40, 40, 45))]))
        return out

    def footprint_below(self, z):
        """Son karede tabanı z metrenin altında olan nesnelerin kuşbakışı maskesi (vinç kolu 33 m: altından geçilir)."""
        m = np.zeros((FRAME_H, FRAME_W), np.uint8)
        for o in self.objs:
            for poly, z0, _, _ in o["parts"]:
                if z0 < z:
                    cv2.fillPoly(m, [poly], 255)
        return m

    def footprint_at(self, alt):
        """Son karede alt metre irtifayı (±2 m pay) kesen nesne parçalarının kuşbakışı maskesi."""
        m = np.zeros((FRAME_H, FRAME_W), np.uint8)
        for o in self.objs:
            for poly, z0, z1, _ in o["parts"]:
                if z0 - 2 <= alt <= z1 + 2:
                    cv2.fillPoly(m, [poly], 255)
        return m

    def step(self):
        self.t += 1
        self.crane_angle += self.crane_speed
        self.truck_y = (self.truck_y + self.truck_speed) % (FRAME_H + 60)
        for i, lim in ((0, FRAME_W), (1, FRAME_H)):
            self.bird[i] += self.bird_v[i]
            if not 14 <= self.bird[i] <= lim - 14:
                self.bird_v[i] = -self.bird_v[i]
                self.bird[i] = min(max(self.bird[i], 14), lim - 14)


# ---------------- Kaynaklar ----------------
class Source:
    def __init__(self, args):
        self.sim, self.cap, self.first = None, None, None
        self.cam = None                                # eğik kamera (--perspektif): sim bu kameradan çizilir
        if args.video or args.kamera:
            self.cap = cv2.VideoCapture(args.video if args.video else args.kamera_id)
            ok, self.first = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                print(f"[!] {'Video' if args.video else 'Kamera'} açılamadı, simülasyona geçiyorum.")
                self.cap = None
        if self.cap is None:
            self.sim = SimScene(args.seed)
        self.name = "sim" if self.sim else ("video" if args.video else "kamera")
        if getattr(args, "perspektif", False) and self.sim:
            import drone_yolo
            self.cam = drone_yolo.Camera(args.irtifa, args.egim, args.fov, args.vfov)
            self.cell_m = args.hucre_m
            self.noise = np.random.default_rng(args.seed)

    def height(self):
        """Son okunan karenin yükseklik haritası (metre). Sadece kuşbakışı simülasyonda var; kamera/videoda None."""
        return self.sim.height_map if self.sim and self.cam is None else None

    def read(self):
        """(kare, gerçek_maske ya da None) döner. Gerçek maske hep kuşbakışı (grid ile aynı)."""
        if self.sim:
            frame, gt = self.sim.render()
            if self.cam is not None:
                import drone_yolo
                frame = drone_yolo.render_oblique(self.sim, self.cam, self.cell_m, self.noise)
            self.sim.step()
            return frame, gt
        if self.first is not None:                     # açılışta test için okunan kareyi kaybetme
            frame, self.first = self.first, None
            return cv2.resize(frame, (FRAME_W, FRAME_H)), None
        ok, frame = self.cap.read()
        if not ok and self.name == "video":           # video bitti, başa sar
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return None, None
        return cv2.resize(frame, (FRAME_W, FRAME_H)), None


# ---------------- Algılama ----------------
class ObstacleDetector:
    """
    Kare → engel maskesi (0/255, kuşbakışı) → hücre doluluğu (ROWS x COLS bool).
    yolo  : drone_yolo.YoloDetector (sınıf + boy; eğik kamerada zemine indirme). Verilmezse ilk kullanımda
            varsayılan model yüklenir. self.last_height = piksel başına engel boyu (metre).
    hibrit: yolo + renk (renk kalibre değilse kenar). YOLO'nun tanımadığı engel de kaçmasın; onun boyu bilinmiyor.
    """
    METHODS = ("renk", "kenar", "yolo", "hibrit")

    def __init__(self, method="renk", yolo=None):
        self.method = method
        self.ground_hist = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.yolo = yolo
        self.last_height = None

    BINS = (30, 32, 16)            # H, S, V kutu sayısı

    # Kanal değeri → düz kutu indeksine katkısı (her piksele bölme yapmamak için hazır tablo)
    _LUT_H = (np.arange(256) * BINS[0] // 180 * BINS[1] * BINS[2]).astype(np.int32)
    _LUT_S = (np.arange(256) * BINS[1] // 256 * BINS[2]).astype(np.int32)
    _LUT_V = (np.arange(256) * BINS[2] // 256).astype(np.int32)

    @classmethod
    def _bin_index(cls, frame):
        """Her pikselin H-S-V histogram kutusunun düz (flat) indeksi."""
        h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        return cls._LUT_H[h] + cls._LUT_S[s] + cls._LUT_V[v]

    def calibrate(self, ground_frame):
        """Boş zeminin H-S-V histogramını öğren. Sonra her piksele 'bu zemin mi?' diye soracağız."""
        idx = self._bin_index(cv2.GaussianBlur(ground_frame, (5, 5), 0))
        hist = np.bincount(idx.ravel(), minlength=int(np.prod(self.BINS))).astype(np.float32)
        hist = hist.reshape(self.BINS)
        # Komşu kutulara biraz yay: kalibrasyonda tam görülmemiş ama çok yakın renkler de zemin sayılsın.
        # (Ton dairesel, o yüzden H ekseninde roll; S ve V'de kenardan taşanı sıfırla.)
        for axis in range(3):
            fwd, bwd = np.roll(hist, 1, axis), np.roll(hist, -1, axis)
            if axis:
                sl = [slice(None)] * 3
                sl[axis] = 0
                fwd[tuple(sl)] = 0
                sl[axis] = -1
                bwd[tuple(sl)] = 0
            hist = hist + 0.5 * (fwd + bwd)
        # En sık kutuya göre normalize etseydik az yer kaplayan zemin renkleri (ince toprak yol gibi)
        # "zemin değil" çıkardı. Onun yerine: kutuya pikselin %0.2'si düşüyorsa tam zemin (255).
        frac = hist / idx.size
        self.ground_hist = np.clip(255.0 * frac / 0.002, 0, 255).astype(np.uint8).ravel()

    def mask(self, frame):
        self.last_height = None
        if self.method not in ("yolo", "hibrit"):
            return self._classic(frame, self.method)
        if self.yolo is None:
            import drone_yolo
            self.yolo = drone_yolo.YoloDetector(drone_yolo.UltralyticsBackend())
        m, h = self.yolo.process(frame)
        if self.method == "hibrit" and not self.yolo.perspective:   # renk yöntemi eğik kamerada anlamsız
            c = self._classic(frame, "renk" if self.ground_hist is not None else "kenar")
            h = np.where(m > 0, h, np.inf)
            m = cv2.bitwise_or(m, c)
        self.last_height = h
        return m

    def _classic(self, frame, method):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        if method == "renk" and self.ground_hist is not None:
            # Back-projection: her pikselin rengi zemin histogramında ne kadar sık? (tablo bakışı)
            prob = self.ground_hist[self._bin_index(blur)]
            prob = cv2.GaussianBlur(prob, (7, 7), 0)
            m = np.where(prob < BACKPROJ_THRESH, 255, 0).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)     # tek tük gürültü pikselleri sil
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)    # engelin içindeki delikleri kapat
        else:
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 140)
            m = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # Konturları doldur, küçük lekeleri at
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(m)
        big = [c for c in contours if cv2.contourArea(c) >= MIN_BLOB_AREA]
        cv2.drawContours(out, big, -1, 255, -1)
        return out

    @staticmethod
    def to_cells(mask):
        """Her hücredeki engel piksel oranı → eşikten büyükse dolu."""
        frac = cv2.resize(mask, (COLS, ROWS), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return frac >= OCC_THRESH


# ---------------- Zaman filtresi + güvenlik payı (kendi kodumuz) ----------------
class OccupancyFilter:
    """
    Tek karelik titremeler D* Lite'ı boş yere yeniden planlatmasın diye histerezis:
    hücre K_ON kare üst üste dolu görülürse dolu, K_OFF kare üst üste boş görülürse boş olur.
    """

    def __init__(self):
        self.state = np.zeros((ROWS, COLS), bool)
        self.on = np.zeros((ROWS, COLS), np.int32)
        self.off = np.zeros((ROWS, COLS), np.int32)

    def update(self, raw):
        self.on = np.where(raw, self.on + 1, 0)
        self.off = np.where(raw, 0, self.off + 1)
        self.state = np.where(self.on >= K_ON, True, np.where(self.off >= K_OFF, False, self.state))
        return self.state


def _shifts(r):
    """Yarıçapı r olan diskin (0,0) hariç ofsetleri ve onlara ait (kaynak, hedef) dilimleri."""
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx, dy) == (0, 0) or dx * dx + dy * dy > r * r:
                continue
            ys, yd = (slice(0, ROWS - dy), slice(dy, ROWS)) if dy >= 0 else (slice(-dy, ROWS), slice(0, ROWS + dy))
            xs, xd = (slice(0, COLS - dx), slice(dx, COLS)) if dx >= 0 else (slice(-dx, COLS), slice(0, COLS + dx))
            yield (ys, xs), (yd, xd)


def spread_min(t, r):
    """
    inflate'in sayısal hali: her hücre, r yarıçaplı komşuluğundaki en küçük değeri alır.
    'Engel bu hücreye kaç kare sonra geliyor' haritasını güvenlik payı kadar yaymak için.
    """
    out = t.copy()
    if r > 0:
        for src, dst in _shifts(r):
            np.minimum(out[dst], t[src], out=out[dst])
    return out


def inflate(occ, r):
    """
    Engelleri r hücre şişir (dairesel). Her dolu hücreyi dx²+dy² <= r² içindeki ofsetlerle kaydırıp
    OR'luyoruz. Böylece rota engelden en az r hücre uzak kalır.
    """
    out = occ.copy()
    if r > 0:
        for src, dst in _shifts(r):
            out[dst] |= occ[src]
    return out


def edge_band(unseen=None):
    """Kenar şeridi (+ eğik kamerada görüş alanı dışı): dışarıdan ne gireceğini göremediğimiz, pahalı hücreler."""
    band = np.ones((ROWS, COLS), bool)
    band[EDGE_BAND:ROWS - EDGE_BAND, EDGE_BAND:COLS - EDGE_BAND] = False
    return band if unseen is None else band | unseen


# ---------------- Algılama + takip (ROS'ta: algılama node'u) ----------------
class Perception:
    """
    Kare → (raw, moving_tti). Planlamaya dair hiçbir karar burada yok, sadece "ne nerede, ne zaman":
      raw        : ROWS x COLS bool, zaman filtresinden geçmiş şu anki engel hücreleri
      moving_tti : ROWS x COLS float, hareketli bir engelin o hücreye kaç kare sonra gireceği (yoksa inf)
    Yükseklik görüntüsü verilirse (derinlik kamerası, 3D planlama için) ayrıca:
      self.height        : raw hücrelerdeki engelin yüksekliği (metre); bilinmiyorsa inf, boşsa 0
      self.moving_height : moving_tti'deki hücreye gelecek engelin yüksekliği (yoksa 0, bilinmiyorsa inf)
    Yükseklik yoksa (webcam) her engel sonsuz yüksek sayılır: üstünden uçulmaz, 3D planlama 2D gibi davranır.
    YOLO yönteminde derinlik kamerası yoksa boylar algılayıcıdan gelir (sınıftan ya da eğik kameradan tahmin).
    """

    def __init__(self, detector):
        self.det = detector
        self.filter = OccupancyFilter()
        self.tracker = tracking.Tracker((FRAME_H, FRAME_W))
        self.last_mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
        self.height = np.zeros((ROWS, COLS))
        self.moving_height = np.zeros((ROWS, COLS))

    @property
    def unseen(self):
        """Eğik kamerada görüş alanı dışındaki hücreler (orada ne olduğunu bilmiyoruz); kuşbakışında None."""
        y = getattr(self.det, "yolo", None)
        return y.projector.unseen if y is not None and y.perspective and self.det.method in ("yolo", "hibrit") else None

    def process(self, frame, height=None):
        self.last_mask = self.det.mask(frame)
        if height is None:
            height = getattr(self.det, "last_height", None)
        raw = self.filter.update(self.det.to_cells(self.last_mask))
        moving = self.tracker.update(self.last_mask, height)   # takip her zaman çalışsın (çizim için)
        self.moving_height = self.tracker.tti_height
        if height is None:
            self.height = np.where(raw, np.inf, 0.0)
        else:
            # Hücredeki engel piksellerinin en yükseği. Zaman filtresi hücreyi birkaç kare daha dolu
            # tutabilir (engel gitti ama "boş" onaylanmadı): o zaman son ölçülen yüksekliği koru.
            hm = np.where(self.last_mask > 0, height, 0).reshape(ROWS, CELL_PX, COLS, CELL_PX).max(axis=(1, 3))
            prev = np.where(self.height > 0, self.height, np.inf)
            self.height = np.where(raw, np.where(hm > 0, hm, prev), 0.0)
        return raw, moving


# ---------------- Planlama (ROS'ta: planlama node'u) ----------------
class GridPlanner:
    """
    (raw, moving_tti) → grid + ceza → sadece değişen hücreler D* Lite'a → rota, drone'u yürütür.
    Hareketli engeller için tti = "engel kaç kare sonra burada":
      tti <= H_HARD          : duvar (yakında gelecek, oraya girme)
      H_HARD < tti <= HORIZON: ceza (geçebilirsin ama pahalı, mümkünse dolan)
    Drone bulunduğu hücreye bir şey geliyorsa ve rota yoksa bekleyip ezilmez, kenara çekilir.
    Görüntüden habersiz: aynı sınıfı hem drone_vision.py hem ROS 2 planlama node'u kullanır.
    """

    def __init__(self, start=None, goal=None, predict=True):
        self.predict = predict
        self.margin = MARGIN
        self.grid = [[False] * COLS for _ in range(ROWS)]   # D* Lite bu listeleri okur, yerinde güncelliyoruz
        self.penalty = [[0.0] * COLS for _ in range(ROWS)]
        self.drone = start or (1, ROWS // 2)
        self.goal = goal or (COLS - 2, ROWS // 2)
        self.raw = np.zeros((ROWS, COLS), bool)
        self.occ = np.zeros((ROWS, COLS), bool)
        self.tti = np.full((ROWS, COLS), np.inf)          # payı eklenmemiş: engelin kendisi ne zaman gelir
        self.stats = dict(frames=0, replans=0, expanded=0, astar=0, changed=0, steps=0, reached=0, ms=0.0,
                          dodges=0)
        self.status = "Hazir"
        self.unseen = None                                # eğik kamera: kameranın görmediği hücreler
        self._new_dstar()

    def _new_dstar(self):
        self.dstar = DStarLite(self.grid, self.drone, self.goal, self.penalty)
        self.path = self.dstar.path()

    def set_goal(self, cell):
        self.goal = cell
        self._new_dstar()
        self.status = "Yeni hedef"

    def set_drone(self, cell):
        self.drone = cell
        self._new_dstar()

    def update(self, raw, moving_tti=None):
        """Bir karelik algılama sonucunu uygula: tahmin → pay → duvar/ceza → değişen hücreler → D* Lite."""
        t0 = time.perf_counter()
        self.raw = raw
        tti = np.where(raw, 0.0, np.inf)
        if self.predict and moving_tti is not None:
            tti = np.minimum(tti, moving_tti)
        self.tti = tti
        near = spread_min(tti, self.margin)                 # güvenlik payı tahmine de uygulanır
        occ = near <= H_HARD
        pen = np.where(occ | (near > tracking.HORIZON), 0.0,
                       PEN_MAX * (tracking.HORIZON + tracking.H_STEP - near) / (tracking.HORIZON - H_HARD))
        if self.predict:
            # Kör nokta: kadrajın dışını görmüyoruz, kamyon kenardan aniden girebilir. Kenar şeridinde
            # gerekmedikçe dolaşma (duvar değil, sadece pahalı; hedef oradaysa yine gider).
            band = edge_band(self.unseen)
            pen = np.where(band & ~occ, np.maximum(pen, EDGE_PEN), pen)
        if self.unseen is not None:                         # tahmin kapalı olsa da: görmediğin yerden kaçın
            pen = np.where(self.unseen & ~occ, np.maximum(pen, UNSEEN_PEN), pen)
        # Drone'un kendi hücresini boş tut: pay yüzünden drone "engelin içinde" kalıp kilitlenmesin,
        # kaçabileceği bir yer olsun. (Gerçekten engel içindeyse aşağıda uyarı basıyoruz.)
        dx, dy = self.drone
        occ[dy, dx] = False
        pen[dy, dx] = 0.0
        self.occ = occ

        ys, xs = np.nonzero((occ != np.array(self.grid, bool)) | (pen != np.array(self.penalty)))
        changed = list(zip(xs.tolist(), ys.tolist()))
        for x, y in changed:
            self.grid[y][x] = bool(occ[y, x])
            self.penalty[y][x] = float(pen[y, x])
        self.stats["frames"] += 1
        if changed:
            self.stats["changed"] += len(changed)
            n = self.dstar.cells_changed(changed)
            self.path = self.dstar.path()
            if n:
                self.stats["replans"] += 1
                self.stats["expanded"] += n
        self.stats["ms"] += (time.perf_counter() - t0) * 1000

        gx, gy = self.goal
        if self.raw[dy, dx]:
            self.status = "UYARI: drone engelin icinde gorunuyor"
        elif self.path is None:
            if self.in_danger():
                self.status = "KACIS: hareketli engel geliyor, kenara cekiliyorum"
            else:
                self.status = "YOL YOK: hedef engelli" if occ[gy, gx] else "YOL YOK: bekliyorum"
        else:
            self.status = f"Ucusta, rota {path_cost(self.path, self.penalty):.1f}"

    def in_danger(self, cell=None):
        """Bu hücreye (varsayılan: drone'un hücresi) yakında hareketli bir engel ya da payı geliyor mu?"""
        x, y = cell or self.drone
        return spread_min(self.tti, self.margin)[y, x] <= H_HARD

    def dodge_cell(self):
        """
        Kaçış hücresi: drone'un kendisi + 8 komşusu içinden engelin en geç geleceği yer.
        Eşitlikte payın da en geç geldiği, o da eşitse hedefe (D* Lite g değeri) en yakın olan.
        Şu an dolu hücreye ve dolu köşeden çapraz geçişe izin yok.
        """
        near = spread_min(self.tti, self.margin)
        x, y = self.drone
        best, best_key = self.drone, None
        for nx in range(x - 1, x + 2):
            for ny in range(y - 1, y + 2):
                if not (0 <= nx < COLS and 0 <= ny < ROWS) or self.tti[ny, nx] == 0:
                    continue
                if nx != x and ny != y and (self.tti[y, nx] == 0 or self.tti[ny, x] == 0):
                    continue
                key = (min(self.tti[ny, nx], 99), min(near[ny, nx], 99), -self.dstar.g_of((nx, ny)))
                if best_key is None or key > best_key:
                    best, best_key = (nx, ny), key
        return best

    def step_drone(self):
        if self.path and len(self.path) > 1:
            nxt = self.path[1]
            self.path = self.path[1:]
        elif self.predict and self.in_danger():
            nxt = self.dodge_cell()                      # rota yok ama üstümüze bir şey geliyor: kenara çekil
            self.path = None
            if nxt != self.drone:
                self.stats["dodges"] += 1
        else:
            return False
        if nxt == self.drone:
            return False
        self.drone = nxt
        self.dstar.move_to(self.drone)
        self.stats["steps"] += 1
        if self.drone == self.goal:
            self.stats["reached"] += 1
            return True
        return False


# ---------------- Tek dosya sürümü: algılama + planlama bir arada ----------------
class VisionPlanner(GridPlanner):
    """Perception + GridPlanner tek nesnede (drone_vision.py penceresi ve testler için). ROS'ta bu ikisi ayrı node."""

    def __init__(self, detector, start=None, goal=None, predict=True):
        self.perception = Perception(detector)
        self.det = detector
        self.filter = self.perception.filter
        self.tracker = self.perception.tracker
        super().__init__(start, goal, predict)

    @property
    def last_mask(self):
        return self.perception.last_mask

    def process(self, frame, height=None):
        """Bir kareyi işle: maske → hücre → filtre → takip/tahmin → pay → değişen hücreler → D* Lite."""
        t0 = time.perf_counter()
        raw, moving = self.perception.process(frame, height)
        self.unseen = self.perception.unseen
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        self.update(raw, moving)


def drone_hit(vp, gt_cells, height_map=None):
    """Drone gerçekte bir engelin içinde mi? 3D'de engelin tepesi drone'un irtifasının altındaysa değil."""
    x, y = vp.drone
    if not gt_cells[y, x]:
        return False
    z = getattr(vp, "z", None)
    if z is None or height_map is None:
        return True
    from drone_dstar3d import ALTS
    top = height_map[y * CELL_PX:(y + 1) * CELL_PX, x * CELL_PX:(x + 1) * CELL_PX].max()
    return top >= ALTS[z]


GOAL_PATIENCE = 40                 # hedef bu kadar kare her irtifada kapalı kalırsa en yakın boş hücreye taşı


def goal_blocked(planner):
    """Hedef hücresi her irtifada kapalı mı? (2D: tek katman)"""
    gx, gy = planner.goal
    occ3 = getattr(planner, "occ3", None)
    return bool(occ3[:, gy, gx].all()) if occ3 is not None else bool(planner.occ[gy, gx])


def nearest_open_cell(planner, cell):
    """cell'e en yakın, en az bir irtifada boş ve kameranın gördüğü hücre (yoksa None)."""
    occ3 = getattr(planner, "occ3", None)
    blocked = occ3.all(axis=0) if occ3 is not None else planner.occ.copy()
    unseen = getattr(planner, "unseen", None)
    if unseen is not None:
        blocked = blocked | unseen
    dx, dy = planner.drone
    blocked[dy, dx] = True                            # drone'un kendi hücresi hedef olmasın
    ys, xs = np.nonzero(~blocked)
    if len(xs) == 0:
        return None
    i = int(np.argmin((xs - cell[0]) ** 2 + (ys - cell[1]) ** 2))
    return int(xs[i]), int(ys[i])


def keep_goal_reachable(planner, patience=GOAL_PATIENCE):
    """
    Hedef her irtifada kapalıysa (hedefin üstüne bina/vinç/insan geldi ya da öyle algılandı) sonsuza kadar
    bekleme: patience kare sonra hedefi ona en yakın boş hücreye taşı. Hareketli engel geçip gidecekse
    o sürede açılır, taşımaya gerek kalmaz. True: hedef taşındı.
    """
    if planner.path is None and goal_blocked(planner):
        planner.blocked_frames = getattr(planner, "blocked_frames", 0) + 1
    else:
        planner.blocked_frames = 0
    if planner.blocked_frames < patience:
        return False
    c = nearest_open_cell(planner, planner.goal)
    planner.blocked_frames = 0
    if c is None or c == planner.goal:
        return False
    old = planner.goal
    planner.set_goal(c)
    planner.status = f"Hedef {old} kapali, en yakin bos nokta {c}"
    return True


def random_free_cell(planner, rng, far_from=None):
    unseen = getattr(planner, "unseen", None)
    for _ in range(500):
        # Kenar şeridine (ve kameranın görmediği yere) hedef koyma: oraya kadraj dışından habersiz bir şey girebilir
        c = (rng.randrange(EDGE_BAND, COLS - EDGE_BAND), rng.randrange(EDGE_BAND, ROWS - EDGE_BAND))
        if planner.occ[c[1], c[0]] or (unseen is not None and unseen[c[1], c[0]]):
            continue
        if far_from and abs(c[0] - far_from[0]) + abs(c[1] - far_from[1]) < COLS // 2:
            continue
        return c
    return None


# ---------------- Çizim ----------------
def draw(frame, vp, det, source_name, paused, preview=None, yolo=None):
    """frame: kuşbakışı görüntü (eğik kamerada yere serilmiş hali). preview: sağ üstte maske yerine gösterilecek kare."""
    view = frame.copy()
    overlay = view.copy()
    z = getattr(vp, "z", None)                       # 3D planlayıcıda drone'un katmanı
    if z is None:
        occ, low = vp.occ, np.zeros_like(vp.raw)
    else:
        from drone_dstar3d import ALTS, CLEARANCE
        occ = vp.occ3[z]
        low = vp.raw & (ALTS[z] >= vp.height + CLEARANCE)    # bu irtifada üstünden geçilir
    infl_only = occ & ~vp.raw
    for y in range(ROWS):
        for x in range(COLS):
            if vp.raw[y, x] or infl_only[y, x]:
                col = C_LOW if low[y, x] else C_RAW if vp.raw[y, x] else C_INFL
                cv2.rectangle(overlay, (x * CELL_PX, y * CELL_PX), ((x + 1) * CELL_PX - 1, (y + 1) * CELL_PX - 1), col, -1)
    view = cv2.addWeighted(overlay, 0.45, view, 0.55, 0)
    # Tahmin: engelin yakında gireceği hücreler; ne kadar yakında gelecekse o kadar koyu
    if vp.predict:
        overlay = view.copy()
        horizon = tracking.HORIZON
        for y, x in zip(*np.nonzero((vp.tti > 0) & (vp.tti <= horizon))):
            cv2.rectangle(overlay, (x * CELL_PX, y * CELL_PX), ((x + 1) * CELL_PX - 1, (y + 1) * CELL_PX - 1), C_PRED, -1)
        view = cv2.addWeighted(overlay, 0.45, view, 0.55, 0)
    # İzler: kimlik + 10 karelik hız oku
    for t in vp.tracker.tracks:
        if not (t.confirmed and t.moving):
            continue
        c = t.center_at(0)
        tip = c + t.v * 10
        cv2.arrowedLine(view, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])), C_TRACK, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(view, f"#{t.id}", (int(c[0]) + 6, int(c[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TRACK, 1, cv2.LINE_AA)
    for x in range(0, FRAME_W, CELL_PX):
        cv2.line(view, (x, 0), (x, FRAME_H), (60, 60, 60), 1)
    for y in range(0, FRAME_H, CELL_PX):
        cv2.line(view, (0, y), (FRAME_W, y), (60, 60, 60), 1)
    mid = lambda c: (c[0] * CELL_PX + CELL_PX // 2, c[1] * CELL_PX + CELL_PX // 2)
    if vp.path and len(vp.path) > 1 and z is None:
        pts = np.array([mid(c) for c in vp.path], np.int32)
        cv2.polylines(view, [pts], False, C_PATH, 3, cv2.LINE_AA)
    elif vp.path and len(vp.path) > 1:
        for a, b in zip(vp.path, vp.path[1:]):        # rengi irtifaya göre; tırmanma/alçalma halka
            if a[2] == b[2]:
                cv2.line(view, mid(a), mid(b), C_LEVELS[a[2]], 3, cv2.LINE_AA)
            else:
                cv2.circle(view, mid(a), CELL_PX // 2 - 3, C_LEVELS[b[2]], 2, cv2.LINE_AA)
    if yolo is not None:
        yolo.annotate_bev(view)
    gx, gy = vp.goal
    cv2.rectangle(view, (gx * CELL_PX + 3, gy * CELL_PX + 3), ((gx + 1) * CELL_PX - 4, (gy + 1) * CELL_PX - 4), C_GOAL, -1)
    dx, dy = vp.drone
    cv2.circle(view, mid(vp.drone), CELL_PX // 2 - 2, C_DRONE, -1, cv2.LINE_AA)
    if z is not None:
        cv2.putText(view, f"{ALTS[z]}m", (dx * CELL_PX + CELL_PX, dy * CELL_PX + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, f"{ALTS[z]}m", (dx * CELL_PX + CELL_PX, dy * CELL_PX + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, C_LEVELS[z], 1, cv2.LINE_AA)

    # Sağda küçük maske önizlemesi
    small = preview if preview is not None else cv2.cvtColor(vp.last_mask, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(small, (FRAME_W // 3, FRAME_H // 3))
    side = np.zeros((FRAME_H, FRAME_W // 3, 3), np.uint8)
    side[:small.shape[0]] = small
    st = vp.stats
    avg = st["ms"] / max(st["frames"], 1)
    moving = sum(1 for t in vp.tracker.tracks if t.confirmed and t.moving)
    lines = [
        f"kaynak: {source_name}   yontem: {det.method}",
        f"guvenlik payi: {vp.margin} hucre",
        f"tahmin: {'acik' if vp.predict else 'KAPALI'}  hareketli iz: {moving}",
        f"kenara cekilme: {st['dodges']}",
        f"kare: {st['frames']}  isleme: {avg:.1f} ms/kare",
        f"yeniden plan: {st['replans']}",
        f"D* acilan dugum: {st['expanded']}",
        f"degisen hucre: {st['changed']}",
        f"adim: {st['steps']}  hedef: {st['reached']}",
        "",
        "kirmizi: engel  turuncu: pay",
        "mor: hareketli engel gelecek",
        "sol tik: hedef  sag tik: drone",
        "M yontem  C kalibre  I pay  T tahmin",
        "+/- pay  N yeni sahne",
        "SPACE dur  Q cik",
    ]
    gap = 22
    if z is not None:                                   # profil grafiğine yer aç
        lines[1:1] = [f"irtifa: {ALTS[z]} m   tirmanma: {st['climbs']}", "mavi: bu irtifada ustunden gecilir"]
        lines = [t for t in lines if not t.startswith(("degisen", "D* acilan", "yeniden plan", "+/- pay"))]
        gap = 18
    if paused:
        lines.insert(0, "DURAKLATILDI")
    y0 = small.shape[0] + 24
    for i, t in enumerate(lines):
        cv2.putText(side, t, (8, y0 + i * gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(side, "kamera" if preview is not None else "maske", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if z is not None:
        draw_profile(side, vp)
    out = np.hstack([view, side])
    cv2.rectangle(out, (0, 0), (FRAME_W, 26), (0, 0, 0), -1)
    cv2.putText(out, vp.status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)
    return out


def draw_profile(side, vp):
    """Yan panelin altına rota profili: x = rota adımı, y = metre. Gri çubuk altındaki engel, renkli çizgi drone."""
    from drone_dstar3d import ALTS
    w, h = side.shape[1] - 16, 110
    x0, y0 = 8, side.shape[0] - 12                      # sol alt köşe
    top = ALTS[-1] + 10
    cv2.rectangle(side, (x0, y0 - h), (x0 + w, y0), (40, 40, 40), -1)
    for a in ALTS:
        ya = int(y0 - h * a / top)
        cv2.line(side, (x0, ya), (x0 + w, ya), (70, 70, 70), 1)
        cv2.putText(side, f"{a}", (x0 + 2, ya - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(side, "rota profili (m)", (x0, y0 - h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
    path = vp.path or [(*vp.drone, vp.z)]
    n = max(len(path) - 1, 1)
    px = lambda i: int(x0 + w * i / n)
    for i, (x, y, z) in enumerate(path):              # altındaki engelin boyu
        hg = vp.height[y, x]
        if hg > 0:
            hh = int(h * min(hg, top) / top)
            cv2.rectangle(side, (px(i) - 1, y0 - hh), (px(i) + 1, y0), (120, 120, 120), -1)
    for i in range(len(path) - 1):
        za, zb = path[i][2], path[i + 1][2]
        cv2.line(side, (px(i), int(y0 - h * ALTS[za] / top)), (px(i + 1), int(y0 - h * ALTS[zb] / top)),
                 C_LEVELS[max(za, zb)], 2, cv2.LINE_AA)


# ---------------- Ana döngü ----------------
def make_yolo(args, src):
    """YOLO algılayıcısı: --model sim ise sahnenin kendisi (kusursuz dedektör), değilse Ultralytics modeli."""
    import drone_yolo as Y
    proj = None
    if args.perspektif:
        cam = src.cam or Y.Camera(args.irtifa, args.egim, args.fov, args.vfov)
        proj = Y.GroundProjector.from_file(args.kalibrasyon, cam, args.hucre_m) if args.kalibrasyon \
            else Y.GroundProjector.from_camera(cam, args.hucre_m)
    if args.model == "sim":
        if not src.sim:
            raise SystemExit("[!] --model sim sadece simülasyonda çalışır")
        backend = Y.SimOracle(src.sim, src.cam)
    else:
        backend = Y.UltralyticsBackend(args.model, args.guven)
    return Y.YoloDetector(backend, proj)


def next_method(det, args, src):
    """M tuşu: sıradaki yöntem. Eğik kamerada sadece yolo anlamlı; YOLO kurulu değilse atla."""
    if args.perspektif:
        return
    i = ObstacleDetector.METHODS.index(det.method)
    nxt = ObstacleDetector.METHODS[(i + 1) % len(ObstacleDetector.METHODS)]
    if nxt in ("yolo", "hibrit") and det.yolo is None:
        try:
            det.yolo = make_yolo(args, src)
        except ImportError as e:
            print(f"[!] {e}")
            nxt = "renk"
    det.method = nxt


def render(frame, vp, det, src, paused):
    """Pencere görüntüsü. YOLO'da sağ üstte tespitli kamera karesi; eğik kamerada ana görüntü yere serilmiş kare."""
    yolo = det.yolo if det.method in ("yolo", "hibrit") else None
    if yolo is None:
        return draw(frame, vp, det, src.name, paused)
    import drone_yolo as Y
    return draw(yolo.bev_view(frame), vp, det, src.name, paused, preview=Y.draw_detections(frame, yolo.dets), yolo=yolo)


def build(args):
    if args.perspektif and args.yontem not in ("yolo", "hibrit"):
        raise SystemExit("[!] --perspektif sadece --yontem yolo ile (renk/kenar kuşbakışı varsayar)")
    src = Source(args)
    det = ObstacleDetector(args.yontem)
    if args.yontem in ("yolo", "hibrit"):
        det.yolo = make_yolo(args, src)
        if args.perspektif and args.yontem == "hibrit":
            print("[i] Eğik kamerada hibrit = yolo (renk yöntemi kuşbakışı varsayar)")
    if src.sim:
        det.calibrate(src.sim.background())
    else:
        # Webcam/videoda ilk kareyi zemin say; daha iyisi: boş zemini gösterip C'ye basmak
        frame, _ = src.read()
        if frame is not None:
            det.calibrate(frame)
    # Eğik kamerada drone (kamera) grid'in altında ortada, ileriye (yukarı) uçuyor
    start, goal = ((COLS // 2, ROWS - 2), (COLS // 2, 3)) if args.perspektif else (None, None)
    if args.uc_boyut:
        from drone_plan3d import VisionPlanner3D
        vp = VisionPlanner3D(det, start, goal, predict=not args.tahminsiz)
    else:
        vp = VisionPlanner(det, start, goal, predict=not args.tahminsiz)
    return src, det, vp


def run_headless(args):
    src, det, vp = build(args)
    rng = random.Random(args.seed)
    moved_into = 0       # drone gerçekte dolu olan hücreye kendisi girdi mi? (planlama/algılama hatası)
    run_over = 0         # hareketli engel drone'un üstüne geldi (kare sayısı)
    tp = fp = fn = 0
    snap_at = {args.headless // 3, 2 * args.headless // 3, args.headless - 1}
    for f in range(args.headless):
        frame, gt = src.read()
        if frame is None:
            break
        hm = src.height()
        vp.process(frame, hm)
        keep_goal_reachable(vp)
        if src.cam is not None:
            # Eğik kamerada derinlik haritası yok; drone en fazla 20 m'de uçuyor, 33 m'deki vinç kolunun altı engel değil
            from drone_dstar3d import ALTS, CLEARANCE
            gt = src.sim.footprint_below(ALTS[-1] + CLEARANCE)
        gt_cells = ObstacleDetector.to_cells(gt) if gt is not None else None
        if gt_cells is not None:
            if vp.unseen is not None:                  # eğik kamera: sadece görüş alanını say
                gt_cells = gt_cells & ~vp.unseen
            tp += int(np.sum(gt_cells & vp.raw)); fp += int(np.sum(~gt_cells & vp.raw)); fn += int(np.sum(gt_cells & ~vp.raw))
        before = vp.drone
        if f % MOVE_EVERY == 0 and vp.step_drone():
            nxt = random_free_cell(vp, rng, far_from=vp.drone)
            if nxt:
                vp.set_goal(nxt)
        if src.cam is not None:                        # eğik kamera: drone'un irtifasını kesen parçalar
            from drone_dstar3d import ALTS
            alt = ALTS[vp.z] if getattr(vp, "z", None) is not None else None
            hit = ObstacleDetector.to_cells(src.sim.footprint_below(22) if alt is None else src.sim.footprint_at(alt))
            hit = bool(hit[vp.drone[1], vp.drone[0]])
        else:
            hit = gt_cells is not None and drone_hit(vp, gt_cells, hm)
        if hit:
            if vp.drone != before:
                moved_into += 1
            else:
                run_over += 1
        if f in snap_at:
            out = os.path.join(args.out, f"vision_frame_{f:04d}.png")
            cv2.imwrite(out, render(frame, vp, det, src, False))
    st = vp.stats
    print(f"kaynak={src.name}{' (egik kamera)' if src.cam else ''} yontem={det.method} kare={st['frames']} ort={st['ms'] / max(st['frames'], 1):.1f}ms "
          f"yeniden_plan={st['replans']} D*_dugum={st['expanded']} adim={st['steps']} hedef={st['reached']} "
          f"kenara_cekilme={st['dodges']} engele_girdi={moved_into} engel_ustune_geldi={run_over}")
    if tp + fp + fn:
        print(f"algilama (hucre bazinda, pay haric): kesinlik={tp / max(tp + fp, 1):.2f} "
              f"duyarlilik={tp / max(tp + fn, 1):.2f}")
    print(f"PNG'ler: {args.out}")


def run_gui(args):
    src, det, vp = build(args)
    win = "DroneRoute - OpenCV engel algilama + D* Lite"
    cv2.namedWindow(win)
    rng = random.Random(args.seed)
    paused, frame_no, frame = False, 0, None

    def on_mouse(event, x, y, flags, _):
        if x >= FRAME_W:
            return
        c = (x // CELL_PX, y // CELL_PX)
        if event == cv2.EVENT_LBUTTONDOWN and c != vp.drone:
            vp.set_goal(c)
        elif event == cv2.EVENT_RBUTTONDOWN and c != vp.goal:
            vp.set_drone(c)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        if not paused or frame is None:
            frame, _ = src.read()
            if frame is None:
                print("[!] Kare alınamadı, çıkıyorum.")
                break
            vp.process(frame, src.height())
            if keep_goal_reachable(vp):
                print(f"[i] {vp.status}")
            frame_no += 1
            if frame_no % MOVE_EVERY == 0 and vp.step_drone():
                nxt = random_free_cell(vp, rng, far_from=vp.drone)   # hedefe vardı, yenisini seç
                if nxt:
                    vp.set_goal(nxt)
        cv2.imshow(win, render(frame, vp, det, src, paused))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("m"):
            next_method(det, args, src)
        elif k == ord("c"):
            det.calibrate(src.sim.background() if src.sim else frame)
            print("Zemin kalibre edildi.")
        elif k == ord("i"):
            vp.margin = 0 if vp.margin else MARGIN
        elif k == ord("t"):
            vp.predict = not vp.predict
        elif k in (ord("+"), ord("=")):
            vp.margin = min(vp.margin + 1, MAX_MARGIN)
        elif k == ord("-"):
            vp.margin = max(vp.margin - 1, 0)
        elif k == ord("n") and src.sim:
            src.sim = SimScene()
            det.calibrate(src.sim.background())
            if det.yolo is not None and hasattr(det.yolo.backend, "scene"):
                det.yolo.backend.scene = src.sim
            vp = type(vp)(det, vp.drone if args.perspektif else None, vp.goal if args.perspektif else None,
                          predict=vp.predict)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="DroneRoute: OpenCV engel algılama + D* Lite")
    ap.add_argument("--kamera", action="store_true", help="webcam kullan")
    ap.add_argument("--kamera-id", type=int, default=0)
    ap.add_argument("--video", help="video dosyası")
    ap.add_argument("--yontem", choices=ObstacleDetector.METHODS, default="renk")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tahminsiz", action="store_true", help="hareketli engel tahminini kapat (eski davranış)")
    ap.add_argument("--headless", type=int, default=0, help="pencere açmadan N kare koş")
    ap.add_argument("--out", default=".", help="headless PNG çıktı klasörü")
    ap.add_argument("--3d", dest="uc_boyut", action="store_true",
                    help="irtifalı planlama (engel yükseklikleri: simde derinlikten, YOLO'da sınıf/perspektiften)")
    ap.add_argument("--model", default=None,
                    help="YOLO ağırlıkları (.pt). Varsayılan models/droneroute_yoloe.pt; 'sim' = sahnenin kusursuz dedektörü")
    ap.add_argument("--guven", type=float, default=0.25, help="YOLO en düşük güven (0-1)")
    import drone_yolo
    drone_yolo.add_camera_args(ap)
    args = ap.parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
