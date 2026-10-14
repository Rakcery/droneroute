"""
DroneRoute — Adım 5: YOLO ile nesne tanıma + eğik kamera (perspektif) → grid

Renk/kenar algılama "zemine benzemeyen her şey engel" der ama ne olduğunu bilmez. Burada önceden eğitilmiş
bir YOLO modeli engelin SINIFINI söylüyor: vinç, bina, ağaç, araç, insan, kuş. Sınıf iki şey kazandırıyor:
  1) Boy: derinlik kamerası yoksa bile engelin kabaca ne kadar yüksek olduğunu biliriz (3D planlama için).
  2) Kural: insanların ve kuşların üstünden uçulmaz (boyu "sonsuz" sayılır), vinç kolunun da.

Model: YOLOE-26s-seg (Ultralytics), AÇIK SÖZLÜK. COCO'da vinç ve bina sınıfı yok; açık sözlük modele
sınıf adlarını İngilizce yazıyla veriyoruz ("tower crane", "building"...), yeniden eğitmeden tanıyor.
Sözlük bir kere modele gömülüp models/droneroute_yoloe.pt olarak kaydediliyor (bir kerelik ~270 MB indirme:
model + metin kodlayıcı, sonra internet gerekmez). CPU'da ~0.2-0.3 sn/kare, GPU şart değil.
Dürüst not: araç, insan ve vinçte iyi; bina ve ağacı (özellikle kuşbakışı) çoğu zaman kaçırıyor. Bu yüzden
"hibrit" mod var: YOLO'nun bulduklarına sınıf + boy, YOLO'nun kaçırdıklarını renk yöntemi yakalar (boyu
bilinmiyor = üstünden uçulmaz). Bina/ağaç için asıl çözüm küçük bir ince ayar (fine-tune), --model ile
kendi modelini verebilirsin (sınıf adları aşağıdaki sözlükle eşleşiyorsa direkt çalışır, COCO modelleri de).

Perspektif: gerçek drone kamerası yere tam dik bakmaz, öne eğik bakar. Görüntüdeki piksel artık zemindeki
bir hücre değil. GroundProjector:
  - Zemin homografisi (3x3): görüntü pikseli ↔ zemin (metre). Kameranın irtifa/eğim/görüş açısından
    hesaplanır ya da sahada 4+ bilinen noktadan kalibre edilir (DLT, kendi kodumuz).
  - Yerdeki engel: kutunun ALT kenarı zemine değer → homografiyle zemine indir = engelin önü ve uzaklığı.
    Arkaya doğru ne kadar uzandığını göremeyiz, sınıfa göre tahmin (bina 12 m, araç 7 m...).
  - Boy (derinlik tahmini): kutunun üst kenarından geçen ışın, engelin uzaklığında ne kadar yüksekte?
    Yukarıdan bakınca kutunun tepesi aslında çatının arka kenarı, yani boy biraz FAZLA çıkar (güvenli taraf).
  - Uçan (kuş/drone): yere değmez. Gerçek boyutu ~biliniyor: uzaklık = odak * boyut / piksel genişliği.
Çıktı yine drone_vision'ın beklediği şey: kuşbakışı (FRAME_H x FRAME_W) engel maskesi + boy görüntüsü.
Böylece zaman filtresi, takip/tahmin, güvenlik payı, 2D/3D D* Lite hiç değişmeden çalışıyor.

Rota hesaplama hâlâ tamamen bizim kodumuz (D* Lite). YOLO sadece "ne nerede" diyor.

Kullanım:
  python drone_yolo.py --model-hazirla                       # modeli bir kere hazırla (internet gerekir)
  python drone_yolo.py --resim foto.jpg                      # kuşbakışı fotoğrafta algıla, grid + rota çiz
  python drone_yolo.py --resim foto.jpg --perspektif --irtifa 40 --egim 30 --fov 82
  python drone_yolo.py --video ucus.mp4 --perspektif         # videoda canlı
  Pencerede: sol tık hedef, sağ tık drone, SPACE dur/devam, R baştan, Q çık. Drone rota boyunca uçar.
  python drone_vision.py --yontem yolo --video ucus.mp4 --perspektif     # tam boru hattı + planlama
  python drone_vision.py --yontem yolo --perspektif --model sim          # eğik kamerayla simülasyon
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

from drone_dstar_lite import COLS, ROWS

CELL_PX = 22                  # drone_vision.CELL_PX ile aynı (döngüsel import olmasın diye burada da)
FRAME_W, FRAME_H = COLS * CELL_PX, ROWS * CELL_PX

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "droneroute_yoloe.pt")
BASE_MODEL = "yoloe-26s-seg.pt"
INF = math.inf
BIG = 1e9                     # fillPoly'ye sonsuz verilemiyor; sonra inf'e çeviriyoruz

# Sınıflar:
#   prompts : açık sözlük modele verilen İngilizce ifadeler (hepsi bu sınıfa düşer)
#   h       : boy (metre) ölçülemezse kullanılan değer. INF = asla üstünden uçma (kural ya da bilinmez)
#   d       : eğik kamerada engelin kameradan uzağa doğru uzunluğu (görünmüyor, tahmin, metre)
#   fly     : yere değmez (kuş, başka drone); size = gerçek genişliği (metre), uzaklık bundan
CLASSES = {
    "vinc":  dict(prompts=("construction crane", "tower crane"), h=INF, d=8.0),
    "bina":  dict(prompts=("building", "building rooftop", "apartment building", "house"), h=30.0, d=12.0),
    "agac":  dict(prompts=("tree", "tree canopy"), h=15.0, d=5.0),
    "arac":  dict(prompts=("truck", "bus", "car", "van", "excavator"), h=4.5, d=7.0),
    "insan": dict(prompts=("person",), h=INF, d=1.0),              # insanların üstünden uçulmaz
    "kus":   dict(prompts=("bird", "drone"), h=INF, fly=True, size=1.0),
}
# Başka modellerin (COCO: yolo11n.pt vb. ya da kendi ince ayarın) sınıf adları
EXTRA_NAMES = {"motorcycle": "arac", "bicycle": "arac", "train": "arac", "crane": "vinc",
               "vinc": "vinc", "bina": "bina", "agac": "agac", "arac": "arac", "insan": "insan", "kus": "kus"}
NAME2CLASS = {p: c for c, v in CLASSES.items() for p in v["prompts"]} | EXTRA_NAMES
SAFETY_H = 1.15               # ölçülen boyu %15 + 0.5 m büyüt (ölçüm kaba)

# BGR, çizim için
CLASS_COLOR = {"vinc": (0, 200, 255), "bina": (180, 120, 60), "agac": (60, 180, 60), "arac": (40, 120, 255),
               "insan": (200, 80, 255), "kus": (255, 255, 255)}


class Detection:
    """Bir nesne: sınıf (bizim adımız), güven, kutu (x1, y1, x2, y2 piksel), varsa segmentasyon poligonu."""
    __slots__ = ("cls", "conf", "box", "poly")

    def __init__(self, cls, conf, box, poly=None):
        self.cls, self.conf, self.box, self.poly = cls, float(conf), tuple(float(v) for v in box), poly

    def __repr__(self):
        return f"Detection({self.cls}, {self.conf:.2f}, {[round(v) for v in self.box]})"


# ---------------- Dedektörler (arka uç) ----------------
def prepare_model(path=DEFAULT_MODEL):
    """Açık sözlük YOLOE modeline bizim sınıf adlarını göm ve kaydet (bir kerelik, internet gerekir)."""
    from ultralytics import YOLOE
    print("[i] YOLO modeli hazırlanıyor (bir kerelik ~270 MB indirme: YOLOE + MobileCLIP metin kodlayıcı)...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))         # metin kodlayıcı çalışma klasörüne iniyor: proje kökünü kirletmesin
    try:
        m = YOLOE(os.path.join(os.path.dirname(path), BASE_MODEL))
        names = [p for v in CLASSES.values() for p in v["prompts"]]
        m.set_classes(names, m.get_text_pe(names))
        m.save(path)
    finally:
        os.chdir(cwd)
    print(f"[i] Hazır: {path}  (artık internet gerekmez)")
    return path


class UltralyticsBackend:
    """Ultralytics YOLO (CPU). model=None: bizim sözlüklü YOLOE; yoksa bir kere hazırlanır."""

    def __init__(self, model=None, conf=0.25, imgsz=640, device="cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("YOLO modu için: pip install ultralytics   (README'de 'Adım 5' kısmına bak)") from e
        path = model or DEFAULT_MODEL
        if model is None and not os.path.exists(path):
            prepare_model(path)
        self.model = YOLO(path)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.map = {int(i): NAME2CLASS.get(n.lower()) for i, n in self.model.names.items()}
        if not any(self.map.values()):
            print(f"[!] {path}: sınıf adlarının hiçbiri tanıdık değil (NAME2CLASS'a ekle): {self.model.names}")

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        out = []
        for box, conf, k, p in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), polys):
            cls = self.map.get(int(k))
            if cls:
                poly = np.round(p).astype(np.int32) if p is not None and len(p) >= 3 else None
                out.append(Detection(cls, conf, box, poly))
        return out


class SimOracle:
    """
    Test/demo için "kusursuz YOLO": simülasyon sahnesinin bildiği nesneleri kutu olarak verir.
    Gerçek YOLO çizgi film sahnede bir şey bulamaz; bununla model indirmeden boru hattının kalanını
    (kutu → grid → boy → planlama) uçtan uca deneriz. cam verilirse eğik kameradaki kutuları verir.
    """

    def __init__(self, scene, cam=None):
        self.scene, self.cam = scene, cam

    def __call__(self, frame):
        out = []
        for obj in self.scene.objs:              # render() anındaki nesneler (kare ile aynı an)
            if self.cam is None:
                for poly, _, _, _ in obj["parts"]:
                    x, y, w, h = cv2.boundingRect(poly)
                    out.append(Detection(obj["cls"], 1.0, (x, y, x + w, y + h), poly))
            else:
                box = self.cam.object_box(obj)
                if box is not None:
                    out.append(Detection(obj["cls"], 1.0, box))
        return out


# ---------------- Kamera + zemin homografisi (kendi kodumuz) ----------------
def homography_dlt(src, dst):
    """
    src → dst noktalarını eşleyen 3x3 homografi (4+ nokta, en küçük kareler). Doğrudan Doğrusal Dönüşüm (DLT):
    her eşleşme 2 doğrusal denklem verir, A h = 0'ın çözümü A'nın en küçük tekil değerine ait vektör.
    Sayısal kararlılık için iki taraf da önce merkezlenip ölçekleniyor (Hartley normalizasyonu).
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    assert len(src) >= 4 and len(src) == len(dst), "en az 4 nokta çifti lazım"

    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = norm(src), norm(dst)
    ps = (Ts @ np.c_[src, np.ones(len(src))].T).T
    pd = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(ps, pd):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def apply_h(H, pts):
    """Noktaları (N, 2) homografiden geçir."""
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.asarray(pts).reshape(-1, 2)))] @ H.T
    return p[:, :2] / p[:, 2:3]


class Camera:
    """
    Eğik bakan kamera, iğne deliği (pinhole) modeli. Dünya: X sağ, Y ileri (zeminde), Z yukarı (metre).
    Kamera (0, 0, irtifa)'da, Y yönüne bakıyor, egim derece aşağı eğik (90 = tam kuşbakışı).
    hfov: yatay görüş açısı; vfov verilmezse kare piksel varsayılır. Görüntü FRAME_W x FRAME_H'ye ölçeklenmiş
    olduğu için 16:9 bir videonun vfov'unu ayrıca vermek gerekebilir (vfov = 2*atan(tan(hfov/2) * 9/16)).
    """

    def __init__(self, alt=25.0, pitch=35.0, hfov=82.0, vfov=None, w=FRAME_W, h=FRAME_H):
        self.alt, self.pitch, self.w, self.h = alt, pitch, w, h
        th = math.radians(hfov) / 2
        tv = math.tan(math.radians(vfov) / 2) if vfov else math.tan(th) * h / w
        self.fx, self.fy = w / 2 / math.tan(th), h / 2 / tv
        self.cx, self.cy = w / 2, h / 2
        p = math.radians(pitch)
        self.C = np.array([0.0, 0.0, alt])
        self.R = np.array([[1.0, 0.0, 0.0],                     # satırlar: kameranın sağ, aşağı, ileri eksenleri
                           [0.0, -math.sin(p), -math.cos(p)],   # (dünya koordinatında)
                           [0.0, math.cos(p), -math.sin(p)]])

    def project(self, P):
        """Dünya noktaları (N, 3) → piksel (N, 2) ve kameranın önündeki derinlik (N,)."""
        Pc = (np.asarray(P, float).reshape(-1, 3) - self.C) @ self.R.T
        z = Pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        return np.c_[self.fx * Pc[:, 0] / zs + self.cx, self.fy * Pc[:, 1] / zs + self.cy], z

    def ray(self, u, v):
        """Pikselden geçen ışının dünya yönü (birim değil)."""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0]) @ self.R

    def horizon_v(self):
        """Ufuk çizgisinin satırı: bunun üstündeki pikseller zemine hiç değmez."""
        return self.cy - self.fy * math.tan(math.radians(self.pitch))

    def ground_homography(self):
        """Görüntü pikseli → zemin (X, Y metre) homografisi; kameranın gördüğü zemin noktalarından DLT ile."""
        ground = np.array([(x, y) for x in (-30.0, -10.0, 10.0, 30.0) for y in (10.0, 30.0, 60.0)])
        px, _ = self.project(np.c_[ground, np.zeros(len(ground))])
        return homography_dlt(px, ground)

    def object_box(self, obj):
        """SimOracle için: sahne nesnesinin (prizmalar) bu kameradaki kutusu; görünmüyorsa None."""
        pts = []
        for poly, z0, z1, _ in obj["parts"]:
            g = bev_to_ground(poly)
            if CLASSES[obj["cls"]].get("fly"):           # render_oblique uçanı CLASSES boyutunda top çiziyor
                c, r = np.r_[g.mean(axis=0), (z0 + z1) / 2], CLASSES[obj["cls"]]["size"] / 2
                px, zc = self.project(c[None])
                if zc[0] <= 0.5:
                    return None
                rp = self.fx * r / zc[0]
                pts.append(np.array([px[0] - rp, px[0] + rp]))
                continue
            for z in (z0, z1):
                pts.append(np.c_[g, np.full(len(g), z)])
        if CLASSES[obj["cls"]].get("fly"):
            px = np.vstack(pts)
        else:
            px, z = self.project(np.vstack(pts))
            px = px[z > 0.5]
        if len(px) == 0:
            return None
        x1, y1 = np.maximum(px.min(axis=0), 0)
        x2, y2 = np.minimum(px.max(axis=0), (self.w - 1, self.h - 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return x1, y1, x2, y2


# Kuşbakışı grid (BEV) ↔ zemin: kamera grid'in alt kenarının ortasında, grid'de yukarı = ileri (Y)
CELL_M = 2.0      # eğik kamerada bir hücre kaç metre (40x28 hücre = 80 m x 56 m)


def ground_to_bev(XY, cell_m=None):
    s = CELL_PX / (cell_m or CELL_M)
    XY = np.asarray(XY, float).reshape(-1, 2)
    return np.c_[FRAME_W / 2 + XY[:, 0] * s, FRAME_H - XY[:, 1] * s]


def bev_to_ground(px, cell_m=None):
    s = (cell_m or CELL_M) / CELL_PX
    px = np.asarray(px, float).reshape(-1, 2)
    return np.c_[(px[:, 0] - FRAME_W / 2) * s, (FRAME_H - px[:, 1]) * s]


class GroundProjector:
    """Eğik kamera görüntüsündeki tespitleri kuşbakışı grid'e (taban poligonu + boy) indirir."""

    def __init__(self, H, cam=None, cell_m=CELL_M):
        self.H, self.cam, self.cell_m = H, cam, cell_m           # H: piksel → zemin (metre)
        self.Hinv = np.linalg.inv(H)
        # Görüş alanı: hücre merkezi kamerada ufkun altına ve kadraja düşüyor mu?
        ys, xs = np.mgrid[0:ROWS, 0:COLS]
        ctr = np.c_[(xs.ravel() + 0.5) * CELL_PX, (ys.ravel() + 0.5) * CELL_PX]
        g = bev_to_ground(ctr, cell_m)
        p = np.c_[g, np.ones(len(g))] @ self.Hinv.T
        uv = p[:, :2] / np.where(np.abs(p[:, 2:3]) > 1e-12, p[:, 2:3], 1e-12)
        g0 = apply_h(H, [(FRAME_W / 2, FRAME_H - 1)])[0]       # kesin görünen nokta: w'nin işareti ona göre
        w0 = np.r_[g0, 1.0] @ self.Hinv[2]
        ok = (p[:, 2] * w0 > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < FRAME_W) & (uv[:, 1] < FRAME_H) & \
             (uv[:, 1] > self._horizon() + 4)
        self.unseen = ~ok.reshape(ROWS, COLS)

    @classmethod
    def from_camera(cls, cam, cell_m=CELL_M):
        return cls(cam.ground_homography(), cam, cell_m)

    @classmethod
    def from_file(cls, path, cam=None, cell_m=CELL_M):
        """
        Saha kalibrasyonu (JSON): {"goruntu": [[u, v], ...], "zemin": [[X, Y], ...]}  4+ nokta.
        Görüntü noktaları FRAME_W x FRAME_H (880x616) karede; zemin metre, kamera ayağı orijin, Y ileri.
        Boy tahmini için cam (irtifa/eğim/fov) yine gerekir; yoksa boylar sınıftan gelir.
        """
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(homography_dlt(d["goruntu"], d["zemin"]), cam, cell_m)

    def to_ground(self, uv):
        return apply_h(self.H, uv)

    def bev_homography(self):
        """Görüntü pikseli → kuşbakışı grid pikseli (kameranın görüntüsünü yere sermek için)."""
        s = CELL_PX / self.cell_m
        G = np.array([[s, 0, FRAME_W / 2], [0, -s, FRAME_H], [0, 0, 1]])
        return G @ self.H

    def _horizon(self):
        if self.cam:
            return self.cam.horizon_v()
        # Homografiden: zeminde sonsuza giden noktaların satırı (H'nin 3. satırı = 0)
        a, b, c = self.H[2]
        return -(a * FRAME_W / 2 + c) / b if abs(b) > 1e-12 else -INF

    def footprint(self, det):
        """Tespit → (zemindeki taban poligonu [metre], boy [metre]); zemine indirilemiyorsa None."""
        spec = CLASSES[det.cls]
        x1, y1, x2, y2 = det.box
        if spec.get("fly"):
            return self._flyer(det, spec)
        yb = min(y2, FRAME_H - 1)
        if yb <= self._horizon() + 2:          # kutunun altı ufkun üstünde: yere değdiği yeri göremiyoruz
            return None
        L, R = self.to_ground([(x1, yb), (x2, yb)])
        mid = (L + R) / 2
        dist = np.hypot(*mid)
        if not np.isfinite(dist) or dist < 1e-6 or mid[1] < 0:
            return None
        u = mid / dist                         # kameradan uzağa doğru (zeminde)
        d = spec["d"]
        cut_bottom, cut_top = y2 >= FRAME_H - 2, y1 <= 1
        back = d if cut_bottom else 0.0        # alttan kesik: ayağı kadrajın dışında, kameraya daha yakın
        # Arka kenar: kutunun sol/sağ kenarından geçen zemin çizgileri kameradan uzaklaştıkça açılır (perspektif).
        # Kameranın dibinden çıkan ışınlar gibi açıyoruz; gerçek açılma bundan biraz az, yani güvenli taraf.
        k = (dist + d) / dist
        poly = np.array([L - u * back, R - u * back, mid + (R - mid) * k + u * d, mid + (L - mid) * k + u * d])
        h = spec["h"]
        if cut_top:
            h = INF                            # tepesini göremiyoruz: ne kadar yüksek bilmiyoruz
        elif self.cam is not None and math.isfinite(h) and not cut_bottom:
            # Kutunun tepesinden geçen ışın, engelin ön yüzünün uzaklığında hangi yükseklikte?
            r = self.cam.ray((x1 + x2) / 2, y1)
            s = dist / max(np.hypot(r[0], r[1]), 1e-9)
            top = self.cam.alt + s * r[2]
            h = max(top, 0.5) * SAFETY_H + 0.5
        return poly, h

    def _flyer(self, det, spec):
        if self.cam is None:
            return None
        x1, y1, x2, y2 = det.box
        r = self.cam.ray((x1 + x2) / 2, (y1 + y2) / 2)     # kamera ekseninde 1 birim ilerleyen ışın
        # derinlik = odak * gerçek boyut / piksel boyut (kenardan kesikse diğer kenar sağlamdır: büyüğünü al)
        depth = self.cam.fx * spec["size"] / max(x2 - x1, y2 - y1, 1.0)
        P = self.cam.C + r * depth
        if P[1] < 0:
            return None
        rad = spec["size"] / 2 + 0.5
        a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        poly = np.c_[P[0] + rad * np.cos(a), P[1] + rad * np.sin(a)]
        # Uçan şey: boyu = tepesinin yüksekliği (altından geçmek riskli, o yüzden kural INF ise INF kalır)
        h = spec["h"] if not math.isfinite(spec["h"]) else (P[2] + spec["size"] / 2) * SAFETY_H + 0.5
        return poly, h


# ---------------- Tespitler → kuşbakışı maske + boy ----------------
class YoloDetector:
    """
    Kare → (engel maskesi, boy görüntüsü), ikisi de kuşbakışı FRAME_H x FRAME_W. drone_vision.ObstacleDetector
    'yolo' ve 'hibrit' yöntemlerinde bunu kullanır.
      projector yok : kamera kuşbakışı (sim / tepeden çekim). Maske = segmentasyon (yoksa kutu), boy = sınıftan.
      projector var : eğik kamera. Her tespit zemine indirilir (GroundProjector.footprint).
    """

    def __init__(self, backend, projector=None):
        self.backend, self.projector = backend, projector
        self.dets, self.items = [], []          # son karenin tespitleri ve (sınıf, boy, bev poligonu)

    @property
    def perspective(self):
        return self.projector is not None

    def process(self, frame):
        self.dets = self.backend(frame)
        items = []
        for d in self.dets:
            if self.projector is None:
                x1, y1, x2, y2 = d.box
                poly = d.poly if d.poly is not None else \
                    np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                h = CLASSES[d.cls]["h"]
            else:
                fp = self.projector.footprint(d)
                if fp is None:
                    continue
                poly, h = ground_to_bev(fp[0], self.projector.cell_m), fp[1]
            items.append((d.cls, h, np.round(poly).astype(np.int32)))
        mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
        height = np.zeros((FRAME_H, FRAME_W), np.float32)
        for cls, h, poly in sorted(items, key=lambda t: t[1]):   # alçaktan yükseğe: üst üste binen yerde yüksek kalır
            cv2.fillPoly(mask, [poly], 255)
            cv2.fillPoly(height, [poly], float(min(h, BIG)))
        self.items = items
        return mask, np.where(height >= BIG, np.inf, height)

    def bev_view(self, frame):
        """Eğik kamera görüntüsünü zemine ser (kuşbakışı); ufkun üstü ve görünmeyen yer siyah."""
        if self.projector is None:
            return frame
        Hb = self.projector.bev_homography()
        mask = np.zeros(frame.shape[:2], np.uint8)
        top = int(max(self.projector._horizon() + 4, 0))       # ufka çok yakın satırlar sonsuza gider
        mask[top:] = 255
        return cv2.warpPerspective(cv2.bitwise_and(frame, frame, mask=mask), Hb, (FRAME_W, FRAME_H))

    def annotate_bev(self, view):
        """Kuşbakışı görünümde her engelin yanına sınıf + boy yaz."""
        for cls, h, poly in self.items:
            x, y = poly.mean(axis=0).astype(int)
            t = f"{cls} {'inf' if not math.isfinite(h) else f'{h:.0f}m'}"
            cv2.putText(view, t, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, t, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLASS_COLOR[cls], 1, cv2.LINE_AA)


def draw_detections(img, dets):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        col = CLASS_COLOR[d.cls]
        if d.poly is not None:
            ov = out.copy()
            cv2.fillPoly(ov, [d.poly], col)
            out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)
    return out


# ---------------- Eğik kamerayla simülasyon görüntüsü ----------------
SKY, FAR_GROUND = (215, 190, 150), (80, 140, 95)


def render_oblique(scene, cam, cell_m=CELL_M, rng=None):
    """
    Kuşbakışı SimScene'i eğik kameradan çiz: zemin dokusu homografiyle, nesneler boylarıyla prizma olarak
    (uzaktan yakına, ressam algoritması). Sahnenin metre ölçeği cell_m'den: bir piksel cell_m / CELL_PX metre.
    """
    img = np.empty((cam.h, cam.w, 3), np.uint8)
    hz = int(np.clip(cam.horizon_v(), 0, cam.h))
    img[:hz], img[hz:] = SKY, FAR_GROUND
    # zemin: grid pikseli → zemin → kamera pikseli
    s = cell_m / CELL_PX
    G = np.array([[s, 0, -FRAME_W / 2 * s], [0, -s, FRAME_H * s], [0, 0, 1]])
    Hg = np.linalg.inv(cam.ground_homography()) @ G
    cv2.warpPerspective(scene.ground, Hg, (cam.w, cam.h), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    objs = []
    for obj in scene.objs:
        g = np.vstack([bev_to_ground(p, cell_m) for p, _, _, _ in obj["parts"]])
        objs.append((np.hypot(*g.mean(axis=0)), obj))
    for _, obj in sorted(objs, key=lambda t: -t[0]):
        for poly, z0, z1, col in obj["parts"]:
            g = bev_to_ground(poly, cell_m)
            if obj["cls"] == "kus":                      # kuş: CLASSES boyutunda bir top
                c = np.r_[g.mean(axis=0), (z0 + z1) / 2]
                px, z = cam.project(c[None])
                if z[0] > 0.5:
                    r = max(int(cam.fx * CLASSES["kus"]["size"] / 2 / z[0]), 2)
                    cv2.circle(img, (int(px[0, 0]), int(px[0, 1])), r, col, -1, cv2.LINE_AA)
                continue
            base, zb = cam.project(np.c_[g, np.full(len(g), z0)])
            top, zt = cam.project(np.c_[g, np.full(len(g), z1)])
            ok = np.r_[zb, zt] > 0.5
            pts = np.vstack([base, top])[ok]
            if len(pts) < 3:
                continue
            side = tuple(int(c * 0.65) for c in col)
            cv2.fillConvexPoly(img, cv2.convexHull(pts.astype(np.int32)), side, cv2.LINE_AA)
            if zt.min() > 0.5:
                cv2.fillPoly(img, [top.astype(np.int32)], col, cv2.LINE_AA)
    if rng is not None:                                  # hafif kamera gürültüsü
        img = cv2.add(img, _noise(rng, img.shape, 1), dtype=cv2.CV_8U)
        img = cv2.subtract(img, _noise(rng, img.shape, 2), dtype=cv2.CV_8U)
    return img


_NOISE = {}


def _noise(rng, shape, k):
    """Kamera gürültüsü (|N(0,3)|): her karede yeniden üretmek yavaş, bir kere üretip rastgele kaydırarak kullan."""
    key = (shape, k)
    if key not in _NOISE:
        big = (shape[0] + 64, shape[1] + 64, shape[2])
        _NOISE[key] = np.clip(np.abs(rng.normal(0, 3, big)) * (rng.random(big) < 0.5), 0, 255).astype(np.uint8)
    dy, dx = rng.integers(0, 64, 2)
    return np.ascontiguousarray(_NOISE[key][dy:dy + shape[0], dx:dx + shape[1]])


# ---------------- Tek başına demo: resim / video ----------------
def build_projector(args):
    if not args.perspektif:
        return None
    cam = Camera(args.irtifa, args.egim, args.fov, args.vfov)
    if args.kalibrasyon:
        return GroundProjector.from_file(args.kalibrasyon, cam, args.hucre_m)
    return GroundProjector.from_camera(cam, args.hucre_m)


class Demo:
    """
    Fotoğraf/video demosu: YOLO → grid → D* Lite, drone rota boyunca uçar.
    Drone (kamera) altta ortada, hedef yukarıda ortada. Fotoğrafta tespit bir kere yapılır, sonra
    sadece drone ilerler; videoda her karede tespit + artımlı yeniden planlama.
    Sol tık: hedef   Sağ tık: drone   SPACE: dur/devam   R: baştan   Q/ESC: çık
    """
    START, GOAL = (COLS // 2, ROWS - 2), (COLS // 2, 2)

    def __init__(self, yd, use3d, source="resim"):
        self.yd, self.use3d, self.source = yd, use3d, source
        self.frame = self.bev = None
        self.reset()

    def reset(self):
        import drone_vision as V
        if self.use3d:
            from drone_plan3d import GridPlanner3D
            self.gp = GridPlanner3D(self.START, self.GOAL, predict=False)
        else:
            self.gp = V.GridPlanner(self.START, self.GOAL, predict=False)
        self.gp.tracker = type("T", (), {"tracks": []})()     # draw() iz listesi istiyor; burada takip yok
        self.gp.last_mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
        self.first = True
        if self.frame is not None:
            self._apply()

    def detect(self, frame):
        """Yeni kare: YOLO + zemine indirme + planlayıcıya ver."""
        self.frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        self.mask, self.height = self.yd.process(self.frame)
        self.bev = self.yd.bev_view(self.frame)
        self._apply()

    def _apply(self):
        import drone_vision as V
        gp = self.gp
        raw = V.ObstacleDetector.to_cells(self.mask)
        h = np.where(self.mask > 0, self.height, 0).reshape(ROWS, CELL_PX, COLS, CELL_PX).max(axis=(1, 3))
        gp.unseen = self.yd.projector.unseen if self.yd.perspective else None
        if self.use3d:
            gp.update(raw, np.where(raw, h, 0.0))
            if self.first:
                gp.set_goal(gp.goal)           # hedefte boş olan en alçak katman (engel bilgisi geldikten sonra)
        else:
            gp.update(raw)
        gp.last_mask = self.mask
        self.first = False
        self._status()
        self._retarget(self.source == "resim")

    def _status(self):
        gp = self.gp
        if gp.drone == gp.goal:
            gp.status = "Hedefe varildi (sol tik: yeni hedef, R: bastan)"
        elif gp.path is None:
            gp.status = "Rota yok: hedef kapali (baska hedef sec)"
        else:
            gp.status = f"Ucusta, kalan {len(gp.path) - 1} adim"
        un = getattr(gp, "unseen", None)
        if un is not None and (un[gp.drone[1], gp.drone[0]] or un[gp.goal[1], gp.goal[0]]):
            gp.status += "  | DIKKAT: kameranin gormedigi yer (siyah), orasi bos sayiliyor"

    def _retarget(self, now):
        """Hedef her irtifada kapalıysa en yakın boş hücreye taşı (fotoğrafta hemen, videoda biraz bekleyip)."""
        import drone_vision as V
        if V.keep_goal_reachable(self.gp, 1 if now else V.GOAL_PATIENCE):
            msg = self.gp.status
            self._status()
            self.gp.status = msg + f"  | kalan {len(self.gp.path) - 1} adim" if self.gp.path else msg

    def step(self):
        if self.gp.path and len(self.gp.path) > 1:
            self.gp.step_drone()
            self._status()

    def click(self, event, x, y):
        if x >= FRAME_W or y >= FRAME_H:
            return
        c = (x // CELL_PX, y // CELL_PX)
        if event == cv2.EVENT_LBUTTONDOWN and c != self.gp.drone:
            self.gp.set_goal(c)
        elif event == cv2.EVENT_RBUTTONDOWN and c != self.gp.goal:
            self.gp.set_drone(c)
        self._status()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._retarget(True)

    def render(self, paused=False):
        import drone_vision as V

        class _Det:          # draw() algılayıcıdan sadece yöntem adını istiyor
            method = "yolo"
        return V.draw(self.bev, self.gp, _Det(), self.source, paused,
                      preview=draw_detections(self.frame, self.yd.dets), yolo=self.yd)


def demo_frame(frame, yd, use3d):
    """Tek kare: tespit + rota çizimi (drone uçmaz)."""
    d = Demo(yd, use3d)
    d.detect(frame)
    return d.render()


def run_demo(args, yd):
    win = "DroneRoute YOLO  (sol tik: hedef, sag tik: drone, SPACE: dur, R: bastan, Q: cik)"
    demo = Demo(yd, args.uc_boyut, "resim" if args.resim else "video")
    cap = None
    if args.resim:
        img = cv2.imread(args.resim)
        if img is None:
            raise SystemExit(f"[!] okunamadı: {args.resim}")
        demo.detect(img)
        print(f"{len(yd.dets)} tespit: " + ", ".join(f"{d.cls} {d.conf:.2f}" for d in yd.dets))
        cv2.imwrite(args.out, demo.render())
        print(f"kaydedildi: {args.out}")
    else:
        cap = cv2.VideoCapture(args.video)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, f, _: demo.click(e, x, y))
    paused, n = False, 0
    while True:
        if cap is not None and not paused:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # video bitti, başa sar
                ok, img = cap.read()
                if not ok:
                    break
            demo.detect(img)
        if not paused and n % 3 == 0:                          # her 3 çizimde bir adım (~5 hücre/sn)
            demo.step()
        n += 1
        cv2.imshow(win, demo.render(paused))
        k = cv2.waitKey(60) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("r"):
            demo.reset()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="DroneRoute Adım 5: YOLO algılama + perspektif → grid")
    ap.add_argument("--model-hazirla", action="store_true", help="sözlüklü YOLOE modelini indir/hazırla")
    ap.add_argument("--resim", help="fotoğraf")
    ap.add_argument("--video", help="video")
    ap.add_argument("--model", default=None, help="YOLO ağırlıkları (.pt); varsayılan models/droneroute_yoloe.pt")
    ap.add_argument("--guven", type=float, default=0.25, help="en düşük güven (0-1)")
    add_camera_args(ap)
    ap.add_argument("--3d", dest="uc_boyut", action="store_true", help="irtifalı planlama")
    ap.add_argument("--out", default="yolo_demo.png", help="resim modunda ilk karenin PNG'si")
    args = ap.parse_args()
    if args.model_hazirla:
        prepare_model()
        return
    if not (args.resim or args.video):
        ap.error("--resim ya da --video ver (ya da --model-hazirla)")
    run_demo(args, YoloDetector(UltralyticsBackend(args.model, args.guven), build_projector(args)))


def add_camera_args(ap):
    """Perspektif ayarları (drone_vision.py de kullanıyor)."""
    ap.add_argument("--perspektif", action="store_true", help="kamera eğik bakıyor: tespitleri zemine indir")
    ap.add_argument("--irtifa", type=float, default=25.0, help="kamera yüksekliği (m)")
    ap.add_argument("--egim", type=float, default=35.0, help="kamera aşağı eğimi (derece, 90 = tam tepeden)")
    ap.add_argument("--fov", type=float, default=82.0, help="yatay görüş açısı (derece)")
    ap.add_argument("--vfov", type=float, default=None, help="dikey görüş açısı (16:9 video için ver)")
    ap.add_argument("--hucre-m", type=float, default=CELL_M, help="eğik kamerada bir hücre kaç metre")
    ap.add_argument("--kalibrasyon", help="zemin kalibrasyonu JSON (4+ görüntü/zemin noktası)")


if __name__ == "__main__":
    main()
