"""
DroneRoute — Adım 2.5: hareketli engel takibi + kısa vadeli tahmin (sıfırdan)

Fikir: Algılama bize her karede "şu pikseller engel" diyor ama hangisinin hareket ettiğini,
nereye gittiğini bilmiyor. Drone sadece şu anki engellere bakarsa yoldaki kamyon gelip üstüne çıkar.
Bu modül her karede:

  1) Sabit harita : uzun süre (N_STATIC kare) hep dolu kalan pikseller sabit (bina, ağaç, vinç kulesi).
                    Kalan "hareketli maske"de sadece kamyon, vinç kolu, kuş gibi şeyler kalır.
                    Böylece vinç kolu binanın üstünden geçerken ikisi tek leke olup takibi bozmaz.
  2) Lekeler      : hareketli maskenin bağlı bileşenleri (OpenCV sadece etiketleme için).
                    Her lekenin ağırlık merkezi, alanı ve yönelimi (ikinci momentlerden) bizde.
  3) Eşleştirme   : önceki izlerin bu karedeki tahmini merkezine en yakın leke (açgözlü, kapılı).
  4) Hareket      : her iz rijit cisim gibi: merkez hızı v (piksel/kare) + açısal hız w (rad/kare).
                    Kamyon için w ~ 0, vinç kolu için w != 0. Ölçümler üstel ortalamayla yumuşatılır.
                    Leke birleşir/bölünürse (alan birden değişirse) hız güncellenmez, eskisiyle devam.
  5) Tahmin       : her izin piksellerini h kare ileri taşı:  p(h) = c + v*h + R(w*h) (p - c)
                    ve her hücre için "engel buraya kaç kare sonra geliyor" (tti) haritası çıkar.

Planlayıcı (drone_vision.VisionPlanner) tti'yi şöyle kullanır: yakında gelecek hücreler (tti <= H_HARD)
duvar sayılır, biraz daha uzaktakiler (tti <= HORIZON) pahalı hücre olur, D* Lite rotayı ona göre kurar.
"""
import math

import cv2
import numpy as np

from drone_dstar_lite import COLS, ROWS

# ---------------- Ayarlar ----------------
N_STATIC = 60          # piksel net bu kadar kare dolu kalırsa sabit sayılır
STATIC_DECAY = 4       # boş görülen karede sayaç bu kadar düşer (sabit şey kalkarsa çabuk unut)
MIN_MOVING_AREA = 80   # bundan küçük hareketli leke gürültü (piksel)
GATE_PX = 40           # tahmini merkezden bu kadar uzaktaki leke o ize eşleşmez (+ hıza göre büyür)
MAX_MISS = 10          # iz bu kadar kare görülmezse silinir (görülmediği sürece tahminle süzülür)
CONFIRM = 3            # iz bu kadar kare görülünce "gerçek" sayılır (tek karelik gürültü tahmin üretmesin)
ALPHA = 0.5            # hız yumuşatma (1 = sadece son ölçüm)
MIN_SPEED = 0.35       # en hızlı noktası bundan yavaşsa (piksel/kare) iz duruyor sayılır
ELONG_MIN = 0.3        # yönelim ancak bu kadar uzunsa güvenilir (daire/kare için açı anlamsız)
SAMPLE = 3             # tahminde her lekeden kaç pikselde bir nokta alınsın
HORIZON = 20           # kaç kare ileriye bakıyoruz
V_MAX = 3.5            # kadraja yeni giren, hızı henüz bilinmeyen cisim için varsayılan en yüksek hız (piksel/kare)
H_STEP = 2             # tahmin adımı (kare)


def wrap_half_pi(a):
    """Yönelim 180 derece belirsiz: farkı (-90, 90] derece aralığına getir."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


class Blob:
    """Bir karedeki hareketli leke: merkez, alan, yönelim ve (merkeze göre) örnek noktaları."""

    def __init__(self, xs, ys):
        self.area = len(xs)
        cx, cy = xs.mean(), ys.mean()
        self.c = np.array([cx, cy])
        dx, dy = xs - cx, ys - cy
        mu20, mu02, mu11 = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
        self.theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)       # ana eksen açısı
        spread = mu20 + mu02
        self.elong = math.hypot(mu20 - mu02, 2 * mu11) / spread if spread > 0 else 0.0
        keep = ((xs % SAMPLE) == 0) & ((ys % SAMPLE) == 0)        # seyrek ızgara örneği
        self.pts = np.stack([dx[keep], dy[keep]], 1)
        self.radius = float(np.sqrt(dx * dx + dy * dy).max())
        self.at_edge = False      # kadraj kenarına değiyor mu (dışarıdan yeni giriyor olabilir)
        self.height = math.inf    # en yüksek pikselinin yüksekliği (metre); derinlik yoksa bilinmiyor


class Track:
    _next_id = 1

    def __init__(self, blob, v=None, w=0.0, hits=1):
        self.id = Track._next_id
        Track._next_id += 1
        self.blob = blob
        self.v = np.zeros(2) if v is None else np.array(v, float)
        self.w = w
        self.hits = hits
        self.misses = 0          # son görüldüğünden beri kaç kare geçti
        self.fresh = v is None   # henüz hiç hız ölçülmedi

    @property
    def confirmed(self):
        return self.hits >= CONFIRM

    def max_speed(self):
        """Cismin en hızlı noktasının hızı: merkez hızı + w * yarıçap."""
        return float(np.hypot(*self.v)) + abs(self.w) * self.blob.radius

    @property
    def moving(self):
        return self.max_speed() >= MIN_SPEED

    def center_at(self, h):
        """Son görüldüğü andan (misses + h) kare sonraki merkez."""
        return self.blob.c + self.v * (self.misses + h)

    def points_at(self, h):
        """Son görüldüğü andan (misses + h) kare sonra cismin noktaları (rijit hareket)."""
        t = self.misses + h
        a = self.w * t
        ca, sa = math.cos(a), math.sin(a)
        p = self.blob.pts
        rot = np.stack([ca * p[:, 0] - sa * p[:, 1], sa * p[:, 0] + ca * p[:, 1]], 1)
        return rot + self.center_at(h)

    def update(self, blob):
        dt = self.misses + 1
        old = self.blob
        ratio = blob.area / max(old.area, 1)
        if 0.75 < ratio < 1.33:
            # Alan aşağı yukarı aynı: aynı cisim, ölçüm güvenilir
            v_meas = (blob.c - old.c) / dt
            w_meas = 0.0
            if blob.elong > ELONG_MIN and old.elong > ELONG_MIN:
                w_meas = wrap_half_pi(blob.theta - old.theta) / dt
            if self.fresh:
                self.v, self.w, self.fresh = v_meas, w_meas, False
            else:
                self.v = ALPHA * v_meas + (1 - ALPHA) * self.v
                self.w = ALPHA * w_meas + (1 - ALPHA) * self.w
        # Alan birden değiştiyse leke birleşti/bölündü/kısmen örtüldü: merkez kayması hız değil,
        # o yüzden hızı ellemiyoruz, eski hızla devam.
        self.blob = blob
        self.hits += 1
        self.misses = 0


class StaticMap:
    """Piksel başına 'ne zamandır dolu' sayacı. Uzun süre dolu = sabit engel."""

    def __init__(self, shape):
        self.cnt = np.zeros(shape, np.int16)

    def update(self, mask):
        m = mask > 0
        self.cnt = np.where(m, np.minimum(self.cnt + 1, 2 * N_STATIC),
                            np.maximum(self.cnt - STATIC_DECAY, 0)).astype(np.int16)
        return self.cnt >= N_STATIC


class Tracker:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.cell = self.w / COLS
        self.static = StaticMap((self.h, self.w))
        self.tracks = []
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.moving_mask = np.zeros((self.h, self.w), np.uint8)
        self.tti_height = np.zeros((ROWS, COLS))   # tti'deki hücreye gelecek engelin yüksekliği (3D için)

    # --- 1 + 2: hareketli lekeler ---
    def blobs(self, mask, height=None):
        static = self.static.update(mask)
        mov = np.where(static, 0, mask).astype(np.uint8)
        mov = cv2.morphologyEx(mov, cv2.MORPH_OPEN, self.kernel)   # sabit engel kenarındaki kırpıntıları at
        self.moving_mask = mov
        n, labels = cv2.connectedComponents(mov, connectivity=8)
        if n <= 1:
            return []
        ys, xs = np.nonzero(labels)
        lab = labels[ys, xs]
        order = np.argsort(lab, kind="stable")
        ys, xs, lab = ys[order], xs[order], lab[order]
        bounds = np.searchsorted(lab, np.arange(1, n + 1))
        out = []
        for i in range(n - 1):
            a, b = bounds[i], bounds[i + 1]
            if b - a >= MIN_MOVING_AREA:
                bx, by = xs[a:b], ys[a:b]
                blob = Blob(bx.astype(np.float64), by.astype(np.float64))
                blob.at_edge = bool(bx.min() <= 2 or by.min() <= 2 or bx.max() >= self.w - 3 or by.max() >= self.h - 3)
                if height is not None:
                    blob.height = float(height[by, bx].max())
                out.append(blob)
        return out

    # --- 3 + 4: eşleştirme ve güncelleme ---
    def update(self, mask, height=None):
        blobs = self.blobs(mask, height)
        pred = [t.center_at(1) for t in self.tracks]
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = GATE_PX + 2 * float(np.hypot(*t.v)) * (t.misses + 1)
            for bi, b in enumerate(blobs):
                d = float(np.hypot(*(b.c - pred[ti])))
                if d < gate:
                    pairs.append((d, ti, bi))
        pairs.sort()
        used_t, used_b = set(), set()
        for d, ti, bi in pairs:                      # açgözlü: en yakın çift önce
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self.tracks[ti].update(blobs[bi])
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        # Eşleşmeyen leke: yeni iz. Bir izin tahmini gövdesinden kopmuşsa (vinç kolu binanın
        # arkasında ikiye bölündü gibi) o izin rijit hareketini miras alır, sıfırdan öğrenmez.
        new = []
        for bi, b in enumerate(blobs):
            if bi in used_b:
                continue
            parent = self._parent_of(b)
            if parent is None:
                new.append(Track(b))
            else:
                c_par = parent.center_at(0)
                r = b.c - c_par
                v = parent.v + parent.w * np.array([-r[1], r[0]])    # v + w x r
                new.append(Track(b, v=v, w=parent.w, hits=parent.hits))
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISS] + new
        return self.tti()

    def _parent_of(self, blob):
        best, best_d = None, self.cell
        for t in self.tracks:
            if not (t.confirmed and t.moving):
                continue
            pts = t.points_at(0)
            d = float(np.sqrt(((pts - blob.c) ** 2).sum(1)).min()) if len(pts) else math.inf
            if d < best_d:
                best, best_d = t, d
        return best

    # --- 5: tahmin ---
    def tti(self):
        """
        ROWS x COLS float: hareketli bir engelin o hücreye kaç kare sonra gireceği (yoksa inf).
        Bir noktanın düştüğü her hücre işaretlenir (algılamadaki %20 kuralından daha temkinli).
        Yanında self.tti_height: o hücreye gelecek izlerin en yükseği (iki iz aynı hücreye gelirse en erken
        zamanla en yüksek boy birlikte alınır: temkinli taraf).
        """
        out = np.full((ROWS, COLS), np.inf)
        hgt = np.zeros((ROWS, COLS))
        for t in self.tracks:
            if not t.confirmed and t.misses == 0 and t.blob.at_edge:
                one = np.full((ROWS, COLS), np.inf)
                self._entering(t, one)
                np.minimum(out, one, out=out)
                hgt = np.where(np.isfinite(one), np.maximum(hgt, t.blob.height), hgt)
            if not (t.confirmed and t.moving):
                continue
            for h in range(0, HORIZON + 1, H_STEP):
                p = t.points_at(h)
                cx = np.floor(p[:, 0] / self.cell).astype(np.int64)
                cy = np.floor(p[:, 1] / self.cell).astype(np.int64)
                ok = (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS)
                np.minimum.at(out, (cy[ok], cx[ok]), h)
                np.maximum.at(hgt, (cy[ok], cx[ok]), t.blob.height)
        self.tti_height = hgt
        return out

    def _entering(self, t, out):
        """
        Kadraj kenarında yeni beliren cisim (kamyon dışarıdan giriyor): hızını henüz bilmiyoruz.
        En kötüsünü varsay: her yöne V_MAX ile gelebilir. k hücre uzaktaki yere en erken k*hücre/V_MAX
        karede varır. Birkaç kare sonra iz onaylanınca gerçek hızıyla tahmin edilir.
        """
        p = t.points_at(0)
        cx = np.clip((p[:, 0] / self.cell).astype(np.int64), 0, COLS - 1)
        cy = np.clip((p[:, 1] / self.cell).astype(np.int64), 0, ROWS - 1)
        reach = np.zeros((ROWS, COLS), bool)
        reach[cy, cx] = True
        k = 0
        while k * self.cell / V_MAX <= HORIZON:
            np.minimum(out, np.where(reach, k * self.cell / V_MAX, np.inf), out=out)
            grown = reach.copy()                          # 8 komşuya bir hücre büyüt
            grown[1:] |= reach[:-1]
            grown[:-1] |= reach[1:]
            grown[:, 1:] |= grown[:, :-1].copy()
            grown[:, :-1] |= grown[:, 1:].copy()
            reach = grown
            k += 1
