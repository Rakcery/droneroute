"""
sahte_veri.py: Arayüz Sözleşmesi v1.0'a uyan SAHTE (mock) veriler.

Ne işe yarar: kimse kimseyi beklemesin diye. Enes, Yağmur'un algısı bitmeden planlayıcısını;
Kerem, Enes'in takip denetleyicisi bitmeden px4_bridge'ini; Lider, ölçümler gelmeden rapor
tablosunu bu sahte verilerle dener. Gerçek parça gelince sahte olanın yerine takılır; biçim aynı
olduğu için başka hiçbir şey değişmez.

Çalıştır:  python sahte_veri.py      (kendi kendini dener, sonda TAMAM yazar)
Sahibi: Lider (L-5). Değiştirmek = sözleşmeyi değiştirmek (docs/SOZLESME.md, S-7 kuralı).
"""
import csv                                   # S-6 ölçüm dosyası yazmak için
import json                                  # S-6 algı özeti yazmak için
import math                                  # sin, cos, hypot
import os                                    # klasör oluşturmak için

import numpy as np                           # sayı tabloları

from drone_dstar_lite import COLS, ROWS      # ızgara boyu: 40 sütun, 28 satır (S-0)

RES = 0.5                                    # kuşbakışında bir hücre 0,5 metre (codec.RESOLUTION ile aynı)
ALTS = (5, 10, 15, 20)                       # 3D katmanların uçuş irtifası, metre (drone_dstar3d.ALTS ile aynı)
KALKIS_HUCRE = (1, ROWS // 2)                # drone'un kalkış hücresi (S-4): (x=1, y=14)
SINIFLAR = ["vinc", "bina", "agac", "arac", "insan", "kus"]   # S-5 sınıf sırası: 0..5


# ---------------- S-0 yardımcıları: hücre <-> metre ----------------
def hucre_metre(hucre, res=RES):
    """Hücre (x, y) merkezinin map koordinatı (metre). codec.cell_to_xy ile aynı formül."""
    x, y = hucre[0], hucre[1]                # hücrenin sütunu ve satırı
    return (x + 0.5) * res, (ROWS - 1 - y + 0.5) * res   # y ekseni ters: görüntüde aşağı, haritada yukarı


KALKIS_XY = hucre_metre(KALKIS_HUCRE)        # kalkış noktasının map koordinatı: (0.75, 6.75) metre


# ---------------- S-1: algı çıktısı (Yağmur -> Enes) ----------------
def sahte_algi(kare=0):
    """
    Bir karelik sahte algı çıktısı: (raw, height, moving_tti, moving_height).
    Sahnede: 25 m'lik bir bina, 3 m'lik alçak bir duvar ve her 2 karede 1 hücre aşağı inen 4 m'lik kamyon.
    """
    raw = np.zeros((ROWS, COLS), bool)       # hepsi boş başla (True = şu an engel var)
    height = np.zeros((ROWS, COLS))          # boylar: 0 = boş hücre
    raw[5:12, 10:16] = True                  # bina: satır 5-11, sütun 10-15
    height[5:12, 10:16] = 25.0               # binanın boyu 25 m (hiçbir katmandan üstünden geçilmez)
    raw[18:24, 24:27] = True                 # alçak duvar: satır 18-23, sütun 24-26
    height[18:24, 24:27] = 3.0               # 3 m: 5 m'de uçan drone geçemez (3 + 2 pay = 5), 10 m'de geçer
    tti = np.full((ROWS, COLS), np.inf)      # hareketli engel tahmini: inf = hiç gelmiyor
    mh = np.zeros((ROWS, COLS))              # gelecek hareketli engelin boyu: 0 = gelmiyor
    bas = (kare // 2) % ROWS                 # kamyonun en üst satırı: her 2 karede bir aşağı kayar
    for y in range(bas, min(bas + 3, ROWS)): # kamyon 3 hücre boyunda
        raw[y, 30:32] = True                 # kamyonun şu an kapladığı hücreler (sütun 30-31)
        height[y, 30:32] = 4.0               # kamyon 4 m
    for y in range(bas + 3, ROWS):           # kamyonun önündeki (aşağıdaki) hücreler
        t = (y - (bas + 2)) * 2.0            # bu hücreye kaç kare sonra girer (1 hücre = 2 kare)
        if t <= 20:                          # 20 kareden (HORIZON) uzağı tahmin edilmez
            tti[y, 30:32] = t                # tahmini yaz
            mh[y, 30:32] = 4.0               # gelecek şeyin boyu 4 m
    return raw, height, tti, mh              # sözleşmedeki sırayla dön


# ---------------- S-2 / S-3: rota (Enes -> Kerem) ----------------
def sahte_rota_hucre():
    """3D hücre rotası [(x, y, z), ...]: kalkıştan sağa düz git, ortada 10 m'ye çık, sonra in."""
    x0, y0 = KALKIS_HUCRE                    # başlangıç hücresi
    rota = []                                # boş liste
    for x in range(x0, COLS - 2):            # soldan sağa
        z = 1 if 20 <= x <= 28 else 0        # ortada 1. katman (10 m), diğer yerde 0. katman (5 m)
        rota.append((x, y0, z))              # düğümü ekle
    return rota                              # gerçek planlayıcının gp.path çıktısıyla aynı biçim


def sahte_ucus_rotasi(v=2.0):
    """S-3 uçuş rotası [(x, y, z, v), ...]: metre ve m/s, map çerçevesi (ENU)."""
    out = []                                 # sonuç listesi
    for x, y, z in sahte_rota_hucre():       # her hücre düğümü için
        mx, my = hucre_metre((x, y))         # hücre merkezini metreye çevir
        out.append((mx, my, float(ALTS[z]), v))   # irtifa = katmanın metresi, hız sabit
    return out                               # E-3 speed_profile çıktısıyla aynı biçim


class SahteTakip:
    """
    S-3 takip denetleyicisinin sahte hâli. Enes'in PurePursuit sınıfıyla AYNI metotlar:
      set_path(waypoints)  -> rotayı ver
      target(pos)          -> (x, y, z, v) hedef nokta ya da None (rota yok / bitti)
      done                 -> hedefe vardı mı
    Bu sahte sürüm akıllı değil: drone'a en yakın rota noktasından bir sonrakini verir.
    """

    def __init__(self, lookahead=2.0):
        self.lookahead = lookahead           # gerçek sınıftaki ileri bakış mesafesi (burada kullanılmıyor)
        self.wps = []                        # rota noktaları
        self.done = False                    # henüz bitmedi

    def set_path(self, waypoints):
        self.wps = list(waypoints)           # rotayı kopyala
        self.done = len(self.wps) == 0       # boş rota = yapacak iş yok

    def target(self, pos):
        if not self.wps:                     # rota yoksa
            return None                      # hedef yok: px4_bridge yerinde bekletir
        d = [math.dist(pos[:3], w[:3]) for w in self.wps]   # her noktaya uzaklık
        i = int(np.argmin(d))                # en yakın noktanın sırası
        if i == len(self.wps) - 1 and d[i] < 0.5:           # son noktaya 0,5 m'den yakınsa
            self.done = True                 # vardık
        return self.wps[min(i + 1, len(self.wps) - 1)]      # bir sonraki noktayı hedef ver


# ---------------- S-4: konum ve çerçeve dönüşümü (Kerem -> herkes) ----------------
def enu_to_ned(x, y, z, kalkis=KALKIS_XY):
    """map (ENU, metre) -> PX4 yerel (NED, kalkış noktası orijin). Referans formül; gerçeği codec.py'de (K-4)."""
    kx, ky = kalkis                          # kalkış noktasının map koordinatı
    return (y - ky, x - kx, -z)              # kuzey = map y, doğu = map x, aşağı = -yukarı


def ned_to_enu(n, e, d, kalkis=KALKIS_XY):
    """PX4 yerel (NED) -> map (ENU). enu_to_ned'in tersi."""
    kx, ky = kalkis                          # kalkış noktası
    return (e + kx, n + ky, -d)              # doğu -> x, kuzey -> y, aşağı -> -z


def sahte_konum(t):
    """t saniyedeki sahte drone konumu (ENU, metre): kalkış noktası etrafında 2 m yarıçaplı daire, 5 m'de."""
    kx, ky = KALKIS_XY                       # dairenin merkezi kalkış noktasının 3 m sağı
    a = 0.3 * t                              # açı: saniyede 0,3 radyan döner
    return (kx + 3.0 + 2.0 * math.cos(a), ky + 2.0 * math.sin(a), 5.0)   # (x, y, z)


# ---------------- S-5: etiketli veri (Kerem <-> Yağmur) ----------------
def sahte_etiket_klasoru(yol, n=5, seed=0):
    """
    S-5 biçiminde n sahte görüntü + etiket yazar: yol/images/*.png, yol/labels/*.txt, yol/meta.json.
    Görüntüde yeşil zemin üstünde rastgele renkli kutular var; her kutu bir sınıf.
    """
    import cv2                               # sadece bu fonksiyon görüntü yazdığı için burada içe aktar
    rng = np.random.default_rng(seed)        # tekrar üretilebilir rastgelelik
    os.makedirs(os.path.join(yol, "images"), exist_ok=True)   # görüntü klasörü
    os.makedirs(os.path.join(yol, "labels"), exist_ok=True)   # etiket klasörü
    W, H = 640, 448                          # görüntü boyu (piksel)
    for i in range(n):                       # her görüntü için
        img = np.full((H, W, 3), (60, 140, 70), np.uint8)    # yeşil zemin (BGR)
        satirlar = []                        # bu görüntünün etiket satırları
        for _ in range(int(rng.integers(1, 5))):              # 1-4 nesne
            c = int(rng.integers(0, len(SINIFLAR)))           # rastgele sınıf numarası
            w, h = int(rng.integers(20, 120)), int(rng.integers(20, 120))   # kutu boyu
            x1, y1 = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))   # sol üst köşe
            renk = tuple(int(v) for v in rng.integers(0, 255, 3))            # rastgele renk
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), renk, -1)         # kutuyu çiz
            xc, yc = (x1 + w / 2) / W, (y1 + h / 2) / H       # merkez, 0-1 arasına ölçeklenmiş
            satirlar.append(f"{c} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")   # YOLO kutu satırı
        ad = f"sahte_{i:04d}"                # dosya adı (görüntü ve etiket aynı ad)
        cv2.imwrite(os.path.join(yol, "images", ad + ".png"), img)            # görüntüyü yaz
        with open(os.path.join(yol, "labels", ad + ".txt"), "w") as f:      # etiketi yaz
            f.write("\n".join(satirlar) + "\n")
    meta = dict(kaynak="sahte", irtifa_m=25.0, egim_derece=35.0, hfov_derece=82.0, lisans="ic-kullanim")
    with open(os.path.join(yol, "meta.json"), "w") as f:      # sahne bilgisi
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return n                                 # kaç görüntü yazıldı


# ---------------- S-6: ölçüm sonuçları (herkes -> Lider) ----------------
SONUC_SUTUNLARI = ["kart", "deney", "yontem", "sahne", "metrik", "deger", "birim"]   # CSV başlığı


def sonuc_yaz(yol, satirlar):
    """S-6 CSV yaz. satirlar: sözlük listesi, anahtarlar SONUC_SUTUNLARI."""
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)   # klasör yoksa oluştur
    with open(yol, "w", newline="", encoding="utf-8") as f:   # dosyayı aç
        w = csv.DictWriter(f, fieldnames=SONUC_SUTUNLARI)     # sütun adlarıyla yazıcı
        w.writeheader()                      # ilk satır: başlık
        w.writerows(satirlar)                # ölçüm satırları


def sahte_sonuc_csv(yol):
    """Lider'in rapor aracını denemesi için sahte bir ölçüm dosyası."""
    sonuc_yaz(yol, [                         # iki örnek satır
        dict(kart="Y-0", deney="algilama", yontem="renk", sahne="0", metrik="kesinlik", deger=0.97, birim="oran"),
        dict(kart="Y-0", deney="algilama", yontem="kenar", sahne="0", metrik="kesinlik", deger=0.81, birim="oran"),
    ])


def sahte_algi_ozet(yol):
    """Y-3 -> Enes algı özeti (S-6 JSON). Gerçek değerler Y-3'te ölçülür."""
    ozet = dict(model="sahte", en_kucuk_nesne_px=18, en_kucuk_nesne_m=1.5, onerilen_pay_hucre=1,
                not_="sahte veri: Y-3 bitince gerçek değerle değişir")
    with open(yol, "w", encoding="utf-8") as f:              # dosyayı aç
        json.dump(ozet, f, ensure_ascii=False, indent=1)     # yaz


# ---------------- Kendi kendini dene ----------------
if __name__ == "__main__":
    raw, h, tti, mh = sahte_algi(7)          # 7. kare
    assert raw.shape == (ROWS, COLS) and raw.dtype == bool    # S-1 biçimi doğru mu
    assert h.shape == tti.shape == mh.shape == (ROWS, COLS)   # hepsi 28 x 40
    from drone_plan3d import GridPlanner3D   # gerçek 3D planlayıcı
    gp = GridPlanner3D()                     # oluştur
    for k in range(40):                      # 40 kare boyunca sahte algıyı ver
        gp.update(*sahte_algi(k))            # S-2: planlayıcı sahte veriyi yiyebiliyor mu
        if k % 4 == 0:                       # 4 karede bir
            gp.step_drone()                  # drone bir adım ilerlesin
    print("planlayıcı durumu:", gp.status)   # ekrana durum yaz
    takip = SahteTakip()                     # sahte takip
    takip.set_path(sahte_ucus_rotasi())      # sahte rotayı ver
    print("ilk hedef:", takip.target((*KALKIS_XY, 5.0)))   # kalkıştaki drone için hedef
    n, e, d = enu_to_ned(*sahte_konum(1.0))  # dönüşüm
    assert all(abs(a - b) < 1e-9 for a, b in zip(ned_to_enu(n, e, d), sahte_konum(1.0)))   # gidip gelince aynı
    print("TAMAM")
