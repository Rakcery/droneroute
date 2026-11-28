"""E-4 testi: PurePursuit rota takibi. Çalıştır: python tests/test_follow.py (sonda TAMAM yazmalı)"""
import math                                   # uzaklık
import os                                     # dosya yolları
import random                                 # rastgele haritalar
import sys                                    # içe aktarma yolu
import time                                   # çağrı süresi ölçümü

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # proje kökü
import numpy as np                            # tablolar (güvenlik payı için)
import drone_dstar_lite as M                  # A* ve ızgara boyu
from drone_follow import LOOKAHEAD, PurePursuit   # test edilen sınıf, varsayılan ileri bakış (m)
from drone_smooth import shortcut, smooth_path, speed_profile   # E-1 + E-3: uçuş rotası
from drone_vision import MARGIN, inflate      # planlayıcının güvenlik payı (hücre) ve şişirme
from sahte_veri import sonuc_yaz, RES         # S-6 ölçüm yazıcısı, hücre boyu (m)

DT = 0.05                                     # px4_bridge çağırma aralığı: 20 Hz (s)
MAX_ADIM = 4000                               # bir uçuş en fazla kaç adım sürebilir (200 s)
HARITA = 100                                  # kaç rastgele harita
LOOKAHEADS = (1.0, 2.0, 3.0)                  # denenen ileri bakış mesafeleri (m)


def harita(rng):
    """Rastgele dikdörtgen engeller -> (gerçek ızgara, paylı ızgara, A* rotası). Rota paylı ızgarada aranır."""
    bas, hedef = (1, M.ROWS // 2), (M.COLS - 2, M.ROWS // 2)   # soldan sağa uçuş
    while True:
        g = np.zeros((M.ROWS, M.COLS), bool)  # boş ızgara
        for _ in range(rng.randint(10, 25)):  # 10-25 engel
            x, y = rng.randrange(3, M.COLS - 3), rng.randrange(M.ROWS)   # sol üst köşe
            g[y:y + rng.randint(1, 6), x:x + rng.randint(1, 4)] = True    # 1-4 geniş, 1-6 uzun
        g[bas[1], bas[0]] = g[hedef[1], hedef[0]] = False   # kalkış ve hedef boş
        pay = inflate(g, MARGIN)              # planlayıcının gördüğü: engel + güvenlik payı
        pay[bas[1], bas[0]] = pay[hedef[1], hedef[0]] = False
        yol, _ = M.astar(pay.tolist(), bas, hedef)   # paylı ızgarada rota
        if yol:                               # rota varsa bu harita olur
            return g.tolist(), pay.tolist(), yol


def parcaya_uzaklik(p, a, b):
    """p noktasının a-b doğru parçasına en kısa uzaklığı (m)."""
    ab = [b[k] - a[k] for k in range(3)]      # parçanın yönü
    boy2 = sum(c * c for c in ab)             # boyun karesi
    t = 0.0 if boy2 == 0 else max(0.0, min(1.0, sum((p[k] - a[k]) * ab[k] for k in range(3)) / boy2))
    return math.dist(p, [a[k] + ab[k] * t for k in range(3)])   # en yakın noktaya uzaklık


def sapma(p, rota):
    """Drone'un rotaya en kısa uzaklığı (m). PurePursuit'ten bağımsız: bütün parçalara bakar."""
    return min(parcaya_uzaklik(p, a[:3], b[:3]) for a, b in zip(rota, rota[1:]))


def duvarda_mi(p, grid):
    """Metre konumu GERÇEK bir engel hücresinde ya da ızgara dışında mı?"""
    x = math.floor(p[0] / RES)                # hücre sütunu
    y = M.ROWS - 1 - math.floor(p[1] / RES)   # hücre satırı (y ters)
    return not (0 <= x < M.COLS and 0 <= y < M.ROWS) or grid[y][x]


def uc(rota, grid, lookahead):
    """Drone'u kalkıştan 20 Hz'de uçur -> (vardı mı, adım, max sapma, ort sapma, çarpma, ort çağrı ms)."""
    pp = PurePursuit(lookahead)               # takipçi
    pp.set_path(rota)                         # rotayı ver
    pos = list(rota[0][:3])                   # drone kalkışta
    sapmalar, carpma, sure = [], 0, 0.0       # ölçümler
    for adim in range(MAX_ADIM):              # her 0,05 s
        t0 = time.perf_counter()              # çağrı süresi başlasın
        hedef = pp.target(pos)                # nereye, ne hızla
        sure += time.perf_counter() - t0      # çağrı süresi
        if pp.done:                           # vardı
            break
        d = math.dist(pos, hedef[:3])         # havuca uzaklık
        s = min(d, hedef[3] * DT)             # bu adımda gidilecek yol (m)
        if d > 0:                             # havuca doğru yürü (basit drone modeli)
            pos = [pos[k] + (hedef[k] - pos[k]) * s / d for k in range(3)]
        sapmalar.append(sapma(pos, rota))     # rotadan ne kadar saptı
        carpma += duvarda_mi(pos, grid)       # gerçek engele girdi mi
    ort = sum(sapmalar) / max(len(sapmalar), 1)   # ortalama sapma
    return pp.done, adim, max(sapmalar, default=0.0), ort, carpma, 1000 * sure / (adim + 1)


# ---------------- 1) elle kontroller ----------------
pp = PurePursuit()                            # varsayılan lookahead
assert pp.lookahead == 2.0 and pp.target((0, 0, 5)) is None   # rota yok -> None
pp.set_path([(0.0, 0.0, 5.0, 0.0), (5.0, 0.0, 5.0, 3.0), (10.0, 0.0, 5.0, 0.0)])   # 10 m düz rota
assert pp.target((0, 0, 5)) == (0.5, 0.0, 5.0, 0.3)   # kalkışta: yavaş, havuç yakın (L_MIN), en düşük hız
assert pp.target((5, 0, 5)) == (7.0, 0.0, 5.0, 3.0)   # ortada: tam hız, havuç tam lookahead ileride
assert pp.target((9.9, 0, 5)) == (10.0, 0.0, 5.0, 0.0) and pp.done   # hedefte: dur, vardı
assert PurePursuit(lookahead=3.5).lookahead == 3.5    # lookahead ayarlanabilir
print("elle kontroller: 5/5")

# ---------------- 2) rastgele haritalarda uçuş ----------------
rng = random.Random(0)                        # tekrar üretilebilir
rotalar = []                                  # (uçuş rotası, gerçek ızgara)
for _ in range(HARITA):
    gercek, pay, yol = harita(rng)            # harita + rota
    rotalar.append((speed_profile(smooth_path(shortcut(yol, pay), pay, RES)), gercek))   # E-1 + E-3
satirlar = []                                 # S-6 ölçüm satırları
for la in LOOKAHEADS:                         # her ileri bakış için
    sonuc = [uc(r, g, la) for r, g in rotalar]   # hepsini uçur
    vardi = sum(s[0] for s in sonuc)          # kaç tanesi vardı
    sn = sum(s[1] for s in sonuc) * DT / HARITA   # ortalama uçuş süresi (s)
    mx = max(s[2] for s in sonuc)             # en kötü sapma (m)
    ort = sum(s[3] for s in sonuc) / HARITA   # ortalama sapma (m)
    carp = sum(s[4] for s in sonuc)           # gerçek engelde geçen toplam adım
    ms = max(s[5] for s in sonuc)             # en yavaş ortalama çağrı (ms)
    print(f"lookahead={la} m: vardi={vardi}/{HARITA} sapma max={mx:.2f} ort={ort:.3f} m, "
          f"carpma={carp}, ucus={sn:.1f} s, cagri={ms:.3f} ms")
    for metrik, deger, birim in (("max_sapma", mx, "m"), ("ort_sapma", ort, "m"), ("carpma", carp, "adim"),
                                 ("ucus_suresi", sn, "s"), ("cagri_suresi", ms, "ms")):
        satirlar.append(dict(kart="E-4", deney="takip", yontem=f"pure_pursuit_la{la}", sahne=f"{HARITA}_rastgele",
                             metrik=metrik, deger=round(deger, 4), birim=birim))
    assert vardi == HARITA                    # hepsi hedefe vardı
    if la <= LOOKAHEAD:                       # kullandığımız ayar ve daha kısası: güvenli olmak ZORUNDA
        assert carp == 0                      # çarpma = 0
        assert mx < MARGIN * RES              # sapma güvenlik payından küçük
    assert ms < 5.0                           # 20 Hz'e (50 ms) rahat yetişiyor
KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # proje kökü
sonuc_yaz(os.path.join(KOK, "sonuclar", "e-4_takip.csv"), satirlar)    # S-6: tablo
print("olcum yazildi: sonuclar/e-4_takip.csv")
print("TAMAM")
