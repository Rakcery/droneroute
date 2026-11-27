"""E-1 testi: shortcut + line_free. Çalıştır: python tests/test_smooth.py (sonda TAMAM yazmalı)"""
import math                                   # floor ve hypot için
import os                                     # dosya yolları için
import random                                 # rastgele haritalar için
import sys                                    # içe aktarma yolu için
from fractions import Fraction as F           # küsuratsız kesirli sayı (EPS derdi yok)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # proje kökünü bul
import drone_dstar_lite as M                  # harita üretici ve A*
from drone_smooth import line_free, shortcut, smooth_path, speed_profile, V_MAX, A_MAX, A_YAN   # test edilenler
from sahte_veri import sonuc_yaz, hucre_metre, RES   # S-6 ölçüm yazıcısı, hücre -> metre

PARCA = 1680                                  # kaba kuvvet kontrolünde ip kaç parçaya bölünür


def kaba_kuvvet_bos_mu(a, b, grid):
    """line_free'den BAĞIMSIZ kontrol: ipi çok sık nokta ile tara, köşede iki yana da bak."""
    (x0, y0), (x1, y1) = a, b                 # uç hücreler
    for i in range(PARCA + 1):                # ipin üstündeki her nokta
        t = F(i, PARCA)                       # yolun yüzde kaçı, tam kesir olarak
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)   # noktanın konumu
        xs = [math.floor(x + F(1, 2))]        # noktanın düştüğü sütun
        ys = [math.floor(y + F(1, 2))]        # noktanın düştüğü satır
        if (x + F(1, 2)).denominator == 1 and x1 != x0:   # nokta tam dikey çizginin üstündeyse
            xs.append(xs[0] - 1)              # soldaki sütuna da değiyor
        if (y + F(1, 2)).denominator == 1 and y1 != y0:   # tam yatay çizginin üstündeyse
            ys.append(ys[0] - 1)              # üstteki satıra da değiyor
        for cx in xs:                         # değdiği her sütun
            for cy in ys:                     # ve her satır için
                if grid[cy][cx]:              # hücre doluysa
                    return False              # ip engele değiyor
    return True                               # hiçbir dolu hücreye değmedi


def uzunluk(rota):
    """Rotanın kuş uçuşu toplam uzunluğu (hücre)."""
    return sum(math.dist(a, b) for a, b in zip(rota, rota[1:]))   # ardışık noktalar arası


def bos_harita(n):
    """n x n boş harita."""
    return [[False] * n for _ in range(n)]    # hepsi False = boş


# ---------------- 1) Elle 3 küçük kontrol ----------------
g = bos_harita(5)                             # boş harita
assert shortcut([(0, 0), (1, 0), (2, 0), (3, 1), (4, 2)], g) == [(0, 0), (4, 2)]   # zikzak tek çizgi olur
g[1][2] = True                                # ortaya duvar: (x=2, y=1)
assert shortcut([(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (4, 2)], g) == [(0, 0), (3, 0), (4, 2)]   # duvarın kenarından
g = bos_harita(3)                             # köşe testi için küçük harita
g[0][1] = g[1][0] = True                      # (1,0) ve (0,1) duvar
assert not line_free((0, 0), (1, 1), g)       # iki duvarın köşesinden geçmek yasak
print("elle kontroller: 3/3")                 # ilk bölüm geçti

# ---------------- 2) line_free = kaba kuvvet ----------------
random.seed(0)                                # her çalıştırmada aynı rastgelelik
for _ in range(2000):                         # 2000 rastgele ip
    g = [[random.random() < 0.2 for _ in range(8)] for _ in range(6)]   # 8x6 rastgele harita
    a = (random.randrange(8), random.randrange(6))   # rastgele başlangıç
    b = (random.randrange(8), random.randrange(6))   # rastgele bitiş
    assert line_free(a, b, g) == kaba_kuvvet_bos_mu(a, b, g), (a, b)   # iki yöntem aynı demeli
print("line_free = kaba kuvvet: 2000/2000")   # ikinci bölüm geçti

# ---------------- 3) 300 rastgele harita ----------------
once_nokta = sonra_nokta = 0                  # toplam nokta sayısı, önce / sonra
once_uz = sonra_uz = 0.0                      # toplam uzunluk, önce / sonra
for deneme in range(300):                     # 300 harita
    random.seed(deneme)                       # tekrar üretilebilir
    grid, bas, hedef = M.make_map()           # projenin rastgele haritası (yolu var)
    rota, _ = M.astar(grid, bas, hedef)       # D* Lite = A* olduğu için A* rotası yeterli
    kisa = shortcut(rota, grid)               # kısalt
    assert kisa[0] == rota[0] and kisa[-1] == rota[-1]   # baş ve son aynı kalmalı
    assert all(p in rota for p in kisa)       # sadece rotadaki noktalardan oluşmalı (alt küme)
    for a, b in zip(kisa, kisa[1:]):          # kısaltılmış rotanın her parçası
        assert kaba_kuvvet_bos_mu(a, b, grid), (deneme, a, b)   # hiçbir dolu hücreden geçmiyor
    assert uzunluk(kisa) <= uzunluk(rota) + M.EPS        # asla daha uzun değil
    once_nokta += len(rota); sonra_nokta += len(kisa)    # nokta sayılarını topla
    once_uz += uzunluk(rota); sonra_uz += uzunluk(kisa)  # uzunlukları topla
print(f"300 harita: nokta {once_nokta / 300:.1f} -> {sonra_nokta / 300:.1f}, "
      f"uzunluk {once_uz / 300:.2f} -> {sonra_uz / 300:.2f} hucre")   # ortalama kazanç
KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # proje kökü
satir = dict(kart="E-1", deney="kisaltma", sahne="300_rastgele")         # her satırda ortak alanlar
sonuc_yaz(os.path.join(KOK, "sonuclar", "e-1_kisaltma.csv"), [          # S-6: ölç, sonra iddia et
    dict(satir, yontem="astar", metrik="nokta_sayisi", deger=round(once_nokta / 300, 2), birim="adet"),
    dict(satir, yontem="astar+shortcut", metrik="nokta_sayisi", deger=round(sonra_nokta / 300, 2), birim="adet"),
    dict(satir, yontem="astar", metrik="uzunluk", deger=round(once_uz / 300, 3), birim="hucre"),
    dict(satir, yontem="astar+shortcut", metrik="uzunluk", deger=round(sonra_uz / 300, 3), birim="hucre"),
])
print("olcum yazildi: sonuclar/e-1_kisaltma.csv")   # Lider'in rapor tablosu için
# ---------------- 4) E-3: yumuşak eğri + hız planı ----------------
TOL = 1e-9                                    # kayan nokta payı


def egri_bos_mu(noktalar, grid):
    """smooth_path'ten BAĞIMSIZ kontrol: metre noktalarını hücreye çevir, aralarını 0,005 hücrede bir tara."""
    for (ax, ay, _), (bx, by, _) in zip(noktalar, noktalar[1:]):   # art arda iki eğri noktası
        n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / RES / 0.005))   # kaç küçük adım
        for k in range(n + 1):                # her küçük adım
            mx = ax + (bx - ax) * k / n       # metre x
            my = ay + (by - ay) * k / n       # metre y
            x = math.floor(mx / RES)          # hücre sütunu
            y = M.ROWS - 1 - math.floor(my / RES)   # hücre satırı (y ters)
            if not (0 <= x < M.COLS and 0 <= y < M.ROWS) or grid[y][x]:
                return False                  # dolu ya da dışarıda
    return True                               # temiz


toplam_nokta = 0                              # eğri nokta sayısı toplamı
toplam_sure = 0.0                             # uçuş süresi toplamı (s)
for deneme in range(300):                     # aynı 300 harita
    random.seed(deneme)                       # tekrar üretilebilir
    grid, bas, hedef = M.make_map()           # harita
    rota, _ = M.astar(grid, bas, hedef)       # hücre rotası
    egri = smooth_path(shortcut(rota, grid), grid, RES)   # kısalt + yumuşat (metre)
    assert egri_bos_mu(egri, grid), deneme    # güvenlik payına taşmıyor
    assert math.dist(egri[0][:2], hucre_metre(bas)) < TOL     # tam kalkışta başlıyor
    assert math.dist(egri[-1][:2], hucre_metre(hedef)) < TOL  # tam hedefte bitiyor
    ucus = speed_profile(egri)                # hız planı
    assert all(len(p) == 4 for p in ucus)     # S-3 biçimi: (x, y, z, v)
    assert ucus[0][3] == 0 and ucus[-1][3] == 0   # ilk ve son hız 0
    assert all(0 <= p[3] <= V_MAX + TOL for p in ucus)   # hız sınırı
    for p, q in zip(ucus, ucus[1:]):          # art arda iki nokta: ivme sınırı
        d = math.dist(p[:3], q[:3])           # aradaki mesafe (metre)
        assert abs(q[3] ** 2 - p[3] ** 2) <= 2 * A_MAX * d + TOL, (deneme, p, q)   # v² = v0² + 2ad
        if p[3] + q[3] > 0:                   # duran iki nokta değilse
            toplam_sure += 2 * d / (p[3] + q[3])   # ortalama hızla geçen süre
    for a, b, c in zip(ucus, ucus[1:], ucus[2:]):   # virajlar: yana ivme sınırı
        cap = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))   # 2 x üçgen alanı
        if cap > 1e-12:                       # düz değilse
            r = math.dist(a[:3], b[:3]) * math.dist(b[:3], c[:3]) * math.dist(c[:3], a[:3]) / (2 * cap)   # yarıçap
            assert b[3] ** 2 / r <= A_YAN + 1e-6, (deneme, b, r)   # v²/r <= A_YAN
    toplam_nokta += len(egri)                 # topla
# keskin köşe: L rotasında köşedeki hız tam hızdan düşük olmalı
bos = [[False] * 30 for _ in range(30)]       # boş harita
L = speed_profile(smooth_path([(0, 0), (20, 0), (20, 20)], bos, RES))   # sağa 20, aşağı 20
assert min(p[3] for p in L[6:-6]) < V_MAX     # köşede yavaşladı
print(f"E-3: 300 harita, egri payi asmadi, ilk/son hiz 0, ivme sinirli; "
      f"ortalama {toplam_nokta / 300:.1f} nokta, {toplam_sure / 300:.1f} s ucus")
sonuc_yaz(os.path.join(KOK, "sonuclar", "e-3_yumusak_rota.csv"), [      # S-6 ölçüm
    dict(kart="E-3", deney="yumusak_rota", yontem="shortcut+bspline", sahne="300_rastgele",
         metrik="nokta_sayisi", deger=round(toplam_nokta / 300, 2), birim="adet"),
    dict(kart="E-3", deney="yumusak_rota", yontem="shortcut+bspline+hiz", sahne="300_rastgele",
         metrik="ucus_suresi", deger=round(toplam_sure / 300, 2), birim="s"),
])
print("olcum yazildi: sonuclar/e-3_yumusak_rota.csv")

print("TAMAM")                                # hepsi geçti
