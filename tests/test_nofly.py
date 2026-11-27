"""Sanal yasak bölge testi (E-2). Çalıştır: python tests/test_nofly.py"""
import os, sys, random, time                       # dosya yolu, rastgele sayı, süre
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"   # ekran açma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))       # proje klasörü
import numpy as np                                 # sayı tabloları
import drone_vision as V                           # 2D planlayıcı + sahte sahne
import drone_plan3d as P                           # 3D planlayıcı
import drone_dstar_lite as M                       # 2D A* (karşılaştırma için)
import drone_dstar3d as D                          # 3D A* (karşılaştırma için)
from drone_nofly import nofly_mask                 # test edilen fonksiyon

SAHNE_2D = 6                                       # 2D sahne sayısı
SAHNE_3D = 3                                       # 3D sahne sayısı (yavaş olduğu için az)
KARE = 240                                         # sahne başına kare sayısı
DEGIS = 40                                         # bölgeler kaç karede bir değişir

# 1) nofly_mask: kaba kuvvetle (her hücreyi tek tek) aynı mı?
rng = np.random.default_rng(0)                     # sabit tohum: her çalıştırmada aynı sayılar
for deneme in range(200):
    zones = [((int(rng.integers(0, M.COLS)), int(rng.integers(0, M.ROWS))), int(rng.integers(0, 5)))
             for _ in range(int(rng.integers(0, 4)))]       # 0-3 rastgele daire
    pay = int(rng.integers(0, 3))                  # 0-2 hücre pay
    kaba = np.zeros((M.ROWS, M.COLS), bool)        # elle doldurulacak tablo
    for y in range(M.ROWS):
        for x in range(M.COLS):
            kaba[y, x] = any((x - cx) ** 2 + (y - cy) ** 2 <= (r + pay) ** 2 for (cx, cy), r in zones)
    assert (nofly_mask(zones, pay) == kaba).all(), (deneme, zones, pay)
print("nofly_mask: 200 rastgele deneme kaba kuvvetle ayni")

# 2) Drone bölgenin ortasındayken bölge açılırsa: en kısa yoldan çıkmalı, bir daha girmemeli
for r in (1, 2, 3):
    gp = V.GridPlanner(start=(20, 14), goal=(36, 14))   # drone ortada, hedef sağda
    gp.set_nofly([((20, 14), r)])                  # drone'un tam üstüne daire
    m = nofly_mask([((20, 14), r)], gp.margin)     # dairenin (pay dahil) tablosu
    icerde = []                                    # her adımda içeride mi
    for _ in range(30):
        gp.update(np.zeros((M.ROWS, M.COLS), bool))   # boş dünya, sadece yasak bölge
        icerde.append(bool(m[gp.drone[1], gp.drone[0]]))
        gp.step_drone()                            # bir adım yürü
    cikis = icerde.index(False)                    # kaçıncı adımda çıktı
    assert cikis <= r + gp.margin + 1, (r, cikis)  # en fazla yarıçap+pay+1 adımda çıkmalı
    assert not any(icerde[cikis:]), (r, icerde)    # çıktıktan sonra hiç girmemeli
    print(f"icerden cikis: r={r} -> {cikis} adimda cikti, geri girmedi")

# 3) Canlı sahne: bölgeler 40 karede bir açılıp kapanıyor. Drone dışarıdan içeri hiç girmemeli,
#    D* Lite her karede sıfırdan A* ile aynı maliyeti bulmalı.
def rastgele_bolgeler(rnd):
    """%70 ihtimalle 1-3 rastgele daire, yoksa bölge yok."""
    if rnd.random() < 0.3:
        return []
    return [((rnd.randrange(3, M.COLS - 3), rnd.randrange(3, M.ROWS - 3)), rnd.randint(1, 3))
            for _ in range(rnd.randint(1, 3))]

def sahne_kos(cls, seed, astar_ayni):
    """Bir sahneyi koşar -> (kontrol sayısı, bölgeye giriş sayısı, D*/A* uyuşmazlık sayısı)."""
    random.seed(seed)                              # sahnenin kendi rastgeleliği
    rnd = random.Random(1000 + seed)               # bölgeler için ayrı rastgele kaynak
    scene = V.SimScene(seed)                       # sahte kamera sahnesi
    det = V.ObstacleDetector("renk")               # renkle engel bulucu
    det.calibrate(scene.background())              # arka planı öğren
    vp = cls(det)                                  # planlayıcı
    kontrol = giris = uyusmaz = 0                  # sayaçlar
    zones = []                                     # şu anki bölgeler
    for f in range(KARE):
        if f % DEGIS == 0:                         # bölgeleri değiştir
            zones = rastgele_bolgeler(rnd)
            vp.set_nofly(zones)
        frame, _ = scene.render(); hm = scene.height_map; scene.step()
        vp.process(frame, hm)                      # algı + planlama
        kontrol += 1
        uyusmaz += 0 if astar_ayni(vp) else 1      # D* = A* mı
        if f % V.MOVE_EVERY == 0:
            m = nofly_mask(zones, vp.margin)       # bölgelerin tablosu (pay dahil)
            once = bool(m[vp.drone[1], vp.drone[0]])   # adımdan önce içeride mi
            if vp.step_drone():                    # hedefe vardıysa yeni hedef
                vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
            sonra = bool(m[vp.drone[1], vp.drone[0]])  # adımdan sonra içeride mi
            giris += (not once) and sonra          # dışarıdan içeri girdi mi
    return kontrol, giris, uyusmaz

def ayni_2d(vp):
    """2D: D* Lite rotasının maliyeti, sıfırdan A* ile aynı mı?"""
    ap, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
    if (vp.path is None) != (ap is None):
        return False
    return ap is None or abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(ap, vp.penalty)) <= 1e-9

def ayni_3d(vp):
    """3D: aynı karşılaştırma, irtifa katmanlarıyla."""
    a, _ = D.astar3(vp.grid, (*vp.drone, vp.z), (*vp.goal, vp.gz), vp.penalty)
    if (vp.path is None) != (a is None):
        return False
    return a is None or abs(vp.path_cost() - D.path_cost3(vp.grid, vp.penalty, a)) <= 1e-9

t0 = time.time()                                   # süre ölçümü başlasın
for ad, cls, n, ayni in (("2D", V.VisionPlanner, SAHNE_2D, ayni_2d), ("3D", P.VisionPlanner3D, SAHNE_3D, ayni_3d)):
    tk = tg = tu = 0                               # toplam sayaçlar
    for seed in range(n):
        k, g, u = sahne_kos(cls, seed, ayni)
        tk += k; tg += g; tu += u
    print(f"{ad}: {n} sahne, kontrol={tk} bolgeye_giris={tg} D*/A* uyusmazlik={tu}")
    assert tg == 0 and tu == 0                     # çarpma = 0 ve D* = A* gevşetilmez
print(f"sure: {time.time() - t0:.0f} s")
print("TAMAM")
