"""Hareketli engel takibi + kaçınma testi. Çalıştır: python tests/test_moving.py  (birkaç dakika sürer)"""
import os, sys, random, math
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import drone_vision as V
import drone_dstar_lite as M

SEEDS = range(8)
FRAMES = 600


def setup(seed, predict=True):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector("renk")
    det.calibrate(scene.background())
    return scene, V.VisionPlanner(det, predict=predict)


def nearest_track(vp, xy):
    best = None
    for t in vp.tracker.tracks:
        if t.confirmed and t.moving:
            d = math.dist(t.center_at(0), xy)
            if best is None or d < best[0]:
                best = (d, t)
    return best


# 1) Takip: kamyonun hızı ve vincin dönüş hızı doğru tahmin ediliyor mu?
v_err, w_err = [], []
for seed in range(4):
    scene, vp = setup(seed)
    for f in range(300):
        truck = (scene.road_x, scene.truck_y)       # render'dan önceki konum = bu karedeki konum
        frame, gt = scene.render(); scene.step()
        vp.process(frame)
        if f < 100:
            continue
        if 40 < truck[1] < V.FRAME_H - 40:          # kamyon tamamen kadrajda
            hit = nearest_track(vp, truck)
            if hit and hit[0] < 15:
                v_err.append(math.dist(hit[1].v, (0, scene.truck_speed)))
        mx, my = scene.crane_mast
        a = scene.crane_angle - scene.crane_speed     # bu karede çizilen açı
        arm = (mx + math.cos(a) * scene.crane_len / 2, my + math.sin(a) * scene.crane_len / 2)
        hit = nearest_track(vp, arm)
        if hit and hit[0] < 40 and abs(hit[1].w) > 0:
            w_err.append(abs(hit[1].w - scene.crane_speed))
print(f"takip: kamyon hiz hatasi medyan={np.median(v_err):.2f} px/kare ({len(v_err)} olcum), "
      f"vinc donus hatasi medyan={np.median(w_err):.4f} rad/kare ({len(w_err)} olcum)")
assert len(v_err) > 100 and np.median(v_err) < 0.5
assert len(w_err) > 100 and np.median(w_err) < 0.004


# 2) Asıl hata: drone kamyon yolunda duruyor (hedefi orası). Tahminsizken kamyon üstünden geçer,
#    tahminliyken drone kenara çekilip kamyon geçince yerine döner.
def park(seed, predict):
    scene, vp = setup(seed, predict)
    spot = (scene.road_x // V.CELL_PX, V.ROWS // 2)
    vp.set_drone(spot); vp.set_goal(spot)
    hits = back = 0
    for f in range(400):                              # kamyon en geç ~340 karede bir geçer
        frame, gt = scene.render(); scene.step()
        vp.process(frame)
        if f % V.MOVE_EVERY == 0:
            vp.step_drone()
        if V.ObstacleDetector.to_cells(gt)[vp.drone[1], vp.drone[0]]:
            hits += 1
        back += vp.drone == spot
    return hits, vp.stats["dodges"], back


for seed in range(3):
    off, _, _ = park(seed, False)
    on, dodges, back = park(seed, True)
    print(f"  yolda bekleme, sahne {seed}: tahminsiz ezilme={off} kare, tahminli ezilme={on} "
          f"kenara_cekilme={dodges} yerinde={back}/400 kare")
    assert off > 0, "tahminsiz de ezilme yok: senaryo sorunu yakalamıyor"
    assert on == 0 and dodges > 0 and back > 200


# 3) Rastgele görevler: tahmin açıkken drone hiçbir karede gerçek engelin içinde olmamalı,
#    D* Lite her karede cezalı A* ile aynı maliyeti bulmalı.
def run(seed):
    scene, vp = setup(seed)
    hits = mism = 0
    for f in range(FRAMES):
        frame, gt = scene.render(); scene.step()
        vp.process(frame)
        ap, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
        if (vp.path is None) != (ap is None) or \
                (ap and abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(ap, vp.penalty)) > 1e-9):
            mism += 1
        if vp.path:
            assert all(not vp.grid[y][x] for x, y in vp.path[1:]), "rota dolu hücreden geçiyor"
        if f % V.MOVE_EVERY == 0 and vp.step_drone():
            vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
        if V.ObstacleDetector.to_cells(gt)[vp.drone[1], vp.drone[0]]:
            hits += 1
    return hits, mism, vp.stats


tot = tot_mism = dodges = reached = 0
for seed in SEEDS:
    h, mism, st = run(seed)
    tot += h; tot_mism += mism; dodges += st["dodges"]; reached += st["reached"]
print(f"gorevler: {len(SEEDS)} sahne x {FRAMES} kare, ezilme={tot} D*/A* uyusmazlik={tot_mism} "
      f"kenara_cekilme={dodges} hedef={reached}")
assert tot == 0 and tot_mism == 0
assert reached >= 2 * len(SEEDS), "drone kaçarken işini yapamıyor"
print("TAMAM")
