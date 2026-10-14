"""Algılama + güvenlik payı + D* Lite entegrasyon testi. Çalıştır: python tests/test_vision.py"""
import os, sys, random
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import drone_vision as V
import drone_dstar_lite as M

# 1) Güvenlik payı: kaydırmalı şişirme, kaba kuvvet sürümüyle birebir aynı mı?
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:M.ROWS, 0:M.COLS]
for trial in range(200):
    occ = rng.random((M.ROWS, M.COLS)) < rng.uniform(0.01, 0.2)
    r = int(rng.integers(0, 4))
    brute = np.zeros_like(occ)
    for y, x in zip(*np.nonzero(occ)):
        brute |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    assert (V.inflate(occ, r) == brute).all(), (trial, r)
print("inflate: 200 rastgele grid kaba kuvvetle aynı")

# 2) + 3) Algılama doğruluğu ve canlı planlama
prec, rec = [], []
checks = mism = moved_into = 0
for seed in range(6):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector("renk")
    det.calibrate(scene.background())
    vp = V.VisionPlanner(det)
    tp = fp = fn = 0
    for f in range(160):
        frame, gt = scene.render(); scene.step()
        vp.process(frame)
        g = V.ObstacleDetector.to_cells(gt)
        tp += int((g & vp.raw).sum()); fp += int((~g & vp.raw).sum()); fn += int((g & ~vp.raw).sum())
        # D* Lite'ın artımlı rotası, aynı grid üzerinde sıfırdan A* ile aynı maliyette mi?
        ap, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
        checks += 1
        if (vp.path is None) != (ap is None) or (ap and abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(ap, vp.penalty)) > 1e-9):
            mism += 1
        if vp.path:
            assert all(not vp.grid[y][x] for x, y in vp.path[1:]), "rota dolu hücreden geçiyor"
        if f % V.MOVE_EVERY == 0:
            before = vp.drone
            if vp.step_drone():
                vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
            if vp.drone != before and g[vp.drone[1], vp.drone[0]]:
                moved_into += 1
    prec.append(tp / max(tp + fp, 1)); rec.append(tp / max(tp + fn, 1))
print(f"algılama: kesinlik min={min(prec):.2f} duyarlılık min={min(rec):.2f}")
print(f"planlama: kontrol={checks} D*/A* uyuşmazlık={mism} engele_girdi={moved_into}")
assert min(prec) > 0.9 and min(rec) > 0.9
assert mism == 0 and moved_into == 0
print("TAMAM")
