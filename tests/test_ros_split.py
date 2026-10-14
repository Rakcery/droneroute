"""
ROS 2 node'larına bölmenin testi, ROS gerekmeden. Çalıştır: python tests/test_ros_split.py  (2-3 dakika)

1) codec: grid/tti/costmap/koordinat dönüşümleri gidip geliyor mu?
2) + 3) Node'ların yaptığını aynen taklit et (2D ve 3D): algılama (Perception) → mesaja kodla → çöz → planlama (GridPlanner).
   Drone hiçbir karede gerçek engelin içinde olmamalı, D* Lite her karede cezalı A* ile aynı maliyeti bulmalı.
"""
import os, sys, random
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import drone_vision as V
import drone_dstar_lite as M
from droneroute_ros import codec

# 1) codec
rng = np.random.default_rng(0)
for _ in range(100):
    a = rng.random((M.ROWS, M.COLS)) < 0.3
    assert (codec.data_to_flags(codec.flags_to_data(a)) == a).all()
    t = np.where(rng.random((M.ROWS, M.COLS)) < 0.5, np.inf, rng.uniform(0, 20, (M.ROWS, M.COLS)))
    back = codec.data_to_tti(codec.tti_to_data(t))
    assert (np.isinf(back) == np.isinf(t)).all()
    fin = np.isfinite(t)
    assert (back[fin] <= t[fin]).all() and (t[fin] - back[fin] < 1 / codec.TTI_SCALE + 1e-9).all()  # erken, asla geç değil
    occ = rng.random((M.ROWS, M.COLS)) < 0.2
    pen = np.where(occ, 0, np.where(rng.random((M.ROWS, M.COLS)) < 0.3, rng.uniform(0.1, 4), 0))
    o2, p2 = codec.data_to_cost(codec.cost_to_data(occ, pen))
    assert (o2 == occ).all() and ((p2 > 0) == (pen > 0)).all()
    h = np.where(rng.random((M.ROWS, M.COLS)) < 0.1, np.inf, rng.uniform(0, 50, (M.ROWS, M.COLS)))
    h2 = codec.data_to_height(codec.height_to_data(h))
    assert (np.isinf(h2) == np.isinf(h)).all() and (h2[np.isfinite(h)] >= h[np.isfinite(h)]).all()  # asla kısa değil
    assert (h2[np.isfinite(h)] - h[np.isfinite(h)] < 1).all()
for x in range(M.COLS):
    for y in range(M.ROWS):
        assert codec.xy_to_cell(*codec.cell_to_xy((x, y))) == (x, y)
assert codec.cell_to_xy((0, M.ROWS - 1)) == (0.25, 0.25)      # görüntünün sol altı = haritanın orijini
data = codec.flags_to_data(np.eye(M.ROWS, M.COLS, dtype=bool)[:, ::-1])
assert data[M.COLS - M.ROWS] == 100                            # alt satır (y=ROWS-1) verinin başında
img = rng.uniform(0, 40, (16, 24)).astype(np.float32)
assert (codec.image_to_array(16, 24, "32FC1", img.tobytes()) == img).all()
print("codec: grid, tti, costmap, boy, koordinat, derinlik goruntusu gidip geliyor")


# 2) node zinciri
def run(seed, frames=600):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector("renk")
    det.calibrate(scene.background())
    perc, gp = V.Perception(det), V.GridPlanner()
    hits = mism = 0
    for f in range(frames):
        frame, gt = scene.render(); scene.step()
        raw, moving = perc.process(frame)                                         # algılama node'u
        grid_data, tti_data = codec.flags_to_data(raw), codec.tti_to_data(moving)  # → mesaj
        gp.update(codec.data_to_flags(grid_data), codec.data_to_tti(tti_data))   # planlama node'u
        ap, _ = M.astar(gp.grid, gp.drone, gp.goal, gp.penalty)
        if (gp.path is None) != (ap is None) or \
                (ap and abs(M.path_cost(gp.path, gp.penalty) - M.path_cost(ap, gp.penalty)) > 1e-9):
            mism += 1
        if (f + 1) % V.MOVE_EVERY == 0 and gp.step_drone():
            gp.set_goal(V.random_free_cell(gp, random, far_from=gp.drone))
        if V.ObstacleDetector.to_cells(gt)[gp.drone[1], gp.drone[0]]:
            hits += 1
    return hits, mism, gp.stats


tot = tot_mism = reached = dodges = 0
for seed in range(6):
    h, m, st = run(seed)
    tot += h; tot_mism += m; reached += st["reached"]; dodges += st["dodges"]
print(f"node zinciri: 6 sahne x 600 kare, ezilme={tot} D*/A* uyusmazlik={tot_mism} "
      f"kenara_cekilme={dodges} hedef={reached}")
assert tot == 0 and tot_mism == 0 and reached >= 12


# 3) 3D zinciri: algılama boyları da yayınlar, planlayıcı GridPlanner3D
import drone_dstar3d as D3
import drone_plan3d as P3


def run3d(seed, frames=400):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector("renk")
    det.calibrate(scene.background())
    perc, gp = V.Perception(det), P3.GridPlanner3D()
    hits = mism = 0
    for f in range(frames):
        frame, gt = scene.render(); hm = scene.height_map; scene.step()
        raw, moving = perc.process(frame, hm)
        msgs = (codec.flags_to_data(raw), codec.tti_to_data(moving),
                codec.height_to_data(perc.height), codec.height_to_data(perc.moving_height))
        gp.update(codec.data_to_flags(msgs[0]), codec.data_to_height(msgs[2]),
                  codec.data_to_tti(msgs[1]), codec.data_to_height(msgs[3]))
        a, _ = D3.astar3(gp.grid, (*gp.drone, gp.z), (*gp.goal, gp.gz), gp.penalty)
        if (gp.path is None) != (a is None) or (a and abs(gp.path_cost() - D3.path_cost3(gp.grid, gp.penalty, a)) > 1e-9):
            mism += 1
        if (f + 1) % V.MOVE_EVERY == 0 and gp.step_drone():
            gp.set_goal(V.random_free_cell(gp, random, far_from=gp.drone))
        hits += V.drone_hit(gp, V.ObstacleDetector.to_cells(gt), hm)
    return hits, mism, gp.stats


tot = tot_mism = reached = 0
for seed in range(3):
    h, m, st = run3d(seed)
    tot += h; tot_mism += m; reached += st["reached"]
print(f"3D node zinciri: 3 sahne x 400 kare, ezilme={tot} D*/A* uyusmazlik={tot_mism} hedef={reached}")
assert tot == 0 and tot_mism == 0 and reached >= 5
print("TAMAM")
