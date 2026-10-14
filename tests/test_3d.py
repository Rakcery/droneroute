"""İrtifalı (3D) planlama testi. Çalıştır: python tests/test_3d.py  (3-4 dakika)"""
import os, sys, random
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import drone_vision as V
import drone_dstar3d as D
import drone_plan3d as P
from drone_dstar_lite import COLS, ROWS

# 1) 3D D* Lite, rastgele yükseklik haritalarında, drone ilerleyip harita değişirken 3D A* ile aynı maliyeti buluyor mu?
random.seed(1)
checks = mism = 0
for trial in range(60):
    hmap = [[random.choice([0, 0, 0, 0, 7, 12, 17, 30]) for _ in range(COLS)] for _ in range(ROWS)]
    grid = [[[D.ALTS[z] < hmap[y][x] + D.CLEARANCE for x in range(COLS)] for y in range(ROWS)] for z in range(D.NZ)]
    pen = [[[random.choice([0, 0, 0, 1.5]) for _ in range(COLS)] for _ in range(ROWS)] for _ in range(D.NZ)]
    s, g = (1, ROWS // 2, 0), (COLS - 2, ROWS // 2, 0)
    for x, y, z in (s, g):
        grid[z][y][x] = False
    ds = D.DStarLite3D(grid, s, g, pen)
    for step in range(15):
        p = ds.path()
        a, _ = D.astar3(grid, ds.start, g, pen)
        checks += 1
        if (p is None) != (a is None) or (a and abs(D.path_cost3(grid, pen, p) - D.path_cost3(grid, pen, a)) > 1e-9):
            mism += 1
        if p and len(p) > 1:
            ds.move_to(p[1])
        changed = []
        for _ in range(20):
            x, y, z = c = (random.randrange(COLS), random.randrange(ROWS), random.randrange(D.NZ))
            if c in (ds.start, g):
                continue
            grid[z][y][x] = not grid[z][y][x]
            pen[z][y][x] = random.choice([0, 2.0])
            changed.append(c)
        ds.cells_changed(changed)
print(f"D* Lite 3D vs A* 3D: kontrol={checks} uyusmazlik={mism}")
assert mism == 0


def setup(seed, cls=P.VisionPlanner3D):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector("renk")
    det.calibrate(scene.background())
    return scene, cls(det)


# 2) Kamyon yolunda bekleme: 2D'de drone yana kaçmak zorunda. 3D'de kamyon (4 m) alçak: drone yükselip
#    yerinde kalmalı, kamyon geçince alçalmalı. Hiç ezilmemeli.
for seed in range(3):
    res = {}
    for name, cls in (("2D", V.VisionPlanner), ("3D", P.VisionPlanner3D)):
        scene, vp = setup(seed, cls)
        spot = (scene.road_x // V.CELL_PX, ROWS // 2)
        vp.set_drone(spot); vp.set_goal(spot)
        hits = home = 0
        for f in range(400):
            frame, gt = scene.render(); hm = scene.height_map; scene.step()
            vp.process(frame, hm)
            if f % V.MOVE_EVERY == 0:
                vp.step_drone()
            hits += V.drone_hit(vp, V.ObstacleDetector.to_cells(gt), hm)
            home += vp.drone == spot
        res[name] = (hits, home, vp.stats.get("climbs", 0))
    print(f"  yolda bekleme, sahne {seed}: 2D ezilme={res['2D'][0]} yerinde={res['2D'][1]}/400 | "
          f"3D ezilme={res['3D'][0]} yerinde={res['3D'][1]}/400 tirmanma={res['3D'][2]}")
    assert res["3D"][0] == 0 and res["3D"][2] > 0 and res["3D"][1] > res["2D"][1]


# 3) Yükseklik yoksa (webcam) her engel sonsuz yüksek: drone hiç tırmanmamalı, 2D gibi davranmalı.
scene, vp = setup(0)
for f in range(300):
    frame, gt = scene.render(); scene.step()
    vp.process(frame, None)
    if f % V.MOVE_EVERY == 0 and vp.step_drone():
        vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
    assert not V.drone_hit(vp, V.ObstacleDetector.to_cells(gt))
print(f"yukseklik yok: tirmanma={vp.stats['climbs']} hedef={vp.stats['reached']}")
assert vp.stats["climbs"] == 0


# 4) Rastgele görevler: drone hiçbir karede (irtifası hesaba katılarak) gerçek engelin içinde olmamalı,
#    D* Lite 3D her karede 3D A* ile aynı maliyeti bulmalı.
tot = tot_mism = reached = climbs = 0
for seed in range(6):
    scene, vp = setup(seed)
    for f in range(600):
        frame, gt = scene.render(); hm = scene.height_map; scene.step()
        vp.process(frame, hm)
        a, _ = D.astar3(vp.grid, (*vp.drone, vp.z), (*vp.goal, vp.gz), vp.penalty)
        if (vp.path is None) != (a is None) or \
                (a and abs(vp.path_cost() - D.path_cost3(vp.grid, vp.penalty, a)) > 1e-9):
            tot_mism += 1
        if f % V.MOVE_EVERY == 0 and vp.step_drone():
            vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
        tot += V.drone_hit(vp, V.ObstacleDetector.to_cells(gt), hm)
    reached += vp.stats["reached"]; climbs += vp.stats["climbs"]
print(f"gorevler: 6 sahne x 600 kare, ezilme={tot} D*/A* uyusmazlik={tot_mism} hedef={reached} tirmanma={climbs}")
assert tot == 0 and tot_mism == 0 and reached >= 12
print("TAMAM")
