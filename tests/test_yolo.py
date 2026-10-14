"""
YOLO modu + eğik kamera testi. Çalıştır: python tests/test_yolo.py  (1-2 dakika, GPU/kamera/internet gerekmez)

1) Homografi (DLT, kendi kodumuz): rastgele homografileri noktalardan geri buluyor mu? Kameradan hesaplanan
   zemin homografisi pinhole modeliyle birebir aynı mı?
2) Eğik kamerada tespit → zemin: sahnenin "kusursuz YOLO"su (SimOracle) ile her nesnenin önü doğru yerde mi,
   boyu asla az tahmin edilmiyor mu (güvenli taraf), kuşun konumu doğru mu?
3) Kuşbakışı YOLO modu (SimOracle): algılama doğruluğu, sınıftan gelen boylar (kamyon 4.5 m, insan/kuş/vinç
   üstünden uçulmaz), hibrit mod, D* Lite = A*, drone engele girmiyor.
4) Eğik kamerayla tam boru hattı (2D ve 3D): görüş alanında algılama + D* Lite = A* + drone hiçbir karede
   gerçek engelin içinde değil.
5) Gerçek YOLO modeli (sadece ultralytics kuruluysa ve models/droneroute_yoloe.pt hazırsa; yoksa atlanır).
"""
import os, sys, math, random
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
import cv2
import numpy as np
import drone_vision as V
import drone_yolo as Y
import drone_dstar_lite as M

assert (Y.CELL_PX, Y.FRAME_W, Y.FRAME_H) == (V.CELL_PX, V.FRAME_W, V.FRAME_H)

# 1) Homografi
rng = np.random.default_rng(0)
for _ in range(200):
    H = np.eye(3) + rng.normal(0, 0.3, (3, 3)); H[2, :2] = rng.normal(0, 0.002, 2); H /= H[2, 2]
    src = rng.uniform(0, 800, (int(rng.integers(4, 12)), 2))
    H2 = Y.homography_dlt(src, Y.apply_h(H, src))
    assert np.allclose(H2, H, atol=1e-6), (H, H2)
cam = Y.Camera(25, 35, 82)
proj = Y.GroundProjector.from_camera(cam, 2.0)
g = np.c_[rng.uniform(-40, 40, 300), rng.uniform(5, 80, 300)]
px, z = cam.project(np.c_[g, np.zeros(len(g))])
assert (z > 0).all() and np.allclose(proj.to_ground(px), g, atol=1e-6)
r = cam.ray(*px[0]); s = -cam.alt / r[2]
assert np.allclose(cam.C[:2] + s * r[:2], g[0])                 # ışın zemine aynı noktada değiyor
assert abs(cam.project([[0, 1e6, 0]])[0][0, 1] - cam.horizon_v()) < 1   # sonsuzdaki zemin = ufuk
print("homografi: DLT 200 rastgele homografiyi buldu, kamera homografisi pinhole ile aynı")

# 2) Eğik kamera: konum + boy
front_err, h_ratio, fly_err, n_cut = [], [], [], 0
for seed in range(10):
    sc = V.SimScene(seed)
    for f in range(5):
        for _ in range(30):
            sc.render(); sc.step()
        sc.render()
        for o in sc.objs:
            box = cam.object_box(o)
            if box is None or o["cls"] == "vinc":
                continue
            fp = proj.footprint(Y.Detection(o["cls"], 1.0, box))
            base = Y.bev_to_ground(o["parts"][0][0], 2.0)
            true_h = max(z1 for _, _, z1, _ in o["parts"])
            x1, y1, x2, y2 = box
            if o["cls"] == "kus":
                if fp and x1 > 0 and x2 < Y.FRAME_W - 1:
                    c = base.mean(axis=0)
                    fly_err.append(np.hypot(*(fp[0].mean(axis=0) - c)) / np.hypot(*c))
                continue
            assert fp is not None
            poly, h = fp
            if y2 >= Y.FRAME_H - 2 or y1 <= 1 or x1 <= 0 or x2 >= Y.FRAME_W - 1:   # kadrajdan kesik: tam ölçülemez
                n_cut += 1
                hull = cv2.convexHull(base.astype(np.float32))
                assert any(cv2.pointPolygonTest(hull, tuple(map(float, p)), False) >= 0 for p in
                           np.linspace(poly[0], poly[3], 20)) or h >= true_h  # en azından bir kısmı üstünde
                assert h >= true_h or (x1 <= 0 or x2 >= Y.FRAME_W - 1)   # yandan kesikte tepe görünüyor, boy ölçülür
                continue
            # Güvenlik için önemli olan: engelin kameraya en yakın noktası tahmini tabanın içinde mi?
            near = base[np.argmin(np.hypot(base[:, 0], base[:, 1]))]
            front_err.append(-cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, near)), True))
            h_ratio.append(h / true_h)
front_err, h_ratio = np.array(front_err), np.array(h_ratio)
print(f"egik kamera: {len(front_err)} nesne, en yakin nokta tabanin disinda max={front_err.max():.2f} m, "
      f"boy tahmini / gercek min={h_ratio.min():.2f} medyan={np.median(h_ratio):.2f}, kadrajdan kesik={n_cut}, "
      f"kus konum hatasi max=%{100 * max(fly_err):.1f}")
assert front_err.max() < 0.5                 # en yakın noktası tahmini tabanın içinde (yarım metre pay)
assert h_ratio.min() >= 1.0                  # boy asla az tahmin edilmiyor
assert np.median(h_ratio) < 2.0              # ama saçma da değil
assert len(fly_err) > 5 and max(fly_err) < 0.05


# 3) Kuşbakışı YOLO (SimOracle)
def topdown(seed, method, frames=150):
    random.seed(seed)
    scene = V.SimScene(seed)
    det = V.ObstacleDetector(method, Y.YoloDetector(Y.SimOracle(scene)))
    det.calibrate(scene.background())
    vp = V.VisionPlanner(det)
    tp = fp = fn = mism = moved_into = 0
    heights = {}
    for f in range(frames):
        frame, gt = scene.render(); scene.step()
        vp.process(frame)                                  # derinlik yok: boylar sınıftan
        g = V.ObstacleDetector.to_cells(gt)
        tp += int((g & vp.raw).sum()); fp += int((~g & vp.raw).sum()); fn += int((g & ~vp.raw).sum())
        ap, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
        if (vp.path is None) != (ap is None) or (ap and abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(ap, vp.penalty)) > 1e-9):
            mism += 1
        if f % V.MOVE_EVERY == 0:
            before = vp.drone
            if vp.step_drone():
                vp.set_goal(V.random_free_cell(vp, random, far_from=vp.drone))
            if vp.drone != before and g[vp.drone[1], vp.drone[0]]:
                moved_into += 1
        for cls, h, poly in det.yolo.items:              # sınıftan gelen boylar
            heights.setdefault(cls, set()).add(h)
        tx, ty = scene.road_x // V.CELL_PX, int(scene.truck_y - scene.truck_speed) // V.CELL_PX
        if 0 <= ty < V.ROWS and vp.raw[ty, tx]:          # kamyonun hücresi: 4.5 m (üstünden vinç kolu geçiyorsa sonsuz)
            assert vp.perception.height[ty, tx] in (4.5, math.inf), vp.perception.height[ty, tx]
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), mism, moved_into, heights


for method in ("yolo", "hibrit"):
    res = [topdown(s, method) for s in range(3)]
    p, r = min(x[0] for x in res), min(x[1] for x in res)
    print(f"kusbakisi {method}: kesinlik min={p:.2f} duyarlilik min={r:.2f} "
          f"D*/A* uyusmazlik={sum(x[2] for x in res)} engele_girdi={sum(x[3] for x in res)}")
    assert p > 0.9 and r > 0.9 and sum(x[2] for x in res) == 0 and sum(x[3] for x in res) == 0
h = res[0][4]
assert h == {"arac": {4.5}, "vinc": {math.inf}, "kus": {math.inf}, "agac": {15.0}, "bina": {30.0}}, h
print("kusbakisi boylar: kamyon 4.5 m, agac 15 m, bina 30 m, vinc/kus sonsuz (ustunden uculmaz)")

# hibrit: YOLO'nun kaçırdığı engeli renk yakalar, boyu bilinmiyor (sonsuz)
scene = V.SimScene(7)
oracle = Y.SimOracle(scene)
blind = lambda frame: [d for d in oracle(frame) if d.cls != "bina"]       # YOLO binaları göremiyor
det = V.ObstacleDetector("hibrit", Y.YoloDetector(blind)); det.calibrate(scene.background())
perc = V.Perception(det)
for _ in range(5):
    frame, gt = scene.render(); scene.step(); raw, _ = perc.process(frame)
x, y, w, hh, _ = scene.buildings[0]
cy, cx = (y + hh // 2) // V.CELL_PX, (x + w // 2) // V.CELL_PX
assert raw[cy, cx] and perc.height[cy, cx] == math.inf
print("hibrit: YOLO'nun kacirdigi bina renkle yakalandi, boyu bilinmiyor = sonsuz")


# 4) Eğik kamerayla tam boru hattı
def collides(scene, cell, alt=None):
    """Drone'un hücresi bir nesne parçasıyla çakışıyor mu? 2D'de (alt=None) 20 m altına inen her parça,
    3D'de sadece drone'un irtifasını kapsayan parça (vinç kolunun altından, kamyonun üstünden geçilir)."""
    m = np.zeros((V.FRAME_H, V.FRAME_W), np.uint8)
    x, y = cell
    for o in scene.objs:
        for p, z0, z1, _ in o["parts"]:
            if (alt is None and z0 < 22) or (alt is not None and z0 - 2 <= alt <= z1 + 2):
                cv2.fillPoly(m, [p], 255)
    return V.ObstacleDetector.to_cells(m)[y, x]


def oblique(seed, use3d, frames=300):
    args = argparse.Namespace(kamera=False, video=None, kamera_id=0, seed=seed, perspektif=True, irtifa=25.0,
                              egim=35.0, fov=82.0, vfov=None, hucre_m=2.0, kalibrasyon=None, yontem="yolo",
                              model="sim", guven=0.25, uc_boyut=use3d, tahminsiz=False)
    random.seed(seed)
    src, det, vp = V.build(args)
    rng = random.Random(seed)
    from drone_dstar3d import ALTS, CLEARANCE, astar3, path_cost3
    hits = mism = tp = fn = 0
    for f in range(frames):
        frame, _ = src.read()
        vp.process(frame)
        gt = V.ObstacleDetector.to_cells(src.sim.footprint_below(ALTS[-1] + CLEARANCE)) & ~vp.unseen
        tp += int((gt & vp.raw).sum()); fn += int((gt & ~vp.raw).sum())
        if use3d:
            ap, _ = astar3(vp.grid, (*vp.drone, vp.z), (*vp.goal, vp.gz), vp.penalty)
            bad = (vp.path is None) != (ap is None) or (ap and abs(path_cost3(vp.grid, vp.penalty, vp.path) -
                                                                 path_cost3(vp.grid, vp.penalty, ap)) > 1e-9)
        else:
            ap, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
            bad = (vp.path is None) != (ap is None) or (ap and abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(ap, vp.penalty)) > 1e-9)
        mism += bool(bad)
        if f % V.MOVE_EVERY == 0 and vp.step_drone():
            nxt = V.random_free_cell(vp, rng, far_from=vp.drone)
            if nxt:
                vp.set_goal(nxt)
        hits += collides(src.sim, vp.drone, ALTS[vp.z] if use3d else None)
    return tp / max(tp + fn, 1), mism, hits, vp.stats


for use3d in (False, True):
    res = [oblique(s, use3d) for s in (3, 4)]
    print(f"egik kamera {'3D' if use3d else '2D'}: gorus alaninda duyarlilik min={min(x[0] for x in res):.2f} "
          f"D*/A* uyusmazlik={sum(x[1] for x in res)} carpma={sum(x[2] for x in res)} "
          f"hedef={sum(x[3]['reached'] for x in res)}")
    assert min(x[0] for x in res) > 0.8 and sum(x[1] for x in res) == 0 and sum(x[2] for x in res) == 0
    assert sum(x[3]["reached"] for x in res) > 0

# 5) Gerçek model (isteğe bağlı)
try:
    import ultralytics
    from ultralytics.utils import ASSETS
    have = os.path.exists(Y.DEFAULT_MODEL)
except ImportError:
    have = False
if not have:
    print("gercek YOLO: atlandi (pip install ultralytics, sonra: python drone_yolo.py --model-hazirla)")
else:
    yd = Y.YoloDetector(Y.UltralyticsBackend())
    img = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (V.FRAME_W, V.FRAME_H))
    mask, height = yd.process(img)
    classes = [d.cls for d in yd.dets]
    print(f"gercek YOLO (bus.jpg): {sorted(classes)}")
    assert classes.count("insan") >= 2 and "arac" in classes and mask.any()
    assert all(not math.isfinite(h) for c, h, _ in yd.items if c == "insan")
print("TAMAM")
