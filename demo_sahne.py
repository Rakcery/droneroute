"""
DroneRoute — 4 adım tek ekranda (sunum / demo videosu için)

Ekran dörde bölünüyor, her çeyrekte projenin bir adımı kendi kendine uçuyor ve kendi testini canlı yapıyor:
  1) D* Lite           : rotanın önüne rastgele engel düşüyor, D* Lite sadece değişen kısmı onarıyor.
                          Test: her yeniden planda sıfırdan A* ile aynı maliyet mi?
  2) Kamera + takip    : kuşbakışı kameradan engel algılama, hareketli engelleri takip edip tahmin etme.
                          Test: algılama doğruluğu, drone hiç çarptı mı, D* = A* mı?
  3) İrtifa (3D)       : 4 irtifa katmanında D* Lite; alçak engelin üstünden geçiyor, yüksekten dolaşıyor.
                          Test: irtifa hesaba katılarak çarpma, 3D D* = 3D A* mı?
  4) YOLO + eğik kamera: eğik kameradaki tespitler geometriyle yere indiriliyor, boyları tahmin ediliyor.
                          Test: drone'un uçtuğu irtifada gerçek bir engele girdi mi?
Her çeyreğin altında o an ne olduğunu anlatan Türkçe yazı var. Simülasyonda "kusursuz YOLO" kullanılıyor
(sahnenin bildiği nesneler), model indirmeye gerek yok.

  python demo_sahne.py                        # izle (F: tam ekran, SPACE: dur, Q: çık)
  python demo_sahne.py --kaydet demo4.mp4     # pencere açmadan 1920x1080 videoya yaz (ekran kaydı gerekmez)
  python demo_sahne.py --sure 120             # süre (saniye)
"""
import argparse
import os
import random
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np

import drone_dstar_lite as M
import drone_dstar3d as D
import drone_vision as V
from drone_dstar3d import ALTS

FPS = 20
W, H = 1920, 1080
TOP = 44                               # başlık şeridi
QW, QH = W // 2, (H - TOP) // 2        # çeyrek: 960 x 518
HEAD, FOOT = 34, 52                    # çeyreğin başlığı ve alt yazısı
BODY = QH - HEAD - FOOT                # 432
MAP_W = int(V.FRAME_W * BODY / V.FRAME_H)   # harita bu boyda: 617 x 432
SIDE_W = QW - MAP_W
HOLD = 60                              # olay yazısı kaç kare kalsın
GREEN, RED, GRAY, WHITE, GOLD = (90, 220, 110), (70, 70, 240), (170, 170, 170), (240, 240, 240), (120, 220, 255)

# ---------------- Türkçe yazı (OpenCV'nin fontu ş/ğ/ı basamıyor; pygame'inki basıyor) ----------------
_fonts = {}


def put(img, text, x, y, size=24, color=WHITE, maxw=None):
    """img'ye (BGR) yazı basar; maxw verilirse kelimeden satıra böler. Son satırın altını döner."""
    import pygame
    if not pygame.font.get_init():
        pygame.font.init()
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    font = _fonts[size]
    lines = [text]
    if maxw:
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if font.size(trial)[0] > maxw and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    for ln in lines:
        if not ln:
            continue
        s = font.render(ln, True, (255, 255, 255))
        a = pygame.surfarray.pixels_alpha(s).T.astype(np.float32)[:, :, None] / 255.0
        h, w = a.shape[:2]
        h, w = min(h, img.shape[0] - y), min(w, img.shape[1] - x)
        if h <= 0 or w <= 0:
            break
        roi = img[y:y + h, x:x + w]
        roi[:] = (roi * (1 - a[:h, :w]) + np.array(color, np.float32) * a[:h, :w]).astype(np.uint8)
        y += h + 3
    return y


class Ticker:
    """Çeyreğin alt yazısı: varsayılan açıklama; bir olay olunca HOLD kare boyunca olayı anlatır."""

    def __init__(self, default):
        self.default, self.text, self.left = default, default, 0

    def event(self, text, urgent=False):
        if self.left > HOLD - 25 and not urgent:     # yeni çıkmış yazıyı hemen ezme
            return
        self.text, self.left = text, HOLD

    def tick(self):
        self.left -= 1
        if self.left <= 0:
            self.text = self.default
        return self.text


def check_line(ok, text):
    return ("[OK] " if ok else "[!!] ") + text, GREEN if ok else RED


# ---------------- 1) D* Lite ----------------
class DStarPanel:
    title = "1) D* Lite: harita değişince anlık yeniden planlama"
    SPAWN = 30

    def __init__(self, seed):
        self.rng = random.Random(seed)
        random.seed(seed)
        self.grid, self.drone, self.goal = M.make_map()
        self.dstar = M.DStarLite(self.grid, self.drone, self.goal)
        self.path = self.dstar.path()
        self.fresh = {}
        self.f = 0
        self.replans = self.sum_d = self.sum_a = self.checks = self.mism = self.reached = 0
        self.last = (0, 0)
        self.ticker = Ticker("Gri: duvar, kırmızı: yeni düşen engel, mavi gölge: D* Lite'ın son planda dokunduğu "
                             "düğümler. Sıfırdan aramak yerine sadece değişen kısmı onarıyor.")

    def _verify(self):
        a, _ = M.astar(self.grid, self.drone, self.goal)
        self.checks += 1
        if (self.path is None) != (a is None) or (a and abs(M.path_cost(self.path) - M.path_cost(a)) > 1e-9):
            self.mism += 1

    def _new_goal(self):
        for _ in range(500):
            c = (self.rng.randrange(1, M.COLS - 1), self.rng.randrange(1, M.ROWS - 1))
            if not self.grid[c[1]][c[0]] and abs(c[0] - self.drone[0]) + abs(c[1] - self.drone[1]) > 25:
                self.goal = c
                self.dstar = M.DStarLite(self.grid, self.drone, self.goal)
                self.path = self.dstar.path()
                if self.path:
                    return

    def _changed(self, cells):
        d = self.dstar.cells_changed(cells)
        self.path = self.dstar.path()
        self._verify()
        return d

    def step(self):
        self.f += 1
        if self.f % self.SPAWN == 0:
            if self.path and len(self.path) > 5 and self.rng.random() < 0.7:
                tx, ty = self.rng.choice(self.path[3:])
            else:
                tx, ty = self.rng.randrange(M.COLS), self.rng.randrange(M.ROWS)
            changed = []
            for dx in (0, 1):
                for dy in (0, 1):
                    nx, ny = tx + dx, ty + dy
                    if (0 <= nx < M.COLS and 0 <= ny < M.ROWS and not M.near((nx, ny), self.drone)
                            and not M.near((nx, ny), self.goal) and not self.grid[ny][nx]):
                        self.grid[ny][nx] = True
                        self.fresh[(nx, ny)] = self.f
                        changed.append((nx, ny))
            if changed:
                on_path = bool(self.path) and any(c in self.path for c in changed)
                d = self._changed(changed)
                if d:
                    a = len(M.astar(self.grid, self.drone, self.goal)[1])
                    self.replans += 1
                    self.sum_d += d
                    self.sum_a += a
                    self.last = (d, a)
                    if on_path:
                        self.ticker.event(f"Engel rotanın üstüne düştü. D* Lite rotayı {d} düğümle onardı; "
                                          f"A* sıfırdan arasaydı {a} düğüm açacaktı.")
            # yeni engeller birikip haritayı tıkamasın: eskiyenleri kaldır
            old = [c for c, t in self.fresh.items() if self.f - t > 400]
            for c in old:
                self.grid[c[1]][c[0]] = False
                del self.fresh[c]
            if old:
                self._changed(old)
        if self.f % 4 == 0:
            if self.path and len(self.path) > 1:
                self.drone = self.path[1]
                self.path = self.path[1:]
                self.dstar.move_to(self.drone)
                if self.drone == self.goal:
                    self.reached += 1
                    self.ticker.event("Hedefe varıldı, yeni hedef verildi. D* Lite yeni rotayı kurdu.")
                    self._new_goal()
            elif self.path is None:
                self._new_goal()

    def render(self):
        C = M.CELL
        img = np.full((M.ROWS * C, M.COLS * C, 3), (28, 20, 18), np.uint8)
        for (x, y) in self.dstar.last_expanded:
            cv2.rectangle(img, (x * C, y * C), ((x + 1) * C, (y + 1) * C), (85, 58, 35), -1)
        for y in range(M.ROWS):
            for x in range(M.COLS):
                if self.grid[y][x]:
                    col = (70, 90, 230) if self.f - self.fresh.get((x, y), -999) < 60 else (115, 95, 90)
                    cv2.rectangle(img, (x * C, y * C), ((x + 1) * C, (y + 1) * C), col, -1)
        for x in range(0, img.shape[1], C):
            cv2.line(img, (x, 0), (x, img.shape[0]), (44, 33, 30), 1)
        for y in range(0, img.shape[0], C):
            cv2.line(img, (0, y), (img.shape[1], y), (44, 33, 30), 1)
        mid = lambda c: (c[0] * C + C // 2, c[1] * C + C // 2)
        if self.path and len(self.path) > 1:
            cv2.polylines(img, [np.array([mid(c) for c in self.path], np.int32)], False, (255, 200, 80), 3, cv2.LINE_AA)
        gx, gy = self.goal
        cv2.rectangle(img, (gx * C + 3, gy * C + 3), ((gx + 1) * C - 4, (gy + 1) * C - 4), (120, 230, 90), -1)
        cv2.circle(img, mid(self.drone), C // 2 - 2, (60, 210, 255), -1, cv2.LINE_AA)
        return img

    def side(self, img):
        y = put(img, "Canlı ölçüm", 12, 10, 26, GOLD)
        d, a = self.last
        y = put(img, f"yeniden plan: {self.replans}", 12, y + 4)
        y = put(img, f"son plan: D* {d} / A* {a} düğüm", 12, y)
        if self.sum_a:
            y = put(img, f"toplamda D*, A*'ın %{100 * self.sum_d / self.sum_a:.0f}'i", 12, y)
            y = put(img, "kadar düğüm açtı", 12, y)
        y = put(img, f"hedefe varış: {self.reached}", 12, y)
        y = put(img, "Test", 12, y + 14, 26, GOLD)
        t, c = check_line(self.mism == 0, f"D* = A* maliyet ({self.checks - self.mism}/{self.checks})")
        y = put(img, t, 12, y + 4, 24, c)
        put(img, "Algoritma sıfırdan yazıldı, kütüphane yok.", 12, y + 14, 22, GRAY, SIDE_W - 20)


# ---------------- 2-4) Kamera boru hattı ----------------
def make(seed, uc_boyut, perspektif):
    ns = argparse.Namespace(kamera=False, video=None, kamera_id=0, seed=seed, tahminsiz=False, uc_boyut=uc_boyut,
                            perspektif=perspektif, irtifa=25.0, egim=35.0, fov=82.0, vfov=None, hucre_m=2.0,
                            kalibrasyon=None, yontem="yolo" if perspektif else "renk", model="sim", guven=0.25)
    random.seed(seed)
    return V.build(ns)


NAMES = {"arac": "araç", "vinc": "vinç kolu", "kus": "kuş", "insan": "insan", "bina": "bina", "agac": "ağaç"}


class VisionPanel:
    def __init__(self, seed, uc_boyut, perspektif, title, default):
        self.title = title
        self.src, self.det, self.vp = make(seed, uc_boyut, perspektif)
        self.rng = random.Random(seed)
        self.three_d, self.persp = uc_boyut, perspektif
        self.f = 0
        self.hits = self.checks = self.mism = 0
        self.tp = self.fp = self.fn = 0
        self.seen = set()
        self.prev = None
        self.frame = None
        self.ticker = Ticker(default)

    def _verify(self):
        """D* Lite'ın artımlı rotası, aynı harita üzerinde sıfırdan A* ile aynı maliyette mi?"""
        vp = self.vp
        self.checks += 1
        if self.three_d:
            a, _ = D.astar3(vp.grid, (*vp.drone, vp.z), (*vp.goal, vp.gz), vp.penalty)
            bad = (vp.path is None) != (a is None) or \
                (a and abs(vp.path_cost() - D.path_cost3(vp.grid, vp.penalty, a)) > 1e-9)
        else:
            a, _ = M.astar(vp.grid, vp.drone, vp.goal, vp.penalty)
            bad = (vp.path is None) != (a is None) or \
                (a and abs(M.path_cost(vp.path, vp.penalty) - M.path_cost(a, vp.penalty)) > 1e-9)
        self.mism += bool(bad)

    def _cls(self, t):
        y = getattr(self.det, "yolo", None)
        if y is None or not y.items:
            return "engel"
        c = t.center_at(0)
        best = min(y.items, key=lambda it: np.hypot(*(it[2].mean(axis=0) - c)))
        return NAMES.get(best[0], best[0])

    def step(self):
        vp, src = self.vp, self.src
        self.f += 1
        frame, gt = src.read()
        self.frame = frame
        hm = src.height()
        vp.process(frame, hm)
        moved = V.keep_goal_reachable(vp)
        if self.f % 3 == 0:
            self._verify()
        if not self.persp and gt is not None:
            g = V.ObstacleDetector.to_cells(gt)
            self.tp += int((g & vp.raw).sum()); self.fp += int((~g & vp.raw).sum()); self.fn += int((g & ~vp.raw).sum())
        if self.f % V.MOVE_EVERY == 0 and vp.step_drone():
            nxt = V.random_free_cell(vp, self.rng, far_from=vp.drone)
            if nxt:
                vp.set_goal(nxt)
        # çarpma testi: drone'un uçtuğu irtifada gerçekte bir engel var mı?
        if self.persp:
            hit = V.ObstacleDetector.to_cells(src.sim.footprint_at(ALTS[vp.z]))[vp.drone[1], vp.drone[0]]
        else:
            hit = V.drone_hit(vp, V.ObstacleDetector.to_cells(gt), hm)
        self.hits += bool(hit)
        self._events(moved)

    def _events(self, moved):
        vp, st = self.vp, self.vp.stats
        z = getattr(vp, "z", None)
        if self.prev is not None:
            pz, pst = self.prev
            if z is not None and z > pz:
                self.ticker.event(f"Tırmanıyor {ALTS[pz]} m -> {ALTS[z]} m: önündeki engel alçak, etrafından "
                                  "dolaşmak yerine üstünden geçmek daha kısa.", urgent=True)
            elif z is not None and z < pz:
                self.ticker.event(f"Alçalıyor {ALTS[pz]} m -> {ALTS[z]} m: engel geride kaldı, alçakta uçmak "
                                  "daha ucuz.")
            if st["dodges"] > pst["dodges"]:
                self.ticker.event("Hareketli engel üstüne geliyor ve rota kapalı: drone beklemiyor, kenara "
                                  "çekiliyor.", urgent=True)
            if st["reached"] > pst["reached"]:
                self.ticker.event("Hedefe varıldı. Yeni hedef verildi, D* Lite rotayı hemen kurdu.")
        if moved:
            self.ticker.event("Hedef bir engelin içinde kaldı: drone beklemek yerine en yakın boş noktaya gidiyor.",
                              urgent=True)
        for t in vp.tracker.tracks:
            if t.confirmed and t.moving and t.id not in self.seen:
                self.seen.add(t.id)
                what = self._cls(t) if self.persp else "engel"
                self.ticker.event(f"Hareketli {what} #{t.id} takibe alındı. Mor: birkaç saniye içinde gireceği "
                                  "yerler, rota oradan kaçınıyor.")
        self.prev = (z, dict(st))

    def render(self):
        return V.render(self.frame, self.vp, self.det, self.src, False)[:, :V.FRAME_W]

    def side(self, img):
        vp, st = self.vp, self.vp.stats
        y = 8
        if self.persp:                                   # kameranın gördüğü + YOLO kutuları
            import drone_yolo as Y
            cam = Y.draw_detections(self.frame, self.det.yolo.dets)
            cw = SIDE_W - 16
            ch = int(V.FRAME_H * cw / V.FRAME_W)
            img[y:y + ch, 8:8 + cw] = cv2.resize(cam, (cw, ch), interpolation=cv2.INTER_AREA)
            put(img, "kamera + YOLO", 14, y + 4, 22, (0, 255, 255))
            y = put(img, f"tespit: {len(self.det.yolo.dets)}   irtifa: {ALTS[vp.z]} m", 12, y + ch + 8)
            y = put(img, f"hedefe varış: {st['reached']}", 12, y)
        else:
            y = put(img, "Canlı ölçüm", 12, y + 2, 26, GOLD)
            moving = sum(1 for t in vp.tracker.tracks if t.confirmed and t.moving)
            if self.three_d:
                y = put(img, f"irtifa: {ALTS[vp.z]} m   tırmanma: {st['climbs']}", 12, y + 4)
            else:
                prec = self.tp / max(self.tp + self.fp, 1)
                rec = self.tp / max(self.tp + self.fn, 1)
                y = put(img, f"algılama kesinliği: {prec:.2f}", 12, y + 4)
                y = put(img, f"algılama duyarlılığı: {rec:.2f}", 12, y)
                y = put(img, f"hareketli iz: {moving}", 12, y)
                y = put(img, f"kenara çekilme: {st['dodges']}", 12, y)
            y = put(img, f"hedefe varış: {st['reached']}", 12, y)
        y = put(img, "Test", 12, y + 8, 26, GOLD)
        t, c = check_line(self.hits == 0, f"çarpma: {self.hits} ({self.f} kare)")
        y = put(img, t, 12, y + 2, 24, c)
        t, c = check_line(self.mism == 0, f"D* = A* ({self.checks - self.mism}/{self.checks})")
        put(img, t, 12, y, 24, c)
        if self.three_d and not self.persp:
            prof = np.zeros((150, SIDE_W, 3), np.uint8)
            prof[:] = img[-150:]
            V.draw_profile(prof, vp)
            img[-150:] = prof


# ---------------- Ekranı birleştir ----------------
def quadrant(panel):
    q = np.zeros((QH, QW, 3), np.uint8)
    q[:HEAD] = (48, 36, 26)
    put(q, panel.title, 14, 8, 30, GOLD)
    q[HEAD:HEAD + BODY, :MAP_W] = cv2.resize(panel.render(), (MAP_W, BODY), interpolation=cv2.INTER_AREA)
    side = np.full((BODY, SIDE_W, 3), (22, 22, 22), np.uint8)
    panel.side(side)
    q[HEAD:HEAD + BODY, MAP_W:] = side
    q[HEAD + BODY:] = (14, 14, 14)
    put(q, panel.ticker.tick(), 14, HEAD + BODY + 6, 24, WHITE, QW - 28)
    cv2.rectangle(q, (0, 0), (QW - 1, QH - 1), (80, 80, 80), 1)
    return q


def build_panel(i):
    if i == 0:
        return DStarPanel(7)
    if i == 1:
        return VisionPanel(3, False, False, "2) Kamera: engel algılama + hareketli engel takibi",
                           "Kuşbakışı kamera. Kırmızı: algılanan engel, turuncu: güvenlik payı, mor: hareketli "
                           "engelin yakında gireceği yer. Beyaz ok: takip edilen engelin hızı.")
    if i == 2:
        return VisionPanel(3, True, False, "3) İrtifa: 4 katmanlı 3D D* Lite (5-10-15-20 m)",
                           "Rota rengi irtifayı gösteriyor (mavi 5 m, yeşil 10, turuncu 15, pembe 20). Alçak engelin "
                           "üstünden geçiyor, yüksekten dolaşıyor. Sağ altta rotanın yükseklik profili.")
    return VisionPanel(5, True, True, "4) YOLO + eğik kamera: tespitleri haritaya indirme",
                       "Sağ üst: drone'un eğik kamerası ve YOLO kutuları. Sol: tespitler geometriyle yere "
                       "indirilmiş, her engelin boyu tahmin edilmiş. Siyah: kameranın görmediği yer.")


def summary(p):
    if isinstance(p, VisionPanel):
        return f"{p.title[:2]} carpma={p.hits} D*/A* uyusmazlik={p.mism}/{p.checks} hedef={p.vp.stats['reached']}"
    return f"{p.title[:2]} D*/A* uyusmazlik={p.mism}/{p.checks} yeniden_plan={p.replans} hedef={p.reached}"


def worker(i, conn):
    """Her çeyrek kendi işleminde (process) koşuyor: 4 sahne aynı anda hesaplanır, pencere akıcı olur."""
    panel = build_panel(i)
    while True:
        cmd = conn.recv()
        if cmd == "step":
            panel.step()
            conn.send_bytes(quadrant(panel).tobytes())
        else:
            conn.send(summary(panel))
            return


class Panels:
    """4 çeyreği paralel işlemlerde koşturur (Windows'ta da çalışır)."""

    def __init__(self):
        import multiprocessing as mp
        self.conns, self.procs = [], []
        for i in range(4):
            a, b = mp.Pipe()
            pr = mp.Process(target=worker, args=(i, b), daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)

    def step(self, f):
        for c in self.conns:
            c.send("step")
        out = np.zeros((H, W, 3), np.uint8)
        put(out, "DroneRoute  |  kameradan engel algılayıp anlık güvenli rota  |  4 adım, hepsi kendi kendine "
                 "uçuyor ve canlı test ediliyor", 16, 11, 32, GOLD)
        put(out, f"{f / FPS:5.1f} sn", W - 110, 13, 28, GRAY)
        for i, c in enumerate(self.conns):
            r, col = divmod(i, 2)
            q = np.frombuffer(c.recv_bytes(), np.uint8).reshape(QH, QW, 3)
            out[TOP + r * QH:TOP + (r + 1) * QH, col * QW:(col + 1) * QW] = q
        return out

    def close(self):
        res = []
        for c in self.conns:
            c.send("stop")
            res.append(c.recv())
        for pr in self.procs:
            pr.join(timeout=5)
        return res


def main():
    ap = argparse.ArgumentParser(description="DroneRoute: 4 adım tek ekranda (demo)")
    ap.add_argument("--sure", type=float, default=90, help="süre (saniye)")
    ap.add_argument("--kaydet", help="pencere açmadan 1920x1080 videoya yaz (ör. demo4.mp4)")
    args = ap.parse_args()

    panels = Panels()
    writer, win = None, "DroneRoute - 4 adim (F: tam ekran, SPACE: dur, Q: cik)"
    if not args.kaydet:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 900)
    frames = int(args.sure * FPS)
    paused, full, f, out = False, False, 0, None
    while f < frames:
        t0 = time.perf_counter()
        if not paused or out is None:
            f += 1
            out = panels.step(f)
            if args.kaydet:
                if writer is None:
                    writer = cv2.VideoWriter(args.kaydet, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
                writer.write(out)
                if f % (FPS * 5) == 0:
                    print(f"  {f // FPS} / {int(args.sure)} sn", flush=True)
                continue
        cv2.imshow(win, out)
        k = cv2.waitKey(max(1, int(1000 / FPS - (time.perf_counter() - t0) * 1000))) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    if writer is not None:
        writer.release()
        print(f"kaydedildi: {args.kaydet}")
    else:
        cv2.destroyAllWindows()
    for line in panels.close():
        print(line)


if __name__ == "__main__":
    main()
