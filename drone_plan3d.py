"""
DroneRoute — Adım 4: irtifalı planlama (drone_vision.GridPlanner'ın 3D hali)

Girdi her karede (ROS'ta planlama node'una gelen topic'lerin aynısı):
  raw, height              : şu anki engel hücreleri ve yükseklikleri (metre, bilinmiyorsa inf)
  moving_tti, moving_height: hareketli engel o hücreye kaç kare sonra gelecek ve boyu ne
Her katman z için (uçuş irtifası ALTS[z]):
  engel o katmanı kapatır  ⇔  ALTS[z] < engelin boyu + CLEARANCE
  tti_z = o katmanı kapatan engellerin en erken geliş zamanı
  sonra 2D'deki gibi: güvenlik payı (yatay), tti_z <= H_HARD duvar, H_HARD..HORIZON ceza, kenar şeridi pahalı
Hepsi DStarLite3D'ye; sadece değişen düğümler bildirilir.

Örnek: kamyon 4 m. 5 m'de uçan drone (4 + 2 > 5) kamyon gelirken kaçmak zorunda; 10 m'ye çıkarsa kamyon
altından geçip gider. Vinç kolu 35 m: hiçbir katmandan geçilmez, etrafından dolaşılır.

Drone konumu: self.drone = (x, y) (2D kodla uyumlu), self.z = katman. Rota: (x, y, z) listesi.
Hedef: verilen hücrede o an boş olan en alçak katman.
"""
import time

import numpy as np

import drone_tracker as tracking
import drone_vision as V
from drone_dstar_lite import COLS, ROWS
from drone_nofly import NOFLY_PEN, nofly_split        # E-2: yasak daireler -> duvar / pahalı tablolar
from drone_dstar3d import ALTS, CLEARANCE, NZ, DStarLite3D, path_cost3

LEVEL_ALT = np.array(ALTS, float)[:, None, None]     # (NZ, 1, 1) yayın için


class GridPlanner3D:
    def __init__(self, start=None, goal=None, predict=True):
        self.predict = predict
        self.margin = V.MARGIN
        self.grid = [[[False] * COLS for _ in range(ROWS)] for _ in range(NZ)]
        self.penalty = [[[0.0] * COLS for _ in range(ROWS)] for _ in range(NZ)]
        self.drone = start or (1, ROWS // 2)
        self.z = 0
        self.goal = goal or (COLS - 2, ROWS // 2)
        self.gz = 0
        self.raw = np.zeros((ROWS, COLS), bool)
        self.height = np.zeros((ROWS, COLS))
        self.occ3 = np.zeros((NZ, ROWS, COLS), bool)
        self.tti3 = np.full((NZ, ROWS, COLS), np.inf)
        self.near3 = self.tti3.copy()
        self.stats = dict(frames=0, replans=0, expanded=0, changed=0, steps=0, reached=0, ms=0.0, dodges=0,
                          climbs=0, high_steps=0)
        self.status = "Hazir"
        self.unseen = None
        self.zones = []                                   # E-2: yasak daireler [((x, y), yarıçap), ...]
        self._new_dstar()

    def set_nofly(self, zones):
        """E-2 (S-2): yasak daireler (hücre), bütün irtifalarda geçerli. Bir sonraki update'te uygulanır."""
        self.zones = list(zones)                          # bölgeleri sakla (kopyasını al)

    # 2D kodla (random_free_cell, çizim) uyum: en alçak katmanın görünümü
    @property
    def occ(self):
        return self.occ3[0]

    @property
    def tti(self):
        return self.tti3[self.z]

    def _new_dstar(self):
        self.dstar = DStarLite3D(self.grid, (*self.drone, self.z), (*self.goal, self.gz), self.penalty)
        self.path = self.dstar.path()

    def set_goal(self, cell):
        self.goal = cell
        free = [z for z in range(NZ) if not self.occ3[z, cell[1], cell[0]]]
        self.gz = free[0] if free else 0
        self._new_dstar()
        self.status = "Yeni hedef"

    def set_drone(self, cell, z=None):
        self.drone = cell
        if z is not None:
            self.z = z
        self._new_dstar()

    def update(self, raw, height=None, moving_tti=None, moving_height=None):
        t0 = time.perf_counter()
        self.raw = raw
        self.height = np.where(raw, np.inf, 0.0) if height is None else height
        # Katman kapatılıyor mu: irtifa < boy + pay
        blocks_now = raw[None] & (LEVEL_ALT < self.height[None] + CLEARANCE)
        tti3 = np.where(blocks_now, 0.0, np.inf)
        if self.predict and moving_tti is not None:
            mh = np.full((ROWS, COLS), np.inf) if moving_height is None else moving_height
            blocks_mv = LEVEL_ALT < mh[None] + CLEARANCE
            tti3 = np.minimum(tti3, np.where(blocks_mv, moving_tti[None], np.inf))
        self.tti3 = tti3
        near3 = np.stack([V.spread_min(tti3[z], self.margin) for z in range(NZ)])
        self.near3 = near3
        occ3 = near3 <= V.H_HARD
        pen3 = np.where(occ3 | (near3 > tracking.HORIZON), 0.0,
                        V.PEN_MAX * (tracking.HORIZON + tracking.H_STEP - near3) / (tracking.HORIZON - V.H_HARD))
        if self.predict:
            band = V.edge_band(self.unseen)
            pen3 = np.where(band[None] & ~occ3, np.maximum(pen3, V.EDGE_PEN), pen3)
        if self.unseen is not None:                   # görüş alanı dışı: bilinmiyor, pahalı
            pen3 = np.where(self.unseen[None] & ~occ3, np.maximum(pen3, V.UNSEEN_PEN), pen3)
  
        duvar, pahali = nofly_split(self.zones, self.drone, self.margin)   # içinde olduğun daire pahalı, gerisi duvar
        occ3 |= duvar[None]                               # bütün katmanlarda duvar (üstünden de geçilmez)
        pen3 = np.where(pahali[None] & ~occ3, np.maximum(pen3, NOFLY_PEN), pen3)   # içindeysen: pahalı, çık

        dx, dy = self.drone
        occ3[self.z, dy, dx] = False                  # drone'un düğümü boş kalsın (2D'deki gibi)
        pen3[self.z, dy, dx] = 0.0
        self.occ3 = occ3

        zs, ys, xs = np.nonzero((occ3 != np.array(self.grid, bool)) | (pen3 != np.array(self.penalty)))
        changed = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
        for x, y, z in changed:
            self.grid[z][y][x] = bool(occ3[z, y, x])
            self.penalty[z][y][x] = float(pen3[z, y, x])
        self.stats["frames"] += 1
        if changed:
            self.stats["changed"] += len(changed)
            n = self.dstar.cells_changed(changed)
            self.path = self.dstar.path()
            if n:
                self.stats["replans"] += 1
                self.stats["expanded"] += n
        gx, gy = self.goal
        if occ3[self.gz, gy, gx]:
            # Hedefin irtifası kapandı (ör. hedefe kamyon girdi): başka katmanı boşsa hedefi oraya taşı.
            # Yoksa (vinç/insan/kuş: hiçbir irtifadan geçilmez) açılmasını bekle.
            free = [z for z in range(NZ) if not occ3[z, gy, gx]]
            if free:
                self.gz = min(free, key=lambda z: (abs(z - self.gz), z))
                self._new_dstar()
        self.stats["ms"] += (time.perf_counter() - t0) * 1000

        if blocks_now[self.z, dy, dx]:
            self.status = f"UYARI: drone {ALTS[self.z]} m'de engelin icinde gorunuyor"
        elif self.path is None:
            if self.in_danger():
                self.status = "KACIS: hareketli engel geliyor, kenara cekiliyorum"
            else:
                self.status = "YOL YOK: hedef her irtifada kapali, bekliyorum" if occ3[self.gz, gy, gx] \
                    else "YOL YOK: bekliyorum"
        else:
            top = max(z for _, _, z in self.path)
            self.status = f"Ucusta {ALTS[self.z]} m, rota {self.path_cost():.1f} (en yuksek {ALTS[top]} m)"

    def path_cost(self):
        return path_cost3(self.grid, self.penalty, self.path) if self.path else 0.0

    def in_danger(self, node=None):
        x, y, z = node or (*self.drone, self.z)
        return self.near3[z, y, x] <= V.H_HARD

    def dodge_node(self):
        """
        Kaçış: drone'un kendisi, aynı katmandaki 8 komşu ve bir üst/alt katman içinden engelin en geç
        geleceği düğüm. Eşitlikte payın da en geç geldiği, o da eşitse hedefe en yakın olan; o da eşitse
        dikey kaçış (yerini kaybetmez, engel geçince aynı noktaya alçalır).
        """
        x, y, z = (*self.drone, self.z)
        best, best_key = (x, y, z), None
        cand = [(x, y, z + 1), (x, y, z - 1)] + [(x + dx, y + dy, z) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        for nx, ny, nz in cand:
            if not (0 <= nx < COLS and 0 <= ny < ROWS and 0 <= nz < NZ) or self.tti3[nz, ny, nx] == 0:
                continue
            if nx != x and ny != y and (self.tti3[z, y, nx] == 0 or self.tti3[z, ny, x] == 0):
                continue
            key = (min(self.tti3[nz, ny, nx], 99), min(self.near3[nz, ny, nx], 99), -self.dstar.g_of((nx, ny, nz)))
            if best_key is None or key > best_key:
                best, best_key = (nx, ny, nz), key
        return best

    def step_drone(self):
        here = (*self.drone, self.z)
        if self.path and len(self.path) > 1:
            nxt = self.path[1]
            self.path = self.path[1:]
        elif self.predict and self.in_danger():
            nxt = self.dodge_node()
            self.path = None
            if nxt != here:
                self.stats["dodges"] += 1
        else:
            return False
        if nxt == here:
            return False
        if nxt[2] > self.z:
            self.stats["climbs"] += 1
        self.drone, self.z = nxt[:2], nxt[2]
        self.dstar.move_to(nxt)
        self.stats["steps"] += 1
        self.stats["high_steps"] += self.z > 0
        if self.drone == self.goal and self.z == self.gz:
            self.stats["reached"] += 1
            return True
        return False


class VisionPlanner3D(GridPlanner3D):
    """Perception + GridPlanner3D tek nesnede (pencere ve testler için)."""

    def __init__(self, detector, start=None, goal=None, predict=True):
        self.perception = V.Perception(detector)
        self.det = detector
        self.filter = self.perception.filter
        self.tracker = self.perception.tracker
        super().__init__(start, goal, predict)

    @property
    def last_mask(self):
        return self.perception.last_mask

    def process(self, frame, height=None):
        t0 = time.perf_counter()
        p = self.perception
        raw, moving = p.process(frame, height)
        self.unseen = p.unseen
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        self.update(raw, p.height, moving, p.moving_height)
