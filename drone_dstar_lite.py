"""
DroneRoute mini demo — D* Lite + dinamik engel + artımlı (incremental) yeniden planlama
Gereken: pip install pygame

A*'dan farkı: her değişiklikte rotayı sıfırdan hesaplamak yerine D* Lite sadece
değişiklikten etkilenen düğümleri günceller. Arama hedeften drone'a doğru yapılır,
böylece drone hareket ettikçe eski hesaplar geçerli kalır (km ile anahtarlar düzeltilir).
HUD'da her yeniden planda D* Lite'ın kaç düğüm güncellediği ile aynı anda tam A*
çalışsaydı kaç düğüm açacağı yan yana gösterilir.

Kontroller:
  Sol tık (sürükle)  : duvar ekle
  Sağ tık (sürükle)  : duvar sil
  G                  : hedefi farenin olduğu yere taşı (D* Lite yeniden kurulur)
  SPACE              : duraklat / devam
  E                  : rastgele engel üretmeyi aç/kapat
  R                  : yeni harita
"""
import heapq
import math
import random
import sys

# ---------------- Ayarlar ----------------
COLS, ROWS = 40, 28
CELL = 22
HUD_H = 78
W, H = COLS * CELL, ROWS * CELL + HUD_H
FPS = 30
MOVE_EVERY = 5        # drone kaç karede bir adım atsın
SPAWN_EVERY = 40      # kaç karede bir rastgele engel çıksın
WALL_DENSITY = 0.18   # başlangıç haritasındaki duvar oranı

C_BG = (18, 20, 28)
C_GRID = (30, 33, 44)
C_WALL = (90, 95, 115)
C_NEW_WALL = (220, 90, 70)
C_EXPLORED = (35, 55, 80)
C_PATH = (80, 200, 255)
C_DRONE = (255, 210, 60)
C_GOAL = (90, 230, 120)
C_TEXT = (220, 220, 230)
C_TEXT2 = (150, 200, 255)

INF = math.inf
EPS = 1e-9
SQRT2 = math.sqrt(2)
# 8 yön: (dx, dy, maliyet)
DIRS = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def heuristic(a, b):
    # Octile mesafe: 8 yönlü hareket için tutarlı (admissible) sezgisel
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def weighted(base, penalty, u, v):
    """
    Kenar maliyeti = uzunluk * (1 + iki hücrenin ceza ortalaması).
    penalty (isteğe bağlı) grid ile aynı boyutta float listesi: 0 = normal hücre, 4 = 5 kat pahalı.
    Hareketli engelin yakında geleceği hücreler böyle "pahalı" yapılıyor. Ceza >= 0 olduğu sürece
    maliyet >= octile mesafe, yani sezgisel hâlâ tutarlı; A* ve D* Lite optimal kalır.
    """
    if penalty is None:
        return base
    return base * (1.0 + 0.5 * (penalty[u[1]][u[0]] + penalty[v[1]][v[0]]))


# ---------------- A* (sıfırdan) — harita üretimi ve karşılaştırma için ----------------
def neighbors(node, grid):
    x, y = node
    for dx, dy, cost in DIRS:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < COLS and 0 <= ny < ROWS) or grid[ny][nx]:
            continue
        # Çaprazda köşe kesmeyi engelle (drone duvarın köşesinden sürtünmesin)
        if dx and dy and (grid[y][nx] or grid[ny][x]):
            continue
        yield (nx, ny), cost


def astar(grid, start, goal, penalty=None):
    """(yol, keşfedilen_hücreler) döner. Yol yoksa yol = None. penalty: bkz. weighted()."""
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_cost = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue  # aynı düğümün eski (daha pahalı) kopyası
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1], closed
        closed.add(cur)

        for nb, cost in neighbors(cur, grid):
            ng = g + weighted(cost, penalty, cur, nb)
            if ng < g_cost.get(nb, INF):
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + heuristic(nb, goal), ng, nb))

    return None, closed


# ---------------- D* Lite (sıfırdan, Koenig & Likhachev 2002, optimize sürüm) ----------------
class DStarLite:
    """
    Arama hedeften başlar: g(u) = u'dan hedefe en kısa mesafe tahmini,
    rhs(u) = bir adım ileriye bakan değer (min over komşu c(u,s') + g(s')).
    g == rhs ise düğüm tutarlıdır; tutarsız düğümler öncelik kuyruğunda bekler.
    Duvar hücreleri de düğümdür ama tüm kenarlarının maliyeti sonsuzdur.
    Izgara yönsüz olduğu için öncül (pred) = ardıl (succ).
    3D sürüm (drone_dstar3d.py) bu sınıfı miras alıp sadece heur/cost/nbrs'i değiştirir.
    """
    heur = staticmethod(heuristic)
    max_len = COLS * ROWS

    def __init__(self, grid, start, goal, penalty=None):
        self.grid = grid          # haritayla aynı liste: değişiklikleri buradan görür
        self.penalty = penalty    # isteğe bağlı hücre cezaları (aynı şekilde yerinde güncellenir)
        self.start = start
        self.goal = goal
        self.last = start         # km için son yeniden planlama anındaki konum
        self.km = 0.0
        self.g = {}
        self.rhs = {goal: 0.0}
        self.open = []            # (k1, k2, sürüm, düğüm) — tembel silmeli yığın
        self.in_open = {}         # düğüm -> (anahtar, sürüm); yığındaki geçerli kayıt bu
        self.ver = 0
        self.last_expanded = set()
        self._push(goal, self.calc_key(goal))
        self.compute()

    # --- yardımcılar ---
    def g_of(self, u):
        return self.g.get(u, INF)

    def rhs_of(self, u):
        return self.rhs.get(u, INF)

    def calc_key(self, u):
        m = min(self.g_of(u), self.rhs_of(u))
        return (m + self.heur(self.start, u) + self.km, m)

    def cost(self, u, v):
        (x1, y1), (x2, y2) = u, v
        gr = self.grid
        if gr[y1][x1] or gr[y2][x2]:
            return INF
        if x1 != x2 and y1 != y2:
            if gr[y1][x2] or gr[y2][x1]:   # köşe kesme yok
                return INF
            return weighted(SQRT2, self.penalty, u, v)
        return weighted(1.0, self.penalty, u, v)

    @staticmethod
    def nbrs(u):
        x, y = u
        for dx, dy, _ in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS:
                yield (nx, ny)

    def best_rhs(self, u):
        return min((self.cost(u, s) + self.g_of(s) for s in self.nbrs(u)), default=INF)

    # --- öncelik kuyruğu ---
    def _push(self, u, key):
        self.ver += 1
        self.in_open[u] = (key, self.ver)
        heapq.heappush(self.open, (key[0], key[1], self.ver, u))

    def _remove(self, u):
        self.in_open.pop(u, None)   # yığındaki kayıt bayatlar, pop'ta atlanır

    def _top(self):
        while self.open:
            k1, k2, ver, u = self.open[0]
            cur = self.in_open.get(u)
            if cur is not None and cur[1] == ver:
                return (k1, k2), u
            heapq.heappop(self.open)    # bayat kayıt
        return None, None

    def update_vertex(self, u):
        if self.g_of(u) != self.rhs_of(u):
            self._push(u, self.calc_key(u))   # ekle ya da anahtarı güncelle
        elif u in self.in_open:
            self._remove(u)

    # --- ana döngü ---
    def compute(self):
        expanded = set()
        s = self.start
        while True:
            k_old, u = self._top()
            if k_old is None:
                break
            # Kayan nokta: 1+√2+1 ile √2+1+1 son basamakta farklı çıkabilir, eşit anahtarlar
            # yanlışlıkla "büyük" görünüp arama erken durmasın diye EPS payı bırakıyoruz.
            # Fazladan birkaç düğüm açmak zararsız, erken durmak rotayı bozar.
            if not (k_old[0] < self.calc_key(s)[0] + EPS or self.rhs_of(s) != self.g_of(s)):
                break
            k_new = self.calc_key(u)
            if k_old < k_new:                 # km yüzünden anahtar eskimiş: sadece güncelle
                self._push(u, k_new)
                continue
            heapq.heappop(self.open)
            del self.in_open[u]
            expanded.add(u)
            g_u, rhs_u = self.g_of(u), self.rhs_of(u)
            if g_u > rhs_u:                   # fazla tutarlı: g'yi düşür, komşulara yay
                self.g[u] = rhs_u
                for p in self.nbrs(u):
                    if p != self.goal:
                        self.rhs[p] = min(self.rhs_of(p), self.cost(p, u) + rhs_u)
                    self.update_vertex(p)
            else:                             # eksik tutarlı: g'yi sonsuz yap, bağımlıları düzelt
                self.g[u] = INF
                for p in list(self.nbrs(u)) + [u]:
                    if p != self.goal and (p == u or self.rhs_of(p) == self.cost(p, u) + g_u):
                        self.rhs[p] = self.best_rhs(p)
                    self.update_vertex(p)
        self.last_expanded = expanded
        return len(expanded)

    def move_to(self, new_start):
        # Drone hareket etti; kenar değişmedikçe yeniden hesap gerekmez
        self.start = new_start

    def cells_changed(self, cells):
        """
        Duvar eklenen/silinen ya da cezası değişen hücreleri bildir. Köşe kesme yasak olduğu için bir hücre
        değişince sadece kendi kenarları değil, komşuları arasındaki çapraz kenarlar da
        değişir; bu kenarların tüm uç noktaları hücrenin 3x3 çevresinde kalır.
        Dönüş: güncellenen (genişletilen) düğüm sayısı.
        """
        self.km += self.heur(self.last, self.start)
        self.last = self.start
        affected = set()
        for c in cells:
            affected.add(c)
            affected.update(self.nbrs(c))
        for u in affected:
            if u != self.goal:
                self.rhs[u] = self.best_rhs(u)
            self.update_vertex(u)
        return self.compute()

    def path(self):
        """Drone'dan hedefe rota; her adımda c(u,s') + g(s')'yi en küçükleyen komşuya git."""
        if self.rhs_of(self.start) == INF:
            return None
        cur, path = self.start, [self.start]
        while cur != self.goal:
            best, best_v = None, INF
            for s in self.nbrs(cur):
                v = self.cost(cur, s) + self.g_of(s)
                if v < best_v:
                    best, best_v = s, v
            if best is None or len(path) > self.max_len:
                return None
            cur = best
            path.append(cur)
        return path


def path_cost(path, penalty=None):
    return sum(weighted(SQRT2 if a[0] != b[0] and a[1] != b[1] else 1.0, penalty, a, b)
               for a, b in zip(path, path[1:]))


# ---------------- Harita ----------------
def make_map():
    """Başlangıçla hedef arasında yol olan rastgele bir harita üretir."""
    start, goal = (1, ROWS // 2), (COLS - 2, ROWS // 2)
    while True:
        grid = [[random.random() < WALL_DENSITY for _ in range(COLS)] for _ in range(ROWS)]
        for (x, y) in (start, goal):
            grid[y][x] = False
        path, _ = astar(grid, start, goal)
        if path:
            return grid, start, goal


def cell_at(pos):
    x, y = pos[0] // CELL, pos[1] // CELL
    if 0 <= x < COLS and 0 <= y < ROWS:
        return x, y
    return None


def boxed_in(grid, cell):
    """Hücreden hiçbir komşuya geçilemiyor mu? (köşe kesme yasak olduğu için çapraz boşluk sayılmaz)"""
    return next(neighbors(cell, grid), None) is None


def no_path_status(s):
    # Duvarlar kendiliğinden kalkmıyor, kullanıcıya ne yapacağını söyleyelim
    if boxed_in(s["grid"], s["goal"]):
        return "YOL YOK — hedefin etrafı kapalı (köşeden geçilmez), sağ tıkla duvar sil"
    if boxed_in(s["grid"], s["drone"]):
        return "YOL YOK — drone'un etrafı kapalı (köşeden geçilmez), sağ tıkla duvar sil"
    return "YOL YOK — sağ tıkla duvar sil ya da G ile hedefi taşı"


def near(a, b):
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1


# ---------------- Ana döngü ----------------
def main():
    import pygame   # sadece demo penceresi için; algoritma (ve ROS 2 node'ları) pygame'siz çalışır

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DroneRoute — D* Lite artımlı yeniden planlama")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    def new_planner(s, reason):
        s["dstar"] = DStarLite(s["grid"], s["drone"], s["goal"])
        s["path"] = s["dstar"].path()
        s["explored"] = s["dstar"].last_expanded
        d = len(s["explored"])
        a = len(astar(s["grid"], s["drone"], s["goal"])[1])
        s["last_d"], s["last_a"] = d, a
        s["status"] = reason if s["path"] else no_path_status(s)

    def reset():
        grid, drone, goal = make_map()
        s = dict(grid=grid, drone=drone, goal=goal, new_walls={}, replans=0, steps=0,
                 sum_d=0, sum_a=0)
        new_planner(s, "Uçuşta (D* Lite ilk arama)")
        return s

    s = reset()
    paused, spawning, frame = False, True, 0

    def notify(cells, reason):
        """Değişen hücreleri D* Lite'a ver; sadece etkilenen kısım yeniden hesaplanır."""
        if not cells:
            return
        had_path = s["path"] is not None
        d = s["dstar"].cells_changed(cells)
        s["path"] = s["dstar"].path()
        if d == 0:
            return  # değişiklik rotayı hiç etkilemedi, iş yok
        a = len(astar(s["grid"], s["drone"], s["goal"])[1])  # sadece kıyas için
        s["replans"] += 1
        s["last_d"], s["last_a"] = d, a
        s["sum_d"] += d
        s["sum_a"] += a
        s["explored"] = s["dstar"].last_expanded
        if s["path"] is None:
            s["status"] = no_path_status(s)
        elif not had_path:
            s["status"] = "Yol bulundu, devam"
        else:
            s["status"] = f"Yeniden planlandı ({reason})"

    while True:
        frame += 1

        # --- Girdi ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_e:
                    spawning = not spawning
                elif e.key == pygame.K_r:
                    s = reset()
                elif e.key == pygame.K_g:
                    c = cell_at(pygame.mouse.get_pos())
                    if c and not s["grid"][c[1]][c[0]] and c != s["drone"]:
                        s["goal"] = c
                        new_planner(s, "Yeni hedef (D* Lite yeniden kuruldu)")

        buttons = pygame.mouse.get_pressed()
        c = cell_at(pygame.mouse.get_pos())
        if c and c not in (s["drone"], s["goal"]):
            x, y = c
            if buttons[0] and not s["grid"][y][x]:
                s["grid"][y][x] = True
                notify([c], "elle engel")
            elif buttons[2] and s["grid"][y][x]:
                s["grid"][y][x] = False
                notify([c], "duvar silindi")  # belki artık daha kısa yol var

        if not paused:
            # --- Rastgele engel: yarı ihtimalle tam rotanın önüne düşer ---
            if spawning and frame % SPAWN_EVERY == 0:
                if s["path"] and len(s["path"]) > 4 and random.random() < 0.5:
                    target = random.choice(s["path"][3:])  # drone'un hemen önü değil, biraz ilerisi
                else:
                    target = (random.randrange(COLS), random.randrange(ROWS))
                tx, ty = target
                changed = []
                for dx in (0, 1):
                    for dy in (0, 1):  # 2x2'lik engel (vinç, ağaç vs. gibi düşün)
                        nx, ny = tx + dx, ty + dy
                        # Drone'un ve hedefin hemen etrafına düşmesin, yoksa onları hapsedebilir
                        if (0 <= nx < COLS and 0 <= ny < ROWS and not near((nx, ny), s["drone"])
                                and not near((nx, ny), s["goal"]) and not s["grid"][ny][nx]):
                            s["grid"][ny][nx] = True
                            s["new_walls"][(nx, ny)] = frame
                            changed.append((nx, ny))
                notify(changed, "yeni engel")

            # --- Drone hareketi ---
            # D* Lite her değişikliği anında işlediği için rota hep günceldir
            if frame % MOVE_EVERY == 0 and s["path"] and len(s["path"]) > 1:
                s["drone"] = s["path"][1]
                s["path"] = s["path"][1:]
                s["dstar"].move_to(s["drone"])
                s["steps"] += 1
                if s["drone"] == s["goal"]:
                    s["status"] = "HEDEFE ULAŞILDI — R ile yeni harita"

        # --- Çizim ---
        screen.fill(C_BG)
        for (x, y) in s["explored"]:  # son planda D* Lite'ın dokunduğu düğümler
            pygame.draw.rect(screen, C_EXPLORED, (x * CELL, y * CELL, CELL, CELL))
        for y in range(ROWS):
            for x in range(COLS):
                r = (x * CELL, y * CELL, CELL, CELL)
                if s["grid"][y][x]:
                    fresh = frame - s["new_walls"].get((x, y), -999) < 60
                    pygame.draw.rect(screen, C_NEW_WALL if fresh else C_WALL, r)
                pygame.draw.rect(screen, C_GRID, r, 1)

        if s["path"] and len(s["path"]) > 1:
            pts = [(x * CELL + CELL // 2, y * CELL + CELL // 2) for x, y in s["path"]]
            pygame.draw.lines(screen, C_PATH, False, pts, 3)

        gx, gy = s["goal"]
        pygame.draw.rect(screen, C_GOAL, (gx * CELL + 3, gy * CELL + 3, CELL - 6, CELL - 6))
        dx, dy = s["drone"]
        pygame.draw.circle(screen, C_DRONE, (dx * CELL + CELL // 2, dy * CELL + CELL // 2), CELL // 2 - 2)

        info = (f"adım: {s['steps']}  |  plan: {s['replans']}  |  "
                f"engel üretimi: {'açık' if spawning else 'kapalı'}{'  |  DURAKLATILDI' if paused else ''}")
        ratio = f"  (%{100 * s['sum_d'] / s['sum_a']:.0f})" if s["sum_a"] else ""
        stats = (f"Son plan: D* Lite {s['last_d']} düğüm, tam A* olsa {s['last_a']} düğüm  |  "
                 f"toplam: D* {s['sum_d']} / A* {s['sum_a']}{ratio}")
        screen.blit(font.render(s["status"], True, C_TEXT), (10, ROWS * CELL + 8))
        screen.blit(font.render(info, True, C_TEXT), (10, ROWS * CELL + 30))
        screen.blit(font.render(stats, True, C_TEXT2), (10, ROWS * CELL + 52))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
