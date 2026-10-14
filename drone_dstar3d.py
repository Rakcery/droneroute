"""
DroneRoute — Adım 4: irtifalı (3D) D* Lite + karşılaştırma için 3D A* (sıfırdan)

2D grid'in her hücresi NZ irtifa katmanına bölünür: düğüm = (x, y, z). Drone:
  - aynı katmanda 8 yöne gider (2D'deki gibi, köşe kesme yok)
  - bulunduğu sütunda bir katman yukarı / aşağı çıkar
Maliyet (yönlü, çünkü tırmanmak alçalmaktan pahalı):
  yatay  : uzunluk * (1 + ceza) * (1 + ALT_COST * z)    yüksekte uçmak biraz pahalı (rüzgâr, enerji, mevzuat)
  yukarı : CLIMB   * (1 + ceza)
  aşağı  : DESCEND * (1 + ceza)
Böylece drone gerekmedikçe alçakta kalır; engelin etrafından dolaşmak, üstünden aşmaktan pahalıysa tırmanır.

Sezgisel: octile(x, y) + |dz| * min(CLIMB, DESCEND). Her kenar maliyeti en az bu kadar, yani sezgisel
tutarlı, A* ve D* Lite optimal. D* Lite algoritması 2D sürümle aynı (miras), sadece komşuluk ve maliyet farklı.

Engel bilgisi z katmanlı listeler: grid[z][y][x] (True = dolu), penalty[z][y][x] (>= 0).
"""
import heapq
import math

from drone_dstar_lite import COLS, ROWS, SQRT2, DStarLite, heuristic as octile

INF = math.inf
ALTS = (5, 10, 15, 20)     # katmanların uçuş irtifası (metre)
NZ = len(ALTS)
CLEARANCE = 2.0            # engelin tepesinden en az bu kadar yukarıda uç (metre)
CLIMB = 3.0                # bir katman tırmanma maliyeti (yatay bir hücre = 1)
DESCEND = 2.0              # bir katman alçalma maliyeti
ALT_COST = 0.05            # her katman yatay hareketi %5 pahalılaştırır: gerekmedikçe alçak uç
DIRS2 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def heuristic3(a, b):
    return octile(a[:2], b[:2]) + abs(a[2] - b[2]) * min(CLIMB, DESCEND)


def edge_cost(grid, penalty, u, v):
    """u → v yönlü kenar maliyeti (komşu olduklarını varsayar). Dolu uç ya da köşe kesme: INF."""
    (x1, y1, z1), (x2, y2, z2) = u, v
    if grid[z1][y1][x1] or grid[z2][y2][x2]:
        return INF
    pen = 1.0 if penalty is None else 1.0 + 0.5 * (penalty[z1][y1][x1] + penalty[z2][y2][x2])
    if z1 != z2:
        return (CLIMB if z2 > z1 else DESCEND) * pen
    if x1 != x2 and y1 != y2:
        if grid[z1][y1][x2] or grid[z1][y2][x1]:     # köşe kesme yok (aynı katmanda)
            return INF
        return SQRT2 * pen * (1.0 + ALT_COST * z1)
    return pen * (1.0 + ALT_COST * z1)


def nbrs3(u):
    x, y, z = u
    for dx, dy, _ in DIRS2:
        nx, ny = x + dx, y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            yield (nx, ny, z)
    if z + 1 < NZ:
        yield (x, y, z + 1)
    if z > 0:
        yield (x, y, z - 1)


def path_cost3(grid, penalty, path):
    return sum(edge_cost(grid, penalty, a, b) for a, b in zip(path, path[1:]))


def astar3(grid, start, goal, penalty=None):
    """Sıfırdan 3D A*: (yol, keşfedilen düğüm sayısı). Yol yoksa None."""
    open_heap = [(heuristic3(start, goal), 0.0, start)]
    came, g_cost, closed = {}, {start: 0.0}, set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1], len(closed)
        closed.add(cur)
        for nb in nbrs3(cur):
            ng = g + edge_cost(grid, penalty, cur, nb)
            if ng < g_cost.get(nb, INF):
                g_cost[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + heuristic3(nb, goal), ng, nb))
    return None, len(closed)


class DStarLite3D(DStarLite):
    """2D D* Lite'ın aynısı; düğüm (x, y, z), komşuluk nbrs3, maliyet edge_cost (yönlü)."""
    heur = staticmethod(heuristic3)
    max_len = COLS * ROWS * NZ

    def cost(self, u, v):
        return edge_cost(self.grid, self.penalty, u, v)

    nbrs = staticmethod(nbrs3)
