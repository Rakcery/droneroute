"""D* Lite rota maliyetlerini rastgele haritalarda A* ile karşılaştırır. Çalıştır: python tests/test_dstar_vs_astar.py"""
import sys, random, os
os.environ["SDL_VIDEODRIVER"]="dummy"; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"]="1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import drone_dstar_lite as M
checks=mism=nopath=0; dsum=asum=0
for trial in range(300):
    random.seed(trial)
    grid, start, goal = M.make_map()
    ds = M.DStarLite(grid, start, goal)
    drone = start
    for step in range(60):
        # random changes
        cells=[]
        for _ in range(random.randint(1,6)):
            c=(random.randrange(M.COLS), random.randrange(M.ROWS))
            if c in (drone, goal): continue
            grid[c[1]][c[0]] = not grid[c[1]][c[0]]; cells.append(c)
        d = ds.cells_changed(cells)
        p = ds.path(); ap, cl = M.astar(grid, drone, goal)
        checks += 1
        if (p is None) != (ap is None): mism += 1; print("existence mismatch", trial, step); continue
        if p is None: nopath += 1; continue
        if abs(M.path_cost(p) - M.path_cost(ap)) > 1e-9: mism += 1; print("cost mismatch", trial, step, M.path_cost(p), M.path_cost(ap)); continue
        assert all(not grid[y][x] for x,y in p)
        assert all(ds.cost(a,b) < M.INF for a,b in zip(p,p[1:]))
        dsum += d; asum += len(cl)
        # move drone 0-3 steps
        for _ in range(random.randint(0,3)):
            if len(p) > 1: p = p[1:]; drone = p[0]; ds.move_to(drone)
        if drone == goal: break
print(f"checks={checks} mismatches={mism} nopath={nopath} D*={dsum} A*={asum} ratio={dsum/asum:.2f}")

# Aynısı hücre cezalarıyla (hareketli engel tahmini bunu kullanıyor): duvar + ceza değişiklikleri karışık
checks=mism=0
for trial in range(150):
    random.seed(1000 + trial)
    grid, start, goal = M.make_map()
    pen = [[0.0] * M.COLS for _ in range(M.ROWS)]
    ds = M.DStarLite(grid, start, goal, pen)
    drone = start
    for step in range(40):
        cells=[]
        for _ in range(random.randint(1,8)):
            c=(random.randrange(M.COLS), random.randrange(M.ROWS))
            if c in (drone, goal): continue
            if random.random() < 0.3: grid[c[1]][c[0]] = not grid[c[1]][c[0]]
            else: pen[c[1]][c[0]] = random.choice((0.0, 0.0, 0.5, 2.0, 4.0))
            cells.append(c)
        ds.cells_changed(cells)
        p = ds.path(); ap, _ = M.astar(grid, drone, goal, pen)
        checks += 1
        if (p is None) != (ap is None) or (p and abs(M.path_cost(p, pen) - M.path_cost(ap, pen)) > 1e-9):
            mism += 1; print("weighted mismatch", trial, step)
        if p:
            for _ in range(random.randint(0,3)):
                if len(p) > 1: p = p[1:]; drone = p[0]; ds.move_to(drone)
        if drone == goal: break
print(f"cezali: checks={checks} mismatches={mism}")
