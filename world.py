import time
import numpy as np
import config
from oel_instance import TERRAINS

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
DIRS = list(MOVES)
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}


class World:
    def __init__(self, overrides=None):
        self.inst, self.grid, self.params = config.load(overrides)
        self.n = self.inst["grid_size"]
        self.start = tuple(self.inst["start"])
        self.stops = [tuple(s) for s in self.inst["delivery_stops"]]
        if self.params.get("start"):
            self.start = tuple(self.params["start"])
        if self.params.get("stops"):
            self.stops = [tuple(s) for s in self.params["stops"]]
        self.check_cells()
        self.slip_p = self.params["slip_probability"]
        self.wind_p = self.params["wind_probability"]
        self.tunnel_cost = self.params["tunnel_cost"]
        self.cost_map = self.build_cost_map()
        self.min_cost = int(min(self.inst["terrain_cost"].values()))
        rng = np.random.default_rng(config.seed_from_ids("|world"))
        self.wind_dir = DIRS[int(rng.integers(0, 4))]
        self.tunnels = self.build_tunnels(rng)
        self.tunnel_map = {}
        for a, b in self.tunnels:
            self.tunnel_map[a] = b
            self.tunnel_map[b] = a

    def check_cells(self):
        cells = [self.start] + list(self.stops)
        for c in cells:
            if not self.passable(*c):
                raise ValueError(f"cell {c} is outside the grid or blocked")
        if len(set(cells)) != len(cells):
            raise ValueError("start and stops must be different cells")
        seen = {self.start}
        queue = [self.start]
        while queue:
            r, c = queue.pop()
            for dr, dc in MOVES.values():
                p = (r + dr, c + dc)
                if self.passable(*p) and p not in seen:
                    seen.add(p)
                    queue.append(p)
        lost = [s for s in self.stops if s not in seen]
        if lost:
            raise ValueError(f"stops {lost} cannot be reached from the start without tunnels")

    def build_cost_map(self):
        table = np.array([self.inst["terrain_cost"][t] for t in TERRAINS], dtype=float)
        return np.where(self.grid >= 0, table[np.clip(self.grid, 0, None)], np.inf)

    def build_cost_map_loop(self):
        out = [[0.0] * self.n for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                g = self.grid[r][c]
                out[r][c] = float("inf") if g < 0 else float(self.inst["terrain_cost"][TERRAINS[g]])
        return np.array(out)

    def build_tunnels(self, rng):
        taken = {self.start, *self.stops}
        free = [tuple(map(int, p)) for p in np.argwhere(self.grid >= 0) if tuple(map(int, p)) not in taken]
        span = self.params["min_tunnel_span"]
        pairs = []
        tries = 0
        while len(pairs) < self.params["num_tunnel_pairs"] and tries < 10000:
            tries += 1
            i, j = rng.choice(len(free), 2, replace=False)
            a, b = free[int(i)], free[int(j)]
            used = {x for p in pairs for x in p}
            if a in used or b in used:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < span:
                continue
            pairs.append((a, b))
        return pairs

    def in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def passable(self, r, c):
        return self.in_bounds(r, c) and self.grid[r, c] >= 0

    def neighbors(self, pos):
        r, c = pos
        out = []
        for a, (dr, dc) in MOVES.items():
            nr, nc = r + dr, c + dc
            if self.passable(nr, nc):
                out.append((a, (nr, nc), float(self.cost_map[nr, nc])))
        if pos in self.tunnel_map:
            out.append(("T", self.tunnel_map[pos], float(self.tunnel_cost)))
        return out

    def stochastic_step(self, pos, action, rng):
        if action == "T":
            return self.tunnel_map[pos], float(self.tunnel_cost), {"slip": False, "wind": False}
        info = {"slip": False, "wind": False}
        act = action
        if rng.random() < self.slip_p:
            act = PERP[action][int(rng.integers(0, 2))]
            info["slip"] = True
        dr, dc = MOVES[act]
        nr, nc = pos[0] + dr, pos[1] + dc
        cost = 0.0
        if self.passable(nr, nc):
            cur = (nr, nc)
            cost += float(self.cost_map[nr, nc])
        else:
            cur = pos
            cost += 1.0
        if rng.random() < self.wind_p:
            wr, wc = MOVES[self.wind_dir]
            dr2, dc2 = cur[0] + wr, cur[1] + wc
            if self.passable(dr2, dc2):
                cur = (dr2, dc2)
                cost += float(self.cost_map[dr2, dc2])
                info["wind"] = True
        return cur, cost, info

    def benchmark_cost_map(self, repeats=20):
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.build_cost_map()
        t_np = (time.perf_counter() - t0) / repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.build_cost_map_loop()
        t_loop = (time.perf_counter() - t0) / repeats
        assert np.array_equal(self.build_cost_map(), self.build_cost_map_loop())
        return {"numpy_s": t_np, "loop_s": t_loop, "speedup": t_loop / t_np}


if __name__ == "__main__":
    w = World()
    print(config.STAMP)
    print("instance:", w.inst["instance_id"], "grid:", w.n, "start:", w.start, "stops:", w.stops)
    print("wind direction:", w.wind_dir, "tunnels:", w.tunnels)
    print("neighbors of start:", w.neighbors(w.start))
    b = w.benchmark_cost_map()
    print(f"numpy {b['numpy_s']*1e6:.1f} us | loop {b['loop_s']*1e6:.1f} us | speedup {b['speedup']:.1f}x")
    print(config.STAMP)
