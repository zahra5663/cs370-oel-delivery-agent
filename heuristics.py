import heapq
import numpy as np
from world import MOVES


def dist_to_target(world, target):
    n = world.n
    dist = np.full((n, n), np.inf)
    dist[target] = 0.0
    pq = [(0.0, target)]
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in MOVES.values():
            ur, uc = r + dr, c + dc
            if world.passable(ur, uc):
                nd = d + world.cost_map[r, c]
                if nd < dist[ur, uc]:
                    dist[ur, uc] = nd
                    heapq.heappush(pq, (nd, (ur, uc)))
        if (r, c) in world.tunnel_map:
            u = world.tunnel_map[(r, c)]
            nd = d + world.tunnel_cost
            if nd < dist[u]:
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    return dist


def mst_weight(nodes, w):
    nodes = list(nodes)
    if len(nodes) <= 1:
        return 0.0
    in_tree = {nodes[0]}
    total = 0.0
    while len(in_tree) < len(nodes):
        best = None
        for a in in_tree:
            for b in nodes:
                if b in in_tree:
                    continue
                if best is None or w[a][b] < best[0]:
                    best = (w[a][b], b)
        total += best[0]
        in_tree.add(best[1])
    return total


class Heuristics:
    def __init__(self, world):
        self.w = world
        self.stops = world.stops
        self.k = len(self.stops)
        self.min_cost = world.min_cost
        self.d_to = [dist_to_target(world, s) for s in self.stops]
        self.sym = [[0.0] * self.k for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    a = self.d_to[j][self.stops[i]]
                    b = self.d_to[i][self.stops[j]]
                    self.sym[i][j] = float(min(a, b))
        self.mst_cache = {}
        for mask in range(1 << self.k):
            rem = [i for i in range(self.k) if not (mask >> i) & 1]
            self.mst_cache[mask] = mst_weight(rem, self.sym)

    def remaining(self, mask):
        return [i for i in range(self.k) if not (mask >> i) & 1]

    def h_own(self, state):
        pos, mask = state
        rem = self.remaining(mask)
        if not rem:
            return 0.0
        nearest = min(self.d_to[i][pos] for i in rem)
        return float(nearest + self.mst_cache[mask])

    def h_manhattan_nearest(self, state):
        pos, mask = state
        rem = self.remaining(mask)
        if not rem:
            return 0.0
        d = min(abs(pos[0] - self.stops[i][0]) + abs(pos[1] - self.stops[i][1]) for i in rem)
        return float(self.min_cost * d)

    def h_farthest(self, state):
        pos, mask = state
        rem = self.remaining(mask)
        if not rem:
            return 0.0
        d = max(abs(pos[0] - self.stops[i][0]) + abs(pos[1] - self.stops[i][1]) for i in rem)
        return float(self.min_cost * d)

    def h_zero(self, state):
        return 0.0

    def inflated(self, weight):
        return lambda s: weight * self.h_own(s)

    def table(self):
        return {
            "own_mst_exact": self.h_own,
            "manhattan_nearest": self.h_manhattan_nearest,
            "farthest_stop": self.h_farthest,
            "zero (UCS)": self.h_zero,
        }


def check_consistency(problem, h):
    start = problem.initial()
    seen = {start}
    stack = [start]
    edges = 0
    violations = 0
    worst = 0.0
    while stack:
        s = stack.pop()
        for _, s2, c in problem.successors(s):
            edges += 1
            gap = h(s) - (c + h(s2))
            if gap > 1e-9:
                violations += 1
                worst = max(worst, gap)
            if s2 not in seen:
                seen.add(s2)
                stack.append(s2)
    return {"states": len(seen), "edges": edges, "violations": violations, "worst_gap": worst}
