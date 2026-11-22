from collections import deque
from world import MOVES

WIN = 10000.0
LOSS = -10000.0
INF = float("inf")

DEFAULT_WEIGHTS = {"delivered": 100.0, "cost": 1.0, "safety": 3.0, "margin": 3.0}


def step_distances(world, target, use_tunnels):
    n = world.n
    dist = [[99] * n for _ in range(n)]
    dist[target[0]][target[1]] = 0
    q = deque([target])
    while q:
        r, c = q.popleft()
        d = dist[r][c]
        nxt = []
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if world.passable(nr, nc):
                nxt.append((nr, nc))
        if use_tunnels and (r, c) in world.tunnel_map:
            nxt.append(world.tunnel_map[(r, c)])
        for p in nxt:
            if dist[p[0]][p[1]] == 99:
                dist[p[0]][p[1]] = d + 1
                q.append(p)
    return dist


class Game:
    def __init__(self, world, heur, weights=None, window=3):
        self.w = world
        self.n = world.n
        self.stops = world.stops
        self.k = len(world.stops)
        self.stop_idx = {s: i for i, s in enumerate(world.stops)}
        self.full = (1 << self.k) - 1
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.window = window
        self.rival_weights = {"dist": 2.0, "target": 2.0}
        self.grid = world.grid.tolist()
        self.cost = world.cost_map.tolist()
        self.tunnel_map = world.tunnel_map
        self.tunnel_cost = float(world.tunnel_cost)
        self.dcost = [d.tolist() for d in heur.d_to]
        self.mst = heur.mst_cache
        self.sa = [step_distances(world, s, True) for s in world.stops]
        self.sr = [step_distances(world, s, False) for s in world.stops]
        self.moves_list = list(MOVES.items())

    def set_weights(self, weights):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(weights)

    def remaining(self, mask):
        return [i for i in range(self.k) if not (mask >> i) & 1]

    def update_mask(self, mask, pos):
        i = self.stop_idx.get(pos)
        if i is not None:
            mask |= 1 << i
        return mask

    def h_cost(self, pos, mask):
        rem = self.remaining(mask)
        if not rem:
            return 0.0
        r, c = pos
        return min(self.dcost[i][r][c] for i in rem) + self.mst[mask]

    def eval_agent(self, apos, rpos, mask, g):
        W = self.weights
        rem = self.remaining(mask)
        delivered = self.k - len(rem)
        h = self.h_cost(apos, mask)
        safety = min(abs(apos[0] - rpos[0]) + abs(apos[1] - rpos[1]), 8)
        margin = 0
        if rem:
            margin = max(max(-10, min(10, self.sr[i][rpos[0]][rpos[1]] - self.sa[i][apos[0]][apos[1]])) for i in rem)
        return W["delivered"] * delivered - W["cost"] * (g + h) + W["safety"] * safety + W["margin"] * margin

    def eval_interceptor(self, apos, rpos, mask, g):
        rem = self.remaining(mask)
        delivered = self.k - len(rem)
        if not rem:
            return 100.0 * delivered
        target = min(rem, key=lambda i: self.sa[i][apos[0]][apos[1]])
        rsteps = self.sr[target][rpos[0]][rpos[1]]
        dist = min(abs(apos[0] - rpos[0]) + abs(apos[1] - rpos[1]), 10)
        return 100.0 * delivered + self.rival_weights["dist"] * dist + self.rival_weights["target"] * min(rsteps, 20)

    def in_window(self, root, p):
        return abs(p[0] - root[0]) <= self.window and abs(p[1] - root[1]) <= self.window

    def agent_moves(self, apos, root):
        out = []
        r, c = apos
        inside = self.in_window(root, apos)
        for a, (dr, dc) in self.moves_list:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n and self.grid[nr][nc] >= 0:
                if (not inside) or self.in_window(root, (nr, nc)):
                    out.append((a, (nr, nc), self.cost[nr][nc]))
        if apos in self.tunnel_map:
            out.append(("T", self.tunnel_map[apos], self.tunnel_cost))
        return out

    def rival_moves(self, rpos, root):
        out = []
        r, c = rpos
        inside = self.in_window(root, rpos)
        for a, (dr, dc) in self.moves_list:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n and self.grid[nr][nc] >= 0:
                if (not inside) or self.in_window(root, (nr, nc)):
                    out.append((nr, nc))
        return out


class Search:
    def __init__(self, game):
        self.g = game
        self.nodes = 0

    def run(self, apos, rpos, mask, depth, first, who, pruning, ordering):
        self.nodes = 0
        self.depth = depth
        self.root = apos
        self.pruning = pruning
        self.ordering = ordering
        self.eval = self.g.eval_agent if who == "agent" else self.g.eval_interceptor
        alpha, beta = -INF, INF
        if first == "A":
            best_v, best_m = -INF, None
            moves = self.order_agent(self.g.agent_moves(apos, self.root), mask)
            for a, np_, c in moves:
                m = self.g.update_mask(mask, np_)
                self.nodes += 1
                if np_ == rpos:
                    v = LOSS + 1
                elif m == self.g.full:
                    v = WIN - 1
                else:
                    v = self.rec(np_, rpos, m, c, 1, 1, alpha, beta)
                if v > best_v:
                    best_v, best_m = v, a
                alpha = max(alpha, best_v)
            return best_v, best_m
        best_v, best_m = INF, None
        moves = self.order_rival(self.g.rival_moves(rpos, self.root), apos)
        for rp in moves:
            self.nodes += 1
            if rp == apos:
                v = LOSS + 1
            else:
                v = self.rec(apos, rp, mask, 0.0, 1, 0, alpha, beta)
            if v < best_v:
                best_v, best_m = v, rp
            beta = min(beta, best_v)
        return best_v, best_m

    def order_agent(self, moves, mask):
        if not self.ordering:
            return moves
        g = self.g

        def key(mv):
            a, p, c = mv
            return c + g.h_cost(p, g.update_mask(mask, p))

        return sorted(moves, key=key)

    def order_rival(self, moves, apos):
        if not self.ordering:
            return moves
        return sorted(moves, key=lambda p: abs(p[0] - apos[0]) + abs(p[1] - apos[1]))

    def rec(self, apos, rpos, mask, g, ply, turn, alpha, beta):
        self.nodes += 1
        if ply >= self.depth:
            return self.eval(apos, rpos, mask, g)
        G = self.g
        if turn == 0:
            moves = G.agent_moves(apos, self.root)
            if not moves:
                return self.eval(apos, rpos, mask, g)
            moves = self.order_agent(moves, mask)
            best = -INF
            for a, np_, c in moves:
                m = G.update_mask(mask, np_)
                if np_ == rpos:
                    self.nodes += 1
                    v = LOSS + ply + 1
                elif m == G.full:
                    self.nodes += 1
                    v = WIN - ply - 1
                else:
                    v = self.rec(np_, rpos, m, g + c, ply + 1, 1, alpha, beta)
                if v > best:
                    best = v
                if self.pruning:
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        break
            return best
        moves = G.rival_moves(rpos, self.root)
        if not moves:
            return self.eval(apos, rpos, mask, g)
        moves = self.order_rival(moves, apos)
        best = INF
        for rp in moves:
            if rp == apos:
                self.nodes += 1
                v = LOSS + ply + 1
            else:
                v = self.rec(apos, rp, mask, g, ply + 1, 0, alpha, beta)
            if v < best:
                best = v
            if self.pruning:
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
        return best
