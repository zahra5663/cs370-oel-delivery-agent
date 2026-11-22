import heapq
import itertools
import time
import tracemalloc
from collections import deque


class DeliveryProblem:
    def __init__(self, world):
        self.w = world
        self.stops = world.stops
        self.stop_idx = {s: i for i, s in enumerate(world.stops)}
        self.full = (1 << len(world.stops)) - 1

    def initial(self):
        mask = 0
        if self.w.start in self.stop_idx:
            mask |= 1 << self.stop_idx[self.w.start]
        return (self.w.start, mask)

    def is_goal(self, state):
        return state[1] == self.full

    def successors(self, state):
        pos, mask = state
        for action, nxt, cost in self.w.neighbors(pos):
            m = mask
            if nxt in self.stop_idx:
                m |= 1 << self.stop_idx[nxt]
            yield action, (nxt, m), cost


class Limit(Exception):
    def __init__(self, kind):
        self.kind = kind


class Counter:
    def __init__(self, node_limit, max_stored):
        self.expanded = 0
        self.generated = 0
        self.max_frontier = 0
        self.node_limit = node_limit
        self.max_stored = max_stored

    def expand(self):
        self.expanded += 1
        if self.node_limit and self.expanded > self.node_limit:
            raise Limit("node_limit")

    def stored(self, n):
        if n > self.max_stored_seen:
            self.max_stored_seen = n
        if self.max_stored and n > self.max_stored:
            raise Limit("memory_limit")

    max_stored_seen = 0


def rebuild(parent, state):
    actions = []
    states = [state]
    while parent[state] is not None:
        prev, act = parent[state]
        actions.append(act)
        states.append(prev)
        state = prev
    actions.reverse()
    states.reverse()
    return actions, states


def path_cost(problem, states):
    total = 0.0
    for a, b in zip(states, states[1:]):
        for _, s2, c in problem.successors(a):
            if s2 == b:
                total += c
                break
    return total


def finish(name, problem, counter, status, states, actions, t0, peak):
    cost = path_cost(problem, states) if states else None
    return {
        "algorithm": name, "status": status, "cost": cost,
        "steps": len(actions) if actions else None, "actions": actions,
        "expanded": counter.expanded, "generated": counter.generated,
        "max_frontier": counter.max_frontier, "stored_peak": counter.max_stored_seen,
        "time_s": time.perf_counter() - t0, "peak_kb": peak / 1024.0,
    }


def run(name, fn, problem, h=None, node_limit=None, max_stored=None, **kw):
    counter = Counter(node_limit, max_stored)
    counter.max_stored_seen = 0
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        status, states, actions = fn(problem, counter, h, **kw)
    except Limit as e:
        status, states, actions = e.kind, None, None
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return finish(name, problem, counter, status, states, actions, t0, peak)


def _bfs(problem, c, h):
    s0 = problem.initial()
    if problem.is_goal(s0):
        return "ok", [s0], []
    parent = {s0: None}
    q = deque([s0])
    while q:
        s = q.popleft()
        c.expand()
        for a, s2, cost in problem.successors(s):
            c.generated += 1
            if s2 in parent:
                continue
            parent[s2] = (s, a)
            c.stored(len(parent))
            if problem.is_goal(s2):
                acts, sts = rebuild(parent, s2)
                return "ok", sts, acts
            q.append(s2)
        c.max_frontier = max(c.max_frontier, len(q))
    return "fail", None, None


def _best_first(problem, c, h, mode):
    s0 = problem.initial()
    tie = itertools.count()
    g = {s0: 0.0}
    parent = {s0: None}

    def key(state, gv):
        if mode == "ucs":
            return gv
        if mode == "greedy":
            return h(state)
        return gv + h(state)

    pq = [(key(s0, 0.0), next(tie), s0)]
    closed = set()
    while pq:
        _, _, s = heapq.heappop(pq)
        if s in closed:
            continue
        if problem.is_goal(s):
            acts, sts = rebuild(parent, s)
            return "ok", sts, acts
        closed.add(s)
        c.expand()
        for a, s2, cost in problem.successors(s):
            c.generated += 1
            ng = g[s] + cost
            if mode == "greedy":
                if s2 in g:
                    continue
            elif s2 in g and ng >= g[s2]:
                continue
            g[s2] = ng
            parent[s2] = (s, a)
            c.stored(len(g))
            heapq.heappush(pq, (key(s2, ng), next(tie), s2))
        c.max_frontier = max(c.max_frontier, len(pq))
    return "fail", None, None


def _ucs(problem, c, h):
    return _best_first(problem, c, h, "ucs")


def _greedy(problem, c, h):
    return _best_first(problem, c, h, "greedy")


def _astar(problem, c, h):
    return _best_first(problem, c, h, "astar")


def _dls_core(problem, c, limit):
    s0 = problem.initial()
    on_path = {s0}
    actions = []
    states = [s0]

    def rec(s, depth):
        if problem.is_goal(s):
            return "found"
        if depth == limit:
            return "cutoff"
        c.expand()
        cutoff = False
        for a, s2, cost in problem.successors(s):
            c.generated += 1
            if s2 in on_path:
                continue
            on_path.add(s2)
            actions.append(a)
            states.append(s2)
            c.stored(len(on_path))
            r = rec(s2, depth + 1)
            if r == "found":
                return "found"
            if r == "cutoff":
                cutoff = True
            on_path.discard(s2)
            actions.pop()
            states.pop()
        return "cutoff" if cutoff else "fail"

    r = rec(s0, 0)
    return r, list(states), list(actions)


def _dls(problem, c, h, limit=30):
    r, states, actions = _dls_core(problem, c, limit)
    if r == "found":
        return "ok", states, actions
    return ("cutoff" if r == "cutoff" else "fail"), None, None


def _ids(problem, c, h, max_depth=60):
    for limit in range(max_depth + 1):
        r, states, actions = _dls_core(problem, c, limit)
        if r == "found":
            return "ok", states, actions
        if r == "fail":
            return "fail", None, None
    return "depth_exhausted", None, None


def bfs(problem, **kw):
    return run("BFS", _bfs, problem, **kw)


def ucs(problem, **kw):
    return run("UCS", _ucs, problem, **kw)


def dls(problem, limit=30, **kw):
    return run(f"DLS(l={limit})", _dls, problem, limit=limit, **kw)


def ids(problem, max_depth=60, **kw):
    return run("IDS", _ids, problem, max_depth=max_depth, **kw)


def greedy(problem, h, **kw):
    return run("Greedy", _greedy, problem, h=h, **kw)


def astar(problem, h, **kw):
    return run("A*", _astar, problem, h=h, **kw)
