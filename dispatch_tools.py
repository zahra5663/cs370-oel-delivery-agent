import numpy as np
import config
from oel_instance import TERRAINS
from world import World, MOVES
from heuristics import Heuristics
from search import DeliveryProblem, bfs, ucs, greedy, astar, ids

ALGORITHMS = ["bfs", "ucs", "greedy", "astar", "ids"]


class ConstrainedProblem(DeliveryProblem):
    def __init__(self, world, first_stop=None, avoid=None):
        super().__init__(world)
        self.first = None if first_stop is None else first_stop - 1
        self.avoid = {TERRAINS.index(t) for t in (avoid or [])}
        self.init = None

    def initial(self):
        return self.init if self.init is not None else super().initial()

    def successors(self, state):
        pos, mask = state
        for action, nxt, cost in self.w.neighbors(pos):
            if action != "T" and nxt not in self.stop_idx and int(self.w.grid[nxt[0], nxt[1]]) in self.avoid:
                continue
            m = mask
            if nxt in self.stop_idx:
                i = self.stop_idx[nxt]
                if self.first is not None and not (mask >> self.first) & 1 and i != self.first:
                    continue
                m |= 1 << i
            yield action, (nxt, m), cost


class Dispatcher:
    def __init__(self, validate=True, seed=0):
        self.world = World()
        self.H = Heuristics(self.world)
        self.validate = validate
        self.plans = {}
        self.next_id = 1
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def describe(self):
        w = self.world
        return {
            "num_stops": len(w.stops),
            "terrains": TERRAINS,
            "terrain_cost": w.inst["terrain_cost"],
            "tunnel_pairs": len(w.tunnels),
            "wind_direction": w.wind_dir,
            "wind_probability": w.wind_p,
            "slip_probability": w.slip_p,
        }

    def check(self, action, args):
        if action not in ("plan_route", "simulate", "wind_risk"):
            return None, f"unknown tool '{action}'. Tools: plan_route, simulate, wind_risk"
        if not isinstance(args, dict):
            return None, "args must be a JSON object"
        k = len(self.world.stops)
        if action == "plan_route":
            allowed = {"algorithm", "first_stop", "avoid_terrain", "max_cost"}
            extra = set(args) - allowed
            if extra:
                return None, f"unknown arguments {sorted(extra)}. Allowed: {sorted(allowed)}"
            out = {"algorithm": args.get("algorithm", "astar"), "first_stop": args.get("first_stop"),
                   "avoid_terrain": args.get("avoid_terrain") or [], "max_cost": args.get("max_cost")}
            if out["algorithm"] not in ALGORITHMS:
                return None, f"algorithm must be one of {ALGORITHMS}"
            fs = out["first_stop"]
            if fs is not None and (isinstance(fs, bool) or not isinstance(fs, int) or not 1 <= fs <= k):
                return None, f"first_stop must be an integer from 1 to {k} (stops are numbered from 1)"
            if not isinstance(out["avoid_terrain"], list) or any(t not in TERRAINS for t in out["avoid_terrain"]):
                return None, f"avoid_terrain must be a list drawn from {TERRAINS}"
            mc = out["max_cost"]
            if mc is not None and (isinstance(mc, bool) or not isinstance(mc, (int, float)) or mc <= 0):
                return None, "max_cost must be a positive number"
            return out, None
        extra = set(args) - {"plan_id"}
        if extra:
            return None, f"unknown arguments {sorted(extra)}. Allowed: ['plan_id']"
        pid = args.get("plan_id")
        if pid not in self.plans:
            return None, f"plan_id {pid} does not exist. Existing plan ids: {sorted(self.plans)}"
        return {"plan_id": pid}, None

    def call(self, action, args):
        cleaned, err = self.check(action, args)
        self.calls.append({"action": action, "args": args, "valid": err is None})
        if self.validate:
            if err:
                return {"error": err}
            return self.run_tool(action, cleaned)
        try:
            return self.run_tool(action, self.loose(action, args))
        except Exception as e:
            return {"error": f"tool crashed: {type(e).__name__}"}

    def loose(self, action, args):
        args = dict(args) if isinstance(args, dict) else {}
        if action == "plan_route":
            terr = [t for t in (args.get("avoid_terrain") or []) if t in TERRAINS]
            return {"algorithm": args.get("algorithm") if args.get("algorithm") in ALGORITHMS else "astar",
                    "first_stop": args.get("first_stop"), "avoid_terrain": terr,
                    "max_cost": args.get("max_cost")}
        return args

    def run_tool(self, action, a):
        if action == "plan_route":
            return self.plan_route(**a)
        if action == "simulate":
            return self.simulate(**a)
        if action == "wind_risk":
            return self.wind_risk(**a)
        return {"error": f"unknown tool {action}"}

    def solve(self, algorithm, first_stop, avoid):
        p = ConstrainedProblem(self.world, first_stop, avoid)
        if algorithm == "bfs":
            return p, bfs(p)
        if algorithm == "ucs":
            return p, ucs(p)
        if algorithm == "greedy":
            return p, greedy(p, self.H.h_own)
        if algorithm == "ids":
            return p, ids(p, max_depth=60, node_limit=50000)
        return p, astar(p, self.H.h_own)

    def plan_route(self, algorithm="astar", first_stop=None, avoid_terrain=None, max_cost=None):
        p, r = self.solve(algorithm, first_stop, avoid_terrain or [])
        if r["status"] != "ok":
            return {"status": r["status"], "message": "no plan found"}
        pid = self.next_id
        self.next_id += 1
        within = True if max_cost is None else r["cost"] <= max_cost
        self.plans[pid] = {"actions": r["actions"], "cost": r["cost"], "algorithm": algorithm,
                           "first_stop": first_stop, "avoid": list(avoid_terrain or []),
                           "max_cost": max_cost, "within_budget": within}
        return {"status": "ok", "plan_id": pid, "cost": r["cost"], "steps": r["steps"],
                "within_budget": within, "nodes_expanded": r["expanded"]}

    def simulate(self, plan_id):
        plan = self.plans[plan_id]
        w = self.world
        p = ConstrainedProblem(w, plan["first_stop"], plan["avoid"])
        state = p.initial()
        actions = list(plan["actions"])
        total, replans, steps = 0.0, 0, 0
        while not p.is_goal(state) and steps < 300:
            if not actions:
                p.init = state
                r = astar(p, self.H.h_own)
                if r["status"] != "ok":
                    return {"status": "stuck", "delivered": bin(state[1]).count("1")}
                actions = list(r["actions"])
            a = actions.pop(0)
            expected = state[0]
            for act, s2, c in p.successors(state):
                if act == a:
                    expected = s2[0]
            new, c, _ = w.stochastic_step(state[0], a, self.rng)
            total += c
            mask = state[1]
            if new in p.stop_idx:
                mask |= 1 << p.stop_idx[new]
            if new != expected:
                replans += 1
                actions = []
            state = (new, mask)
            steps += 1
        return {"status": "ok" if p.is_goal(state) else "stuck", "delivered": bin(state[1]).count("1"),
                "actual_cost": round(total, 1), "planned_cost": plan["cost"], "replans": replans, "steps": steps}

    def wind_risk(self, plan_id):
        w = self.world
        plan = self.plans[plan_id]
        pos = w.start
        dr, dc = MOVES[w.wind_dir]
        exposed, expected = 0, 0.0
        for a in plan["actions"]:
            nxt = pos
            for act, p2, c in w.neighbors(pos):
                if act == a:
                    nxt = p2
            pos = nxt
            q = (pos[0] + dr, pos[1] + dc)
            if w.passable(*q):
                exposed += 1
                expected += w.wind_p * float(w.cost_map[q[0], q[1]])
        return {"wind_direction": w.wind_dir, "steps_exposed_to_drift": exposed,
                "expected_extra_cost": round(expected, 1), "planned_cost": plan["cost"]}
