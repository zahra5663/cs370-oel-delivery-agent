import math
import numpy as np
import config
from world import World
from search import DeliveryProblem, bfs, ucs, greedy, astar
from heuristics import Heuristics
from simulator import start_component

N_SCENARIOS = 30


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


def make_scenarios(world, n=N_SCENARIOS):
    rng = np.random.default_rng(config.seed_from_ids("|scenarios"))
    cells = sorted(start_component(world))
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 5))
        idx = rng.choice(len(cells), 1 + k, replace=False)
        pts = [cells[int(i)] for i in idx]
        out.append((pts[0], pts[1:]))
    return out


def apply_scenario(world, scenario):
    world.start = scenario[0]
    world.stops = list(scenario[1])


def ablation_variants(H):
    def no_mst(s):
        pos, mask = s
        rem = H.remaining(mask)
        return float(min(H.d_to[i][pos] for i in rem)) if rem else 0.0

    def mst_only(s):
        return H.mst_cache[s[1]]

    def no_exact_dist(s):
        pos, mask = s
        rem = H.remaining(mask)
        if not rem:
            return 0.0
        d = min(abs(pos[0] - H.stops[i][0]) + abs(pos[1] - H.stops[i][1]) for i in rem)
        return H.min_cost * d + H.mst_cache[mask]

    return {
        "full (nearest exact + MST)": H.h_own,
        "without MST term": no_mst,
        "without nearest-stop term": mst_only,
        "Manhattan instead of exact dist": no_exact_dist,
    }


def run_experiments(progress=print):
    base = World()
    scenarios = make_scenarios(base)
    rows = {}
    ablation = {}
    failures = {"suboptimal_astar": {}, "greedy_gap": [], "greedy_suboptimal": 0, "unsolved": 0}
    tunnel_use = 0
    optimal = []
    for si, sc in enumerate(scenarios):
        w = World()
        apply_scenario(w, sc)
        p = DeliveryProblem(w)
        H = Heuristics(w)
        ref = ucs(p)
        if ref["status"] != "ok":
            failures["unsolved"] += 1
            continue
        optimal.append(ref["cost"])
        if "T" in ref["actions"]:
            tunnel_use += 1
        configs = {"BFS": lambda: bfs(p), "UCS": lambda: ucs(p),
                   "Greedy + own h": lambda: greedy(p, H.h_own)}
        for name, h in H.table().items():
            if name != "zero (UCS)":
                configs["A* + " + name] = (lambda h=h: astar(p, h))
        for name, fn in configs.items():
            r = fn()
            if r["status"] != "ok":
                continue
            d = rows.setdefault(name, {"cost": [], "expanded": [], "time_ms": [], "peak_kb": [], "gap": []})
            d["cost"].append(r["cost"])
            d["expanded"].append(r["expanded"])
            d["time_ms"].append(r["time_s"] * 1000)
            d["peak_kb"].append(r["peak_kb"])
            d["gap"].append(100.0 * (r["cost"] - ref["cost"]) / ref["cost"])
            if name.startswith("A* + ") and r["cost"] > ref["cost"] + 1e-9:
                failures["suboptimal_astar"][name] = failures["suboptimal_astar"].get(name, 0) + 1
            if name == "Greedy + own h" and r["cost"] > ref["cost"] + 1e-9:
                failures["greedy_suboptimal"] += 1
                failures["greedy_gap"].append(100.0 * (r["cost"] - ref["cost"]) / ref["cost"])
        for name, h in ablation_variants(H).items():
            r = astar(p, h)
            if r["status"] == "ok":
                a = ablation.setdefault(name, {"expanded": [], "time_ms": [], "gap": []})
                a["expanded"].append(r["expanded"])
                a["time_ms"].append(r["time_s"] * 1000)
                a["gap"].append(100.0 * (r["cost"] - ref["cost"]) / ref["cost"])
        if (si + 1) % 10 == 0:
            progress(f"  scenarios done: {si + 1}/{len(scenarios)}")
    return {"rows": rows, "ablation": ablation, "failures": failures, "n": len(optimal),
            "optimal_mean": mean_ci(optimal), "tunnel_use": tunnel_use}


def print_experiments(res):
    print(config.STAMP)
    print(f"\nTABLE E1: SEARCH ON {res['n']} SEEDED SCENARIOS (random start + 3-4 stops on the instance grid), mean +/- 95% CI")
    print(f"{'configuration':<30}{'cost':>16}{'expanded':>18}{'time (ms)':>16}{'peak (KB)':>14}{'gap %':>10}")
    for name, d in res["rows"].items():
        c, e, t, m, g = (mean_ci(d[k]) for k in ("cost", "expanded", "time_ms", "peak_kb", "gap"))
        print(f"{name:<30}{c[0]:>9.1f}+/-{c[1]:<5.1f}{e[0]:>11.0f}+/-{e[1]:<6.0f}{t[0]:>9.1f}+/-{t[1]:<5.1f}"
              f"{m[0]:>8.0f}+/-{m[1]:<4.0f}{g[0]:>8.2f}")
    print(f"optimal plan used a tunnel in {res['tunnel_use']} of {res['n']} scenarios")
    print(config.STAMP)
    print("\nTABLE E2: ABLATION OF THE OWN HEURISTIC (A*, same scenarios)")
    print(f"{'variant':<34}{'expanded':>18}{'time (ms)':>16}{'gap %':>10}")
    for name, d in res["ablation"].items():
        e, t, g = mean_ci(d["expanded"]), mean_ci(d["time_ms"]), mean_ci(d["gap"])
        print(f"{name:<34}{e[0]:>9.0f}+/-{e[1]:<6.0f}{t[0]:>9.1f}+/-{t[1]:<5.1f}{g[0]:>8.2f}")
    print("note: 'without nearest-stop term' is admissible but inconsistent, and our A* does not reopen closed states,")
    print("      so it can return a slightly worse plan. The Manhattan variants are also inadmissible because tunnels shortcut distance.")
    print(config.STAMP)
    f = res["failures"]
    print("\nFAILURE ANALYSIS")
    n = res["n"]
    print(f"- scenarios unsolved by UCS: {f['unsolved']}")
    for name, k in f["suboptimal_astar"].items():
        print(f"- {name} returned a worse-than-optimal plan in {k}/{n} scenarios (Manhattan-based bound is broken by tunnels; A* does not reopen closed states)")
    if not f["suboptimal_astar"]:
        print("- every A* variant returned an optimal plan in all scenarios")
    if f["greedy_gap"]:
        print(f"- Greedy was suboptimal in {f['greedy_suboptimal']}/{n} scenarios, mean gap {np.mean(f['greedy_gap']):.1f}% on those")
    print(config.STAMP)
