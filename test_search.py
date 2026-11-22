import config
from world import World
from search import DeliveryProblem, bfs, ucs, dls, ids, greedy, astar
from heuristics import Heuristics, check_consistency


def show(rows, title):
    print()
    print(title)
    print(f"{'algorithm':<26}{'status':<14}{'cost':>8}{'steps':>7}{'expanded':>10}{'stored':>8}{'time(ms)':>10}{'peak(KB)':>10}")
    for r in rows:
        cost = "-" if r["cost"] is None else f"{r['cost']:.0f}"
        steps = "-" if r["steps"] is None else str(r["steps"])
        print(f"{r['algorithm']:<26}{r['status']:<14}{cost:>8}{steps:>7}{r['expanded']:>10}{r['stored_peak']:>8}{r['time_s']*1000:>10.1f}{r['peak_kb']:>10.0f}")
    print(config.STAMP)


def main():
    w = World()
    p = DeliveryProblem(w)
    H = Heuristics(w)
    print(config.STAMP)
    print("instance", w.inst["instance_id"], "start", w.start, "stops", w.stops)

    rows = [bfs(p), ucs(p), greedy(p, H.h_own), astar(p, H.h_own),
            ids(p, max_depth=60, node_limit=300000), dls(p, limit=40, node_limit=300000)]
    show(rows, "ALL ALGORITHMS (own heuristic for Greedy and A*)")

    rows = []
    for name, h in H.table().items():
        r = astar(p, h)
        r["algorithm"] = "A* + " + name
        rows.append(r)
    r = astar(p, H.inflated(1.5))
    r["algorithm"] = "A* + own x1.5 (inadm)"
    rows.append(r)
    r = astar(p, lambda st: 5 * H.h_manhattan_nearest(st))
    r["algorithm"] = "A* + manhattan x5 (inadm)"
    rows.append(r)
    show(rows, "A* HEURISTIC COMPARISON")
    opt = ucs(p)["cost"]
    print(f"INADMISSIBLE VARIANT: manhattan x5 returns cost {r['cost']:.0f} vs optimal {opt:.0f} "
          f"(gap {100*(r['cost']-opt)/opt:.1f}%), expanded {r['expanded']} nodes vs {rows[0]['expanded']} for own heuristic")
    print(config.STAMP)

    print()
    print("CONSISTENCY CHECK (every edge of the whole state space)")
    for name, h in list(H.table().items())[:3]:
        print(name, check_consistency(p, h))
    print(config.STAMP)

    print()
    print("MEMORY LIMIT: A* vs IDS with max 500 stored states")
    rows = [astar(p, H.h_zero, max_stored=500), astar(p, H.h_own, max_stored=500), ids(p, max_depth=60, node_limit=300000, max_stored=500)]
    rows[0]["algorithm"] = "A* zero-h (UCS-like)"
    rows[1]["algorithm"] = "A* own heuristic"
    show(rows, "MEMORY LIMITED")

    print()
    print("SMALL PROBLEM WHERE IDS CAN FINISH: only stop 1, at most 40 stored states")
    w1 = World()
    w1.stops = [w1.stops[0]]
    p1 = DeliveryProblem(w1)
    H1 = Heuristics(w1)
    rows = [ucs(p1, max_stored=40), astar(p1, H1.h_zero, max_stored=40), astar(p1, H1.h_own, max_stored=40),
            ids(p1, max_depth=30, node_limit=300000, max_stored=40)]
    rows[0]["algorithm"] = "UCS"
    rows[1]["algorithm"] = "A* zero-h"
    rows[2]["algorithm"] = "A* own heuristic"
    show(rows, "IDS UNDER A MEMORY LIMIT (single stop)")
    free = [ucs(p1), ids(p1, max_depth=30, node_limit=300000)]
    if free[1]["cost"] is None:
        print(f"without a limit: UCS stores {free[0]['stored_peak']} states; IDS did not finish within its node budget on this problem")
    else:
        print(f"without a limit: UCS stores {free[0]['stored_peak']} states (cost {free[0]['cost']:.0f}), "
              f"IDS stores {free[1]['stored_peak']} states and returns cost {free[1]['cost']:.0f}: IDS is optimal in steps, not in cost")
    print(config.STAMP)


if __name__ == "__main__":
    main()
