import json, math, os
import numpy as np
import config
from world import World
from heuristics import Heuristics
from game import Game, Search
from simulator import run_game, rival_start_for

RIVAL_SPEED = None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.mean()
    h = 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, h


def tally(rs):
    return {k: sum(1 for r in rs if r["result"] == k) for k in ("win", "draw", "loss")}


def play(world, game, indices, seed_base, **kw):
    return [run_game(world, game, rival_start_for(world, i), seed_base + i,
                     rival_speed=RIVAL_SPEED, **kw) for i in indices]


def node_counts(world, game, n_scen=30):
    S = Search(game)
    rows = []
    for depth in range(2, world.inst["adversary_search_depth"] + 1):
        mm, ab, abo, per_ord, per_plain = [], [], [], [], []
        for i in range(n_scen):
            rs = rival_start_for(world, 200 + i)
            S.run(world.start, rs, 0, depth, "A", "agent", False, False)
            mm.append(S.nodes)
            S.run(world.start, rs, 0, depth, "A", "agent", True, False)
            ab.append(S.nodes)
            S.run(world.start, rs, 0, depth, "A", "agent", True, True)
            abo.append(S.nodes)
            per_plain.append(100 * (1 - ab[-1] / mm[-1]))
            per_ord.append(100 * (1 - abo[-1] / mm[-1]))
        rows.append({"ordered_ci": mean_ci(per_ord)[1], "plain_ci": mean_ci(per_plain)[1], "ordered_mean": mean_ci(per_ord)[0],
                     "plain_mean": mean_ci(per_plain)[0], **{"depth": depth, "minimax": float(np.mean(mm)), "ab_plain": float(np.mean(ab)),
                     "ab_ordered": float(np.mean(abo)),
                     "saved_plain_pct": 100 * (1 - np.mean(ab) / np.mean(mm)),
                     "saved_ordered_pct": 100 * (1 - np.mean(abo) / np.mean(mm))}})
    return rows


def main():
    global RIVAL_SPEED
    world = World()
    RIVAL_SPEED = world.params["rival_speed"]
    H = Heuristics(world)
    game = Game(world, H, window=world.params["game_window"])
    out = {}
    print(config.STAMP)
    print("instance", world.inst["instance_id"], "| rival", world.inst["adversary"],
          "depth", world.inst["adversary_search_depth"], "| slip", world.slip_p,
          "| wind", world.wind_p, world.wind_dir, "| rival speed", RIVAL_SPEED)

    print("\nTABLE 1: NODE COUNTS (mean of 30 scenarios, agent to move; saved % = mean over scenarios +/- 95% CI)")
    rows = node_counts(world, game)
    print(f"{'depth':>5}{'minimax':>10}{'AB plain':>10}{'AB ordered':>12}{'saved plain %':>20}{'saved ordered %':>20}")
    for r in rows:
        print(f"{r['depth']:>5}{r['minimax']:>10.0f}{r['ab_plain']:>10.0f}{r['ab_ordered']:>12.0f}"
              f"{r['plain_mean']:>12.1f} +/-{r['plain_ci']:<4.1f}{r['ordered_mean']:>12.1f} +/-{r['ordered_ci']:<4.1f}")
    print(config.STAMP)
    out["node_counts"] = rows

    print("\nTABLE 2: WEIGHT TUNING (grid search, 20 games each, tuning seeds)")
    grid = [(c, s, m) for c in (1.0, 2.0) for s in (0, 3, 6, 10) for m in (0, 3, 6)]
    results = []
    for c, s, m in grid:
        game.set_weights({"cost": c, "safety": s, "margin": m})
        rs = play(world, game, range(100, 120), 9000)
        t = tally(rs)
        results.append(((c, s, m), t))
    results.sort(key=lambda x: (-x[1]["win"], x[1]["loss"]))
    print(f"{'cost':>5}{'safety':>8}{'margin':>8}{'win':>5}{'draw':>6}{'loss':>6}")
    for (c, s, m), t in results[:10]:
        print(f"{c:>5}{s:>8}{m:>8}{t['win']:>5}{t['draw']:>6}{t['loss']:>6}")
    best = results[0][0]
    print("best weights: cost", best[0], "safety", best[1], "margin", best[2], "(delivered = 100)")
    print(config.STAMP)
    out["tuning"] = [{"cost": c, "safety": s, "margin": m, **t} for (c, s, m), t in results]
    game.set_weights({"cost": best[0], "safety": best[1], "margin": best[2]})

    N = 30
    stoch = play(world, game, range(N), 5000)
    det = [run_game(world, game, rival_start_for(world, i), 5000 + i, slip_p=0.0, wind_p=0.0,
                    rival_speed=RIVAL_SPEED) for i in range(N)]

    print(f"\nTABLE 3: {N} GAMES vs {world.inst['adversary']} rival (win / draw / loss with 95% Wilson CI)")
    for name, rs in (("with slip + wind", stoch), ("deterministic (no slip/wind)", det)):
        t = tally(rs)
        cells = []
        for k in ("win", "draw", "loss"):
            lo, hi = wilson(t[k], N); lo = max(0.0, lo)
            cells.append(f"{k} {100*t[k]/N:.0f}% [{100*lo:.0f}, {100*hi:.0f}]")
        print(f"{name:<30}" + "   ".join(cells))
    print(config.STAMP)
    out["games"] = {"stochastic": tally(stoch), "deterministic": tally(det)}

    dev = [r["deviations"] for r in stoch]
    turns = [r["turns"] for r in stoch]
    dm, dh = mean_ci(dev)
    rate = sum(dev) / sum(turns)
    diff = [a["cost"] - b["cost"] for a, b in zip(stoch, det)]
    xm, xh = mean_ci(diff)
    cm, ch = mean_ci([r["cost"] for r in stoch])
    print("\nTABLE 4: REPLANNING DUE TO SLIP / WIND (deterministic assumption)")
    print(f"replans triggered per game: {dm:.1f} +/- {dh:.1f}   ({100*rate:.0f}% of all turns)")
    print(f"slips per game: {np.mean([r['slips'] for r in stoch]):.1f}   wind pushes per game: {np.mean([r['winds'] for r in stoch]):.1f}")
    print(f"mean cost per game (stochastic): {cm:.1f} +/- {ch:.1f}")
    print(f"extra cost vs deterministic (paired): {xm:.1f} +/- {xh:.1f}")
    print(f"mean turns: stochastic {np.mean(turns):.1f}, deterministic {np.mean([r['turns'] for r in det]):.1f}")
    print(config.STAMP)
    out["replanning"] = {"replans_per_game": [dm, dh], "turn_rate": rate, "extra_cost": [xm, xh]}

    print("\nTABLE 5: SENSITIVITY TO SLIP PROBABILITY (30 games each, wind 0.15, rival depth 5)")
    print(f"{'slip':>6}{'win %':>10}{'CI':>14}{'draw %':>9}{'loss %':>9}{'replans/game':>14}{'mean cost':>11}")
    sens = {"slip": [], "depth": []}
    for sp in (0.0, 0.1, 0.2, 0.3, 0.5):
        rs = [run_game(world, game, rival_start_for(world, i), 5000 + i, slip_p=sp, rival_speed=RIVAL_SPEED) for i in range(N)]
        t = tally(rs)
        lo, hi = wilson(t["win"], N)
        rp = float(np.mean([r["deviations"] for r in rs]))
        mc = float(np.mean([r["cost"] for r in rs]))
        sens["slip"].append({"slip": sp, **t, "ci": [max(0.0, lo), hi], "replans": rp, "cost": mc})
        print(f"{sp:>6}{100*t['win']/N:>10.0f}{'['+str(round(100*max(0.0, lo)))+', '+str(round(100*hi))+']':>14}"
              f"{100*t['draw']/N:>9.0f}{100*t['loss']/N:>9.0f}{rp:>14.1f}{mc:>11.1f}")
    print(config.STAMP)
    print("\nTABLE 6: SENSITIVITY TO RIVAL SEARCH DEPTH (30 games each, slip 0.2, wind 0.15)")
    print(f"{'depth':>6}{'win %':>10}{'CI':>14}{'draw %':>9}{'loss %':>9}")
    for d in (3, 4, 5, 6):
        rs = [run_game(world, game, rival_start_for(world, i), 5000 + i, rival_depth=d, rival_speed=RIVAL_SPEED) for i in range(N)]
        t = tally(rs)
        lo, hi = wilson(t["win"], N)
        sens["depth"].append({"depth": d, **t, "ci": [max(0.0, lo), hi]})
        print(f"{d:>6}{100*t['win']/N:>10.0f}{'['+str(round(100*max(0.0, lo)))+', '+str(round(100*hi))+']':>14}"
              f"{100*t['draw']/N:>9.0f}{100*t['loss']/N:>9.0f}")
    print(config.STAMP)
    out["sensitivity"] = sens

    os.makedirs("results", exist_ok=True)
    with open("results/part_c.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
