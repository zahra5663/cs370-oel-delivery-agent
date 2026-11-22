import json
import os
import sys
import time
import config
from world import World
from search import DeliveryProblem, astar
from heuristics import Heuristics
from game import Game
from simulator import run_game, rival_start_for
from viz import draw_grid, bar_with_ci, stamp_figure
import experiments
import test_search
import run_games


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def part_a():
    w = World()
    print(config.STAMP)
    print("\nPART A: WORLD MODEL")
    print("instance", w.inst["instance_id"], "| grid", w.n, "x", w.n, "| start", w.start, "| stops", w.stops)
    print("twists:", w.inst["twists"], "| rival:", w.inst["adversary"], "depth", w.inst["adversary_search_depth"],
          "| slip", w.slip_p, "| wind", w.wind_dir, w.wind_p)
    print("tunnels:", w.tunnels)
    b = w.benchmark_cost_map(repeats=50)
    print(f"cost map build: numpy {b['numpy_s']*1e6:.1f} us | pure python loop {b['loop_s']*1e6:.1f} us | speed-up {b['speedup']:.1f}x")
    print(config.STAMP)


def figures(exp):
    os.makedirs("figures", exist_ok=True)
    w = World()
    p = DeliveryProblem(w)
    H = Heuristics(w)
    r = astar(p, H.h_own)
    fig = draw_grid(w, r["actions"], title=f"A* route, cost {r['cost']:.0f} (instance {w.inst['instance_id']})")
    fig.savefig("figures/fig1_route.png", dpi=150)
    labels = list(exp["rows"])
    means = [sum(exp["rows"][k]["expanded"]) / len(exp["rows"][k]["expanded"]) for k in labels]
    cis = [experiments.mean_ci(exp["rows"][k]["expanded"])[1] for k in labels]
    bar_with_ci(labels, means, cis, "Nodes expanded (mean, 95% CI, 30 scenarios)", "nodes expanded", "figures/fig2_nodes.png")
    labels = list(exp["ablation"])
    means = [sum(exp["ablation"][k]["expanded"]) / len(exp["ablation"][k]["expanded"]) for k in labels]
    cis = [experiments.mean_ci(exp["ablation"][k]["expanded"])[1] for k in labels]
    bar_with_ci(labels, means, cis, "Ablation of own heuristic: nodes expanded", "nodes expanded", "figures/fig3_ablation.png")
    if os.path.exists("results/part_c.json"):
        import matplotlib.pyplot as plt
        with open("results/part_c.json") as f:
            c = json.load(f)
        nc = c["node_counts"]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        xs = range(len(nc))
        ax.bar([x - 0.27 for x in xs], [r_["minimax"] for r_ in nc], 0.27, label="Minimax", color="#8e24aa")
        ax.bar(list(xs), [r_["ab_plain"] for r_ in nc], 0.27, label="Alpha-Beta", color="#00897b")
        ax.bar([x + 0.27 for x in xs], [r_["ab_ordered"] for r_ in nc], 0.27, label="Alpha-Beta + own ordering", color="#fb8c00")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"depth {r_['depth']}" for r_ in nc])
        ax.set_ylabel("nodes")
        ax.set_title("Game search nodes by depth", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        stamp_figure(fig)
        fig.savefig("figures/fig4_game_nodes.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        names = ["with slip + wind", "deterministic"]
        for i, key in enumerate(("stochastic", "deterministic")):
            t = c["games"][key]
            tot = sum(t.values())
            bottom = 0
            for k, col in (("win", "#43a047"), ("draw", "#fdd835"), ("loss", "#e53935")):
                ax.bar(i, 100 * t[k] / tot, bottom=bottom, color=col, label=k if i == 0 else None)
                bottom += 100 * t[k] / tot
        ax.set_xticks([0, 1])
        ax.set_xticklabels(names)
        ax.set_ylabel("% of games")
        ax.set_title("Results vs interceptor rival (30 games)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        stamp_figure(fig)
        fig.savefig("figures/fig5_win_draw_loss.png", dpi=150)
        plt.close(fig)
        sens = c.get("sensitivity")
        if sens:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            xs = [r_["slip"] for r_ in sens["slip"]]
            ys = [100 * r_["win"] / 30 for r_ in sens["slip"]]
            lo = [y - 100 * r_["ci"][0] for y, r_ in zip(ys, sens["slip"])]
            hi = [100 * r_["ci"][1] - y for y, r_ in zip(ys, sens["slip"])]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#00897b", capsize=4)
            ax.set_xlabel("slip probability")
            ax.set_ylabel("win rate (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Win rate against the interceptor vs slip probability (30 games each, 95% CI)", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            stamp_figure(fig)
            fig.savefig("figures/fig7_sensitivity.png", dpi=150)
            plt.close(fig)
    w2 = World()
    H2 = Heuristics(w2)
    g = Game(w2, H2, weights={"cost": 1.0, "safety": 3.0, "margin": 6.0}, window=w2.params["game_window"])
    trace = []
    res = run_game(w2, g, rival_start_for(w2, 0), 5000, rival_speed=w2.params["rival_speed"], trace=trace)
    fig = draw_grid(w2, None, agent=trace[-1][0], rival=trace[-1][1], mask=trace[-1][2],
                    title=f"One game vs interceptor: {res['result']} after {res['turns']} turns")
    ax = fig.axes[0]
    ax.plot([t[0][1] for t in trace], [t[0][0] for t in trace], color="#00897b", lw=2, zorder=4)
    ax.plot([t[1][1] for t in trace], [t[1][0] for t in trace], color="#e53935", lw=2, ls="--", zorder=4)
    fig.savefig("figures/fig6_game.png", dpi=150)
    plt.close("all")
    print("figures saved in ./figures (all carry the member stamp)")


def part_d_summary():
    print("\nPART D: LLM DISPATCHER (summary of saved results)")
    if not os.path.exists("results/part_d.json"):
        print("results/part_d.json not found. Run: python llm_agent.py   (needs GEMINI_API_KEY), or python run_all.py --with-llm")
        print(config.STAMP)
        return
    import llm_agent
    with open("results/part_d.json") as f:
        d = json.load(f)
    llm_agent.print_tables(d["direct"], d["llm_no_validation"], d["llm_with_validation"])
    llm_agent.print_failure_log(d["llm_no_validation"], d["llm_with_validation"])


def apply_live_changes():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--slip", type=float)
    ap.add_argument("--wind", type=float)
    ap.add_argument("--rival-depth", type=int)
    ap.add_argument("--rival-speed", type=float)
    ap.add_argument("--no-tunnels", action="store_true")
    ap.add_argument("--no-wind", action="store_true")
    ap.add_argument("--start", type=str)
    ap.add_argument("--stops", type=str)
    a, _ = ap.parse_known_args()

    def cells(text):
        return [tuple(int(v) for v in part.split(",")) for part in text.split(";")]

    changes = []
    if a.slip is not None:
        config.PARAMS["slip_probability"] = a.slip
        changes.append(f"slip probability = {a.slip}")
    if a.wind is not None:
        config.PARAMS["wind_probability"] = a.wind
        changes.append(f"wind probability = {a.wind}")
    if a.no_wind:
        config.PARAMS["wind_probability"] = 0.0
        changes.append("wind drift twist OFF")
    if a.no_tunnels:
        config.PARAMS["num_tunnel_pairs"] = 0
        changes.append("teleport tunnels twist OFF")
    if a.rival_depth is not None:
        config.PARAMS["adversary_search_depth"] = a.rival_depth
        changes.append(f"rival depth = {a.rival_depth}")
    if a.rival_speed is not None:
        config.PARAMS["rival_speed"] = a.rival_speed
        changes.append(f"rival speed = {a.rival_speed}")
    if a.start:
        config.PARAMS["start"] = cells(a.start)[0]
        changes.append(f"start = {config.PARAMS['start']}")
    if a.stops:
        config.PARAMS["stops"] = cells(a.stops)
        changes.append(f"stops = {config.PARAMS['stops']}")
    if changes:
        os.makedirs("viva_run", exist_ok=True)
        os.chdir("viva_run")
    return changes


def main():
    changes = apply_live_changes()
    os.makedirs("results", exist_ok=True)
    sys.stdout = Tee("results/run_all_output.txt")
    t0 = time.time()
    print(config.STAMP)
    if changes:
        print("LIVE CHANGE APPLIED:", "; ".join(changes), "(outputs are written to ./viva_run)")
    part_a()
    print("\nPART B: ROUTE PLANNING ON THE INSTANCE")
    test_search.main()
    print("\nPART E: EXPERIMENTS")
    exp = experiments.run_experiments()
    experiments.print_experiments(exp)
    print("\nPART C: RIVAL AGENT")
    run_games.main()
    if "--with-llm" in sys.argv:
        import llm_agent
        sys.argv = [a for a in sys.argv if a != "--with-llm"]
        llm_agent.main()
    part_d_summary()
    figures(exp)
    print(f"\nall done in {time.time()-t0:.0f}s. Tables: results/run_all_output.txt, figures: ./figures")
    print(config.STAMP)


if __name__ == "__main__":
    main()
