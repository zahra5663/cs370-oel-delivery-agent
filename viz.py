import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import config

COLORS = ["#d9d9d9", "#9ccc65", "#8d6e63", "#64b5f6"]


def positions_from_actions(world, actions):
    pos = world.start
    out = [pos]
    kinds = []
    for a in actions:
        nxt = pos
        for act, p2, c in world.neighbors(pos):
            if act == a:
                nxt = p2
        kinds.append(a == "T")
        pos = nxt
        out.append(pos)
    return out, kinds


def stamp_figure(fig):
    fig.text(0.5, 0.005, config.STAMP, ha="center", va="bottom", fontsize=8, color="#333333")


def draw_grid(world, actions=None, agent=None, rival=None, mask=0, title="", figsize=(7.2, 7.6)):
    n = world.n
    img = np.where(world.grid < 0, 4, world.grid)
    cmap = ListedColormap(COLORS + ["#212121"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    for a, b in world.tunnels:
        ax.plot([a[1], b[1]], [a[0], b[0]], color="#8e24aa", ls=":", lw=1.2, alpha=0.7)
        for p in (a, b):
            ax.scatter(p[1], p[0], marker="s", s=60, color="#8e24aa", zorder=3)
    if actions:
        pts, kinds = positions_from_actions(world, actions)
        for i in range(len(pts) - 1):
            ls = "--" if kinds[i] else "-"
            ax.plot([pts[i][1], pts[i + 1][1]], [pts[i][0], pts[i + 1][0]], color="#d81b60", lw=2.2, ls=ls, zorder=4)
    ax.scatter(world.start[1], world.start[0], marker="o", s=140, color="#ffffff", edgecolor="#000000", zorder=5)
    ax.annotate("S", (world.start[1], world.start[0]), ha="center", va="center", fontsize=8, zorder=6)
    for i, s in enumerate(world.stops):
        done = (mask >> i) & 1
        ax.scatter(s[1], s[0], marker="*", s=260, color="#fdd835" if not done else "#9e9e9e", edgecolor="#000000", zorder=5)
        ax.annotate(str(i + 1), (s[1] + 0.5, s[0] - 0.5), fontsize=9, fontweight="bold", zorder=6)
    if agent is not None:
        ax.scatter(agent[1], agent[0], marker="o", s=180, color="#00897b", edgecolor="#000000", zorder=7)
    if rival is not None:
        ax.scatter(rival[1], rival[0], marker="X", s=200, color="#e53935", edgecolor="#000000", zorder=7)
    ax.set_xticks(range(0, n, 5))
    ax.set_yticks(range(0, n, 5))
    ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=c, label=l) for c, l in
               zip(COLORS + ["#212121"], ["road", "grass", "mud", "ford", "blocked"])]
    handles.append(plt.Line2D([0], [0], marker="s", ls="", color="#8e24aa", label="tunnel"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp_figure(fig)
    return fig


def bar_with_ci(labels, means, cis, title, ylabel, path=None, figsize=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(labels)), means, yerr=cis, capsize=4, color="#00897b")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp_figure(fig)
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
