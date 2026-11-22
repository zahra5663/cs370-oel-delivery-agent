import json
import os
import streamlit as st
import config
from world import World
from search import DeliveryProblem, bfs, ucs, dls, ids, greedy, astar
from heuristics import Heuristics
from game import Game
from simulator import run_game, rival_start_for
from viz import draw_grid

st.set_page_config(page_title="CS-370 OEL Delivery Agent", layout="wide")
st.title("Autonomous Delivery Agent under Uncertainty and Competition")
st.caption(config.STAMP)

inst0, _, params0 = config.load()
with st.sidebar:
    st.header("Parameters")
    algo = st.selectbox("Algorithm", ["A*", "UCS", "BFS", "Greedy", "IDS", "DLS"])
    heur_name = st.selectbox("Heuristic (A*, Greedy)", ["own_mst_exact", "manhattan_nearest", "farthest_stop", "own x1.5 (inadmissible)"])
    slip = st.slider("Slip probability", 0.0, 0.6, float(inst0["slip_probability"]), 0.05)
    wind = st.slider("Wind drift probability", 0.0, 0.6, float(params0["wind_probability"]), 0.05)
    rdepth = st.slider("Rival search depth", 2, 6, int(inst0["adversary_search_depth"]))
    rspeed = st.slider("Rival speed", 0.2, 1.0, float(params0["rival_speed"]), 0.1)
    adepth = st.slider("Agent search depth", 2, 5, 4)
    st.subheader("Start and stops (row, col)")
    base = World()
    sr = st.number_input("start row", 0, base.n - 1, int(base.start[0]))
    sc = st.number_input("start col", 0, base.n - 1, int(base.start[1]))
    stops = []
    for i, s in enumerate(base.stops):
        a = st.number_input(f"stop {i+1} row", 0, base.n - 1, int(s[0]), key=f"r{i}")
        b = st.number_input(f"stop {i+1} col", 0, base.n - 1, int(s[1]), key=f"c{i}")
        stops.append((int(a), int(b)))

world = World({"slip_probability": slip, "wind_probability": wind, "adversary_search_depth": rdepth})
world.start = (int(sr), int(sc))
world.stops = stops
cells = [world.start] + stops
bad = [c for c in cells if not world.passable(*c)]
if bad or len(set(cells)) != len(cells):
    st.error(f"Start/stops must be distinct and not blocked. Problem cells: {bad}")
    st.stop()

H = Heuristics(world)
hmap = {"own_mst_exact": H.h_own, "manhattan_nearest": H.h_manhattan_nearest,
        "farthest_stop": H.h_farthest, "own x1.5 (inadmissible)": H.inflated(1.5)}
h = hmap[heur_name]

tab1, tab2, tab3 = st.tabs(["Route planning", "Rival game", "LLM dispatcher results"])

with tab1:
    p = DeliveryProblem(world)
    if algo == "A*":
        r = astar(p, h)
    elif algo == "UCS":
        r = ucs(p)
    elif algo == "BFS":
        r = bfs(p)
    elif algo == "Greedy":
        r = greedy(p, h)
    elif algo == "IDS":
        r = ids(p, max_depth=60, node_limit=200000)
    else:
        r = dls(p, limit=30, node_limit=200000)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", r["status"])
    c2.metric("Cost", "-" if r["cost"] is None else f"{r['cost']:.0f}")
    c3.metric("Nodes expanded", r["expanded"])
    c4.metric("Time (ms)", f"{r['time_s']*1000:.1f}")
    c5.metric("Peak memory (KB)", f"{r['peak_kb']:.0f}")
    if r["status"] != "ok":
        st.warning("No plan found within the node limit. IDS/DLS are exponential in time on this instance.")
    plot_col, _ = st.columns([3, 2])
    plot_col.pyplot(draw_grid(world, r["actions"], title=f"{algo} route on instance {world.inst['instance_id']}"))
    st.caption(config.STAMP)

with tab2:
    st.write("Agent (teal) plays Alpha-Beta against the interceptor rival (red X). Slip and wind apply to the agent only.")
    seed = st.number_input("Scenario / seed", 0, 999, 0)
    if st.button("Play game"):
        game = Game(world, H, weights={"cost": 1.0, "safety": 3.0, "margin": 6.0}, window=params0["game_window"])
        trace = []
        res = run_game(world, game, rival_start_for(world, int(seed)), 5000 + int(seed),
                       agent_depth=adepth, rival_depth=rdepth, rival_speed=rspeed, trace=trace)
        st.session_state["trace"] = trace
        st.session_state["res"] = res
    if "trace" in st.session_state:
        res, trace = st.session_state["res"], st.session_state["trace"]
        st.success(f"Result: {res['result']} after {res['turns']} turns, cost {res['cost']:.0f}, "
                   f"slips {res['slips']}, wind pushes {res['winds']}, replans {res['deviations']}")
        step = st.slider("Step (each agent move and each rival move is one step)", 0, len(trace) - 1, len(trace) - 1)
        a, rv, m = trace[step]
        fig = draw_grid(world, None, agent=a, rival=rv, mask=m, title=f"Step {step} of {len(trace) - 1}")
        ax = fig.axes[0]
        ax.plot([t[0][1] for t in trace[:step + 1]], [t[0][0] for t in trace[:step + 1]], color="#00897b", lw=2, zorder=4)
        ax.plot([t[1][1] for t in trace[:step + 1]], [t[1][0] for t in trace[:step + 1]], color="#e53935", lw=2, ls="--", zorder=4)
        plot_col, _ = st.columns([3, 2])
        plot_col.pyplot(fig)
    st.caption(config.STAMP)

with tab3:
    if os.path.exists("results/part_d.json"):
        with open("results/part_d.json") as f:
            d = json.load(f)
        import llm_agent
        rows = []
        for i, req in enumerate(llm_agent.REQUESTS):
            rows.append({"#": i + 1, "request": req["text"], "direct": d["direct"][i]["success"],
                         "LLM": d["llm_no_validation"][i]["success"],
                         "LLM + validation": d["llm_with_validation"][i]["success"]})
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No LLM results yet. Run: python llm_agent.py")
    st.caption(config.STAMP)

st.divider()
st.caption(config.STAMP)
