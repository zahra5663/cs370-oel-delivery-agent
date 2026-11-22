from collections import deque
import numpy as np
import config
from world import MOVES
from game import Search


def start_component(world):
    seen = {world.start}
    q = deque([world.start])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            p = (r + dr, c + dc)
            if world.passable(*p) and p not in seen:
                seen.add(p)
                q.append(p)
    return seen


def rival_start_for(world, index, min_d=8, max_d=16):
    comp = sorted(start_component(world))
    banned = {world.start, *world.stops}
    cands = [p for p in comp if p not in banned
             and min_d <= abs(p[0] - world.start[0]) + abs(p[1] - world.start[1]) <= max_d]
    rng = np.random.default_rng(config.seed_from_ids("|rival") + index)
    return cands[int(rng.integers(0, len(cands)))]


def intended_position(world, pos, action):
    if action == "T":
        return world.tunnel_map[pos]
    dr, dc = MOVES[action]
    p = (pos[0] + dr, pos[1] + dc)
    return p if world.passable(*p) else pos


def run_game(world, game, rival_start, seed, agent_depth=4, rival_depth=None,
             slip_p=None, wind_p=None, max_turns=120, rival_speed=1.0, trace=None):
    rival_depth = rival_depth or world.inst["adversary_search_depth"]
    old = (world.slip_p, world.wind_p)
    if slip_p is not None:
        world.slip_p = slip_p
    if wind_p is not None:
        world.wind_p = wind_p
    rng = np.random.default_rng(seed)
    S = Search(game)
    apos, rpos, mask, cost = world.start, rival_start, 0, 0.0
    slips = winds = deviations = turns = 0
    agent_nodes = rival_nodes = 0
    result = "draw"
    if trace is not None:
        trace.append((apos, rpos, mask))
    try:
        for t in range(max_turns):
            turns = t + 1
            _, action = S.run(apos, rpos, mask, agent_depth, "A", "agent", True, True)
            agent_nodes += S.nodes
            if action is None:
                break
            expected = intended_position(world, apos, action)
            new, c, info = world.stochastic_step(apos, action, rng)
            cost += c
            slips += info["slip"]
            winds += info["wind"]
            if new != expected:
                deviations += 1
            apos = new
            mask = game.update_mask(mask, apos)
            if trace is not None:
                trace.append((apos, rpos, mask))
            if apos == rpos:
                result = "loss"
                break
            if mask == game.full:
                result = "win"
                break
            if rng.random() < rival_speed:
                _, rp = S.run(apos, rpos, mask, rival_depth, "R", "rival", True, True)
                rival_nodes += S.nodes
                if rp is not None:
                    rpos = rp
                if trace is not None:
                    trace.append((apos, rpos, mask))
            if rpos == apos:
                result = "loss"
                break
    finally:
        world.slip_p, world.wind_p = old
    return {"result": result, "turns": turns, "cost": cost, "slips": slips, "winds": winds,
            "deviations": deviations, "agent_nodes": agent_nodes, "rival_nodes": rival_nodes}
