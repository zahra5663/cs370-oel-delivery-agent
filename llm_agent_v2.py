import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import numpy as np
import config
from dispatch_tools import Dispatcher
from run_games import wilson

MAX_STEPS = 8

SYSTEM_PROMPT = """You are the dispatcher of a delivery agent in a grid city. You control the agent only through tools.
City facts: there are {num_stops} delivery stops, numbered 1 to {num_stops}. Terrain types: road, grass, mud, ford. Travel costs per cell: {terrain_cost}. The city has teleport tunnels and a wind that sometimes pushes the agent; moves can also slip.

Tools (call exactly one per step):
1. plan_route(algorithm, first_stop, avoid_terrain, max_cost)
   algorithm: one of bfs, ucs, greedy, astar, ids (default astar). first_stop: stop number to deliver first (optional). avoid_terrain: list of terrain names to avoid (optional). max_cost: budget limit for the whole route (optional).
   Returns plan_id, cost, steps, within_budget.
2. simulate(plan_id): runs the plan in the stochastic world (slips and wind) and reports the actual cost.
3. wind_risk(plan_id): reports how exposed a plan is to wind drift and the expected extra cost.

Reply with ONE JSON object and nothing else, in this form:
{{"thought": "your reasoning", "action": "<tool name>", "args": {{...}}}}
When you are finished, reply with:
{{"thought": "...", "action": "final", "args": {{"plan_id": <number or null>, "feasible": true or false, "answer": "short answer for the dispatcher"}}}}
If the request cannot be met, or asks for something that does not exist in this city, set feasible to false and plan_id to null."""

REQUESTS = [
    {"text": "Plan the cheapest route that delivers all parcels.", "args": {}, "ref": {}},
    {"text": "Deliver all parcels using A*.", "args": {"algorithm": "astar"}, "ref": {"algorithm": "astar"}},
    {"text": "Plan a route with breadth-first search.", "args": {"algorithm": "bfs"}, "ref": {"algorithm": "bfs"}},
    {"text": "Deliver stop 1 first, then the rest, as cheaply as possible.", "args": {"first_stop": 1}, "ref": {"first_stop": 1}},
    {"text": "Deliver stop 3 first, then the rest, as cheaply as possible.", "args": {"first_stop": 3}, "ref": {"first_stop": 3}},
    {"text": "Avoid mud and plan the cheapest route.", "args": {"avoid_terrain": ["mud"]}, "ref": {"avoid": ["mud"]}},
    {"text": "Avoid grass and plan the cheapest route.", "args": {"avoid_terrain": ["grass"]}, "ref": {"avoid": ["grass"]}},
    {"text": "Avoid lava and plan the cheapest route.", "args": {"avoid_terrain": ["lava"]}, "ref": {}, "expect": "invalid"},
    {"text": "Deliver stop 2 first and avoid mud, as cheaply as possible.", "args": {"first_stop": 2, "avoid_terrain": ["mud"]}, "ref": {"first_stop": 2, "avoid": ["mud"]}},
    {"text": "Deliver stop 1 first and avoid grass, as cheaply as possible.", "args": {"first_stop": 1, "avoid_terrain": ["grass"]}, "ref": {"first_stop": 1, "avoid": ["grass"]}},
    {"text": "Plan a route with a total cost under 90.", "args": {"max_cost": 90}, "ref": {"max_cost": 90}},
    {"text": "Plan a route that costs less than 60.", "args": {"max_cost": 60}, "ref": {"max_cost": 60}, "expect": "infeasible"},
    {"text": "Plan the cheapest route and tell me how exposed it is to wind drift.", "args": {}, "ref": {}, "wind": True},
    {"text": "Deliver stop 3 first and check how exposed that route is to wind drift.", "args": {"first_stop": 3}, "ref": {"first_stop": 3}, "wind": True},
    {"text": "Avoid mud, plan the cheapest route, then simulate it and report the actual cost.", "args": {"avoid_terrain": ["mud"]}, "ref": {"avoid": ["mud"]}, "simulate": True},
    {"text": "Use uniform-cost search, deliver stop 2 first, and simulate the result.", "args": {"algorithm": "ucs", "first_stop": 2}, "ref": {"algorithm": "ucs", "first_stop": 2}, "simulate": True},
    {"text": "Deliver stop 7 first.", "args": {"first_stop": 7}, "ref": {}, "expect": "invalid"},
    {"text": "Avoid grass and deliver stop 3 first.", "args": {"first_stop": 3, "avoid_terrain": ["grass"]}, "ref": {"first_stop": 3, "avoid": ["grass"]}, "expect": "infeasible"},
    {"text": "Avoid road, grass, mud and ford at the same time.", "args": {"avoid_terrain": ["road", "grass", "mud", "ford"]}, "ref": {"avoid": ["road", "grass", "mud", "ford"]}, "expect": "infeasible"},
    {"text": "Deliver stop 1 first, avoid mud, keep the cost under 80, use A*, simulate the route and report its wind exposure.",
     "args": {"algorithm": "astar", "first_stop": 1, "avoid_terrain": ["mud"], "max_cost": 80},
     "ref": {"algorithm": "astar", "first_stop": 1, "avoid": ["mud"], "max_cost": 80}, "simulate": True, "wind": True},
]


class Gemini:
    def __init__(self):
        self.key = os.environ.get("GEMINI_API_KEY")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
        if " " in self.model:
            sys.exit(f"Model name has a space: '{self.model}'. Fix GEMINI_MODEL.")
        self.delay = float(os.environ.get("GEMINI_DELAY", "3"))
        if not self.key:
            sys.exit("Set GEMINI_API_KEY first (see instructions).")

    def list_models(self):
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models",
                                     headers={"x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]

    def generate(self, system, messages, attempts=4):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": r, "parts": [{"text": t}]} for r, t in messages],
                "generationConfig": {"temperature": 0}}
        data = json.dumps(body).encode()
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, headers={
                "Content-Type": "application/json", "x-goog-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read().decode())
                time.sleep(self.delay)
                parts = out["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:400]
                if e.code in (429, 500, 503) and attempt < attempts - 1:
                    wait = 8 * (attempt + 1)
                    print(f"  [retry {attempt+1}] HTTP {e.code}, waiting {wait}s: {msg[:150]}", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {msg}")
            except Exception as e:
                if attempt < attempts - 1:
                    print(f"  [retry {attempt+1}] {type(e).__name__}: {e}", flush=True)
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError("Gemini failed after retries")


class MockLLM:
    def __init__(self, seed=0, error_rate=0.35):
        self.rng = np.random.default_rng(seed)
        self.error_rate = error_rate

    def generate(self, system, messages):
        text = messages[0][1].lower()
        steps = sum(1 for r, _ in messages if r == "model")
        last_obs = messages[-1][1]
        args = {}
        m = re.search(r"stop (\d+) first", text)
        if m:
            args["first_stop"] = int(m.group(1))
        terr = [t for t in ("road", "grass", "mud", "ford", "lava") if t in text and "avoid" in text]
        if terr:
            args["avoid_terrain"] = terr
        m = re.search(r"(?:under|less than) (\d+)", text)
        if m:
            args["max_cost"] = int(m.group(1))
        for name, key in (("breadth", "bfs"), ("uniform", "ucs"), ("a*", "astar")):
            if name in text:
                args["algorithm"] = key
        if self.rng.random() < self.error_rate and "first_stop" in args:
            args["first_stop"] -= 1
        if self.rng.random() < self.error_rate and "avoid_terrain" in args:
            args["avoid_terrain"] = ["swamp"]
        if steps == 0:
            return json.dumps({"thought": "plan first", "action": "plan_route", "args": args})
        pid = re.search(r'"plan_id": (\d+)', "\n".join(t for r, t in messages if r == "user" and "Observation" in t))
        if "error" in last_obs.lower():
            return json.dumps({"thought": "retry", "action": "plan_route", "args": {**args, "first_stop": args.get("first_stop", 1) if args.get("first_stop", 1) in (1, 2, 3) else 1, "avoid_terrain": [t for t in args.get("avoid_terrain", []) if t in ("road", "grass", "mud", "ford")]}})
        used = [json.loads(t).get("action") for r, t in messages if r == "model" for t in [t]]
        if pid is None:
            return json.dumps({"thought": "no plan", "action": "final", "args": {"plan_id": None, "feasible": False, "answer": "cannot"}})
        plan_id = int(pid.group(1))
        if "simulate" in text and "simulate" not in used:
            return json.dumps({"thought": "simulate", "action": "simulate", "args": {"plan_id": plan_id}})
        if "wind" in text and "wind_risk" not in used:
            return json.dumps({"thought": "wind", "action": "wind_risk", "args": {"plan_id": plan_id}})
        within = '"within_budget": false' not in last_obs and "within_budget\": false" not in "\n".join(t for r, t in messages)
        return json.dumps({"thought": "done", "action": "final", "args": {"plan_id": plan_id, "feasible": bool(within), "answer": "ok"}})


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def run_agent(backend, request_text, dispatcher):
    system = SYSTEM_PROMPT.format(num_stops=dispatcher.describe()["num_stops"],
                                  terrain_cost=dispatcher.describe()["terrain_cost"])
    messages = [("user", "Request: " + request_text)]
    trace, final = [], None
    for _ in range(MAX_STEPS):
        raw = backend.generate(system, messages)
        msg = parse_json(raw)
        messages.append(("model", raw))
        if not isinstance(msg, dict) or "action" not in msg:
            obs = "ERROR: reply with exactly one JSON object with keys thought, action, args."
            trace.append({"thought": "", "action": "(unparseable)", "args": {}, "observation": obs})
            messages.append(("user", "Observation: " + obs))
            continue
        action, args = msg.get("action"), msg.get("args", {})
        if action == "final":
            final = args if isinstance(args, dict) else None
            trace.append({"thought": msg.get("thought", ""), "action": "final", "args": args, "observation": ""})
            break
        result = dispatcher.call(action, args)
        obs = json.dumps(result)
        trace.append({"thought": msg.get("thought", ""), "action": action, "args": args, "observation": obs})
        messages.append(("user", "Observation: " + obs))
    return final, trace


def evaluate(spec, d, final):
    expect = spec.get("expect", "ok")
    reasons = []
    used = {c["action"] for c in d.calls}
    needed = {"plan_route"} | ({"simulate"} if spec.get("simulate") else set()) | ({"wind_risk"} if spec.get("wind") else set())
    if expect != "ok":
        needed = {"plan_route"} if expect == "infeasible" else set()
    wrong_tool = bool(used - needed) or bool(needed - used and expect == "ok")
    invalid_attempts = sum(1 for c in d.calls if not c["valid"])
    if final is None:
        return False, ["no_final_answer"], wrong_tool, invalid_attempts, None
    if expect in ("invalid", "infeasible"):
        ok = final.get("feasible") is False
        if not ok:
            reasons.append("claimed_feasible")
        return ok, reasons, wrong_tool, invalid_attempts, None
    plan = d.plans.get(final.get("plan_id"))
    if plan is None or final.get("feasible") is not True:
        return False, ["no_valid_plan"], wrong_tool, invalid_attempts, None
    ref = spec["ref"]
    if plan["first_stop"] != ref.get("first_stop"):
        reasons.append("ignored_first_stop")
    if not set(ref.get("avoid", [])) <= set(plan["avoid"]):
        reasons.append("ignored_avoid")
    if ref.get("algorithm") and plan["algorithm"] != ref["algorithm"]:
        reasons.append("wrong_algorithm")
    if ref.get("max_cost") is not None and (plan["max_cost"] != ref["max_cost"] or not plan["within_budget"]):
        reasons.append("ignored_budget")
    ref_plan = Dispatcher(validate=True)
    r = ref_plan.plan_route(ref.get("algorithm", "astar"), ref.get("first_stop"), ref.get("avoid", []), None)
    if r["status"] == "ok" and plan["cost"] != r["cost"]:
        reasons.append("not_cheapest")
    pid = final["plan_id"]
    if spec.get("simulate") and not any(c["action"] == "simulate" and c["valid"] and c["args"].get("plan_id") == pid for c in d.calls):
        reasons.append("missing_simulate")
    if spec.get("wind") and not any(c["action"] == "wind_risk" and c["valid"] and c["args"].get("plan_id") == pid for c in d.calls):
        reasons.append("missing_wind_risk")
    return not reasons, reasons, wrong_tool, invalid_attempts, plan["cost"]


def run_direct(spec, idx):
    d = Dispatcher(validate=True, seed=config.seed_from_ids("|llm") + idx)
    res = d.call("plan_route", spec["args"])
    final = {"plan_id": None, "feasible": False}
    if res.get("status") == "ok" and res.get("within_budget"):
        final = {"plan_id": res["plan_id"], "feasible": True}
        if spec.get("simulate"):
            d.call("simulate", {"plan_id": res["plan_id"]})
        if spec.get("wind"):
            d.call("wind_risk", {"plan_id": res["plan_id"]})
    ok, reasons, wt, inv, cost = evaluate(spec, d, final)
    return {"success": ok, "calls": len(d.calls), "cost": cost, "reasons": reasons, "wrong_tool": False, "invalid": 0}


def run_llm(backend, spec, idx, validate):
    d = Dispatcher(validate=validate, seed=config.seed_from_ids("|llm") + idx)
    try:
        final, trace = run_agent(backend, spec["text"], d)
    except Exception as e:
        return {"success": False, "calls": len(d.calls), "cost": None, "reasons": [f"backend_error: {e}"],
                "wrong_tool": False, "invalid": 0, "trace": []}
    ok, reasons, wt, inv, cost = evaluate(spec, d, final)
    return {"success": ok, "calls": len(d.calls), "cost": cost, "reasons": reasons, "wrong_tool": wt,
            "invalid": inv, "trace": trace}


def summarize(name, rows):
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, n)
    costs = [r["cost"] for r in rows if r["cost"] is not None and r["success"]]
    return (f"{name:<28}{100*k/n:>7.0f}% [{100*max(0,lo):.0f}, {100*hi:.0f}]"
            f"{np.mean([r['calls'] for r in rows]):>9.1f}"
            f"{(np.mean(costs) if costs else float('nan')):>10.1f}"
            f"{sum(r['wrong_tool'] for r in rows):>10}{sum(r['invalid'] for r in rows):>9}"
            f"{sum(1 for r in rows if any(x.startswith('ignored') or x in ('wrong_algorithm', 'not_cheapest') for x in r['reasons'])):>9}")


def main():
    use_mock = "--mock" in sys.argv
    if "--list-models" in sys.argv:
        print("\n".join(Gemini().list_models()))
        return
    backend = MockLLM() if use_mock else Gemini()
    if use_mock:
        print("MOCK BACKEND: pipeline test only, do NOT use these numbers in the report")
    print(config.STAMP)
    print("model:", "mock" if use_mock else backend.model)
    if not use_mock:
        try:
            t0 = time.time()
            out = backend.generate("Reply with the single word OK.", [("user", "ping")], attempts=2)
            print(f"preflight OK in {time.time()-t0:.1f}s, model said: {out[:40]!r}", flush=True)
        except Exception as e:
            sys.exit(f"Preflight check failed, nothing was run:\n{e}")
    only = None
    for a in sys.argv:
        if a.startswith("--only="):
            only = [int(x) - 1 for x in a.split("=")[1].split(",")]
    prev = None
    if only is not None and os.path.exists("results/part_d.json"):
        with open("results/part_d.json") as f:
            prev = json.load(f)
    direct = [run_direct(s, i) for i, s in enumerate(REQUESTS)]
    no_val = []
    with_val = []
    for i, s in enumerate(REQUESTS):
        if prev is not None and i not in only:
            no_val.append(prev["llm_no_validation"][i])
            with_val.append(prev["llm_with_validation"][i])
            continue
        no_val.append(run_llm(backend, s, i, False))
        with_val.append(run_llm(backend, s, i, True))
        print(f"request {i+1:>2}: direct={direct[i]['success']!s:<5} llm={no_val[i]['success']!s:<5} llm+validation={with_val[i]['success']!s:<5}", flush=True)
    print()
    print(f"{'mode':<28}{'success [95% CI]':>18}{'calls':>9}{'cost':>10}{'wrongtool':>10}{'badargs':>9}{'ignored':>9}")
    print(summarize("direct planner", direct))
    print(summarize("LLM (no validation)", no_val))
    print(summarize("LLM + validation layer", with_val))
    print(config.STAMP)
    print("\nFAILURE LOG (LLM without validation / with validation)")
    for i, (a, b) in enumerate(zip(no_val, with_val)):
        if a["reasons"] or b["reasons"] or a["wrong_tool"] or b["wrong_tool"] or a["invalid"] or b["invalid"]:
            print(f"request {i+1:>2}: no-val {a['reasons']} wrong_tool={a['wrong_tool']} bad_args={a['invalid']} | "
                  f"val {b['reasons']} wrong_tool={b['wrong_tool']} bad_args={b['invalid']}")
    print(config.STAMP)
    os.makedirs("results", exist_ok=True)
    with open("results/part_d.json", "w") as f:
        json.dump({"direct": direct, "llm_no_validation": no_val, "llm_with_validation": with_val}, f, indent=2, default=str)
    with open("results/part_d_traces.txt", "w") as f:
        f.write(config.STAMP + "\n")
        shown = 0
        for i, r in enumerate(with_val):
            if r["trace"] and (shown < 2 or r["invalid"]):
                f.write(f"\n=== Request {i+1}: {REQUESTS[i]['text']} ===\n")
                for k, t in enumerate(r["trace"], 1):
                    f.write(f"Step {k}\n  Thought (Reason): {t['thought']}\n  Action (Act): {t['action']} {json.dumps(t['args'])}\n"
                            f"  Observation (Sense): {t['observation']}\n")
                f.write(config.STAMP + "\n")
                shown += 1
                if shown >= 3:
                    break
    print("saved results/part_d.json and results/part_d_traces.txt")


if __name__ == "__main__":
    main()
