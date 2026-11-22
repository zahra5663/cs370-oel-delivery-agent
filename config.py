import hashlib
import numpy as np
from oel_instance import make_instance, stamp

CMS_IDS = ["532524", "510899", "529059"]

MEMBERS = [
    ("Zahra Saboor", "532524"),
    ("Zainab Khalid", "510899"),
    ("Umema Sufyan", "529059"),
]

STAMP = stamp(MEMBERS)

PARAMS = {
    "slip_probability": None,
    "wind_probability": 0.15,
    "tunnel_cost": 1,
    "num_tunnel_pairs": 2,
    "min_tunnel_span": 10,
    "adversary_search_depth": None,
    "game_window": 3,
    "rival_speed": 0.6,
    "start": None,
    "stops": None,
}


def seed_from_ids(tag=""):
    key = "|".join(sorted(c.strip().upper() for c in CMS_IDS)) + tag
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def load(overrides=None):
    inst, grid = make_instance(CMS_IDS)
    params = dict(PARAMS)
    if params["slip_probability"] is None:
        params["slip_probability"] = inst["slip_probability"]
    if params["adversary_search_depth"] is None:
        params["adversary_search_depth"] = inst["adversary_search_depth"]
    if overrides:
        params.update(overrides)
    inst = dict(inst)
    inst["slip_probability"] = params["slip_probability"]
    inst["adversary_search_depth"] = params["adversary_search_depth"]
    return inst, np.asarray(grid), params


if __name__ == "__main__":
    inst, grid, params = load()
    print(STAMP)
    print(inst["instance_id"], grid.shape, params)
