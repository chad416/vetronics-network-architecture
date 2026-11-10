"""Trade study (analysis of alternatives) per [W2016] ch. 32: screening on must-meet conditions, utility
functions, weighted figure of merit, one-at-a-time -10 % weight sensitivity and a random-weight Monte Carlo."""
from __future__ import annotations

import math
import random

from . import data


def _lin(x: float, lo: float, hi: float, decreasing: bool = False) -> float:
    u = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return 1.0 - u if decreasing else u


def utilities(raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """raw[alt][criterion] -> utility 0..1 with the shapes of DOSSIER 11.4."""
    out = {}
    for a, r in raw.items():
        out[a] = {
            "timing": _lin(r["timing"], 0.0, 0.6),                       # budget share left
            "load": _lin(r["load"], 0.2, 0.8, decreasing=True),
            "reliability": _lin(-math.log10(max(r["reliability"], 1e-12)), 6.0, 8.0),   # -log10 F per 12 h
            "maturity": _lin(r["maturity"], 4, 9),
            "integration": _lin(r["integration"], 0, 20, decreasing=True),
            "cost": _lin(r["cost"], 0, 25000, decreasing=True),
            "mass": _lin(r["mass"], 0, 12, decreasing=True),
            "capacity": _lin(r["capacity"], 0.0, 1.0),
        }
    return out


def weights(override: dict[str, float] | None = None) -> dict[str, float]:
    c = data.load("aoa")["criteria"]
    w = {k: float(v["rating"]) for k, v in c.items()}
    if override:
        w.update(override)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def fom(util: dict[str, dict[str, float]], w: dict[str, float]) -> dict[str, float]:
    return {a: sum(w[k] * u[k] for k in w) for a, u in util.items()}


def sensitivity(util, base_w_ratings: dict[str, float]) -> list[dict]:
    rows = []
    for k in base_w_ratings:
        r = dict(base_w_ratings)
        r[k] *= 0.9
        s = sum(r.values())
        w = {kk: v / s for kk, v in r.items()}
        f = fom(util, w)
        rows.append({"criterion": k, "winner": max(f, key=f.get), "fom": f})
    return rows


def monte_carlo(util, base_w_ratings: dict[str, float], n: int, seed: int = 1) -> dict[str, float]:
    rng = random.Random(seed)
    wins = {a: 0 for a in util}
    for _ in range(n):
        r = {k: v * rng.uniform(0.5, 1.5) for k, v in base_w_ratings.items()}
        s = sum(r.values())
        w = {k: v / s for k, v in r.items()}
        f = fom(util, w)
        wins[max(f, key=f.get)] += 1
    return {a: c / n for a, c in wins.items()}


def crossover(util, base_w_ratings: dict[str, float], criterion: str, rival: str, leader: str) -> float:
    """Nearest rating of `criterion` (searched up and down from its current value in 0.25 steps, range 0-30)
    at which `rival` overtakes `leader`; inf if none."""
    cur = base_w_ratings[criterion]
    best = float("inf")
    for x in [i / 4 for i in range(0, 121)]:
        r = dict(base_w_ratings)
        r[criterion] = x
        s = sum(r.values())
        f = fom(util, {k: v / s for k, v in r.items()})
        if f[rival] > f[leader] and abs(x - cur) < abs(best - cur):
            best = x
    return best
