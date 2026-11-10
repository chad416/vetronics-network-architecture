"""Reliability of the steer-command path (VCU -> steering actuators) per architecture [W2016 ch. 34; MIL338B].

Exponential components, R = exp(-lambda t). One channel is a series system: the two end nodes' ports, its
connectors, harness segments and terminations, plus every other node on that channel through the fraction
of its failures that hold the bus dominant or babble ("bus killers"). Redundant channels form a 1-out-of-2
parallel block with a beta-factor common-cause term [IEC61508-6]:
R = (1 - (1 - R_ind)^2) * R_cc, with lambda_cc = beta * lambda_channel.
"""
from __future__ import annotations

import math

from . import data, network

FIT = 1e-9


def channel_lambda(kind: str, n_nodes: int, scale: float = 1.0) -> float:
    f = data.values("rma", "fit")
    d = data.load("rma")
    kf = d["bus_killer_fraction"]["value"]
    if kind == "eth":
        lam = 2 * f["eth_phy"] + 2 * f["eth_cable"] + f["eth_switch"]
    else:
        port = f["can_transceiver"] + f["can_controller_port"]
        lam = (2 * port + d["connectors_per_channel"]["value"] * f["connector"]
               + d["segments_per_channel"]["value"] * f["harness_segment"] + 2 * f["termination"]
               + max(0, n_nodes - 2) * kf * port)
    return lam * FIT * scale


def path_reliability(arch: str, t_h: float | None = None, scale: float = 1.0, beta: float | None = None) -> dict:
    d = data.load("rma")
    t = t_h if t_h is not None else d["mission_h"]["value"]
    b = d["common_cause_beta"]["value"] if beta is None else beta
    a = data.load("architectures")["architectures"][arch]
    if a.get("bywire_on_ethernet"):
        lam = channel_lambda("eth", 2, scale)
        nodes = 2
    else:
        _, buses = network.build_arch(arch)
        bw = next(x for x in buses if any(m.name == "STEER_CMD" for m in x.msgs))
        nodes = len({m.sa for m in bw.msgs})
        lam = channel_lambda("can", nodes, scale)
    if a.get("bywire_redundant"):
        r_ind = math.exp(-(1 - b) * lam * t)
        r = (1 - (1 - r_ind) ** 2) * math.exp(-b * lam * t)
    else:
        r = math.exp(-lam * t)
    mttf_h = 1 / lam if not a.get("bywire_redundant") else _mttf_1oo2(lam, b)
    mttr = d["mttr_h"]["value"]
    return {"arch": arch, "nodes_on_channel": nodes, "lambda_channel_per_h": lam, "R_mission": r,
            "F_mission": 1 - r, "MTTF_h": mttf_h, "Ai": mttf_h / (mttf_h + mttr)}


def _mttf_1oo2(lam: float, beta: float) -> float:
    li, lc = (1 - beta) * lam, beta * lam
    # integral of (2 e^{-li t} - e^{-2 li t}) e^{-lc t}
    return 2 / (li + lc) - 1 / (2 * li + lc)
