"""Build the per-bus CAN message sets of each candidate architecture and analyse them.

A message's worst-case transmission time C comes from `canframe` (classical) or the CAN FD bound; its
identifier from `j1939.make_id`, so arbitration order is the real J1939 order (priority, then PGN, then SA).
Growth clones every third message (rounded up to `growth` x count) one identifier step below the original.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import data
from .canframe import fd_worst_case_time, worst_case_bits
from .j1939 import make_id
from .rta import NO_ERRORS, NS, ErrorModel, Msg, Result, analyse, utilisation


@dataclass
class CanMsg:
    name: str
    can_id: int
    n_bytes: int
    T_ms: float
    J_ms: float
    domain: str
    e2e: bool
    sa: str


@dataclass
class Bus:
    id: str
    kind: str
    bps: int
    data_bps: int = 0
    dual: bool = False
    msgs: list[CanMsg] = field(default_factory=list)

    @property
    def tau_ns(self) -> int:
        return round(NS / self.bps)

    def C_ns(self, m: CanMsg) -> int:
        if self.kind == "canfd":
            return math.ceil(fd_worst_case_time(m.n_bytes, self.bps, self.data_bps) * NS)
        return worst_case_bits(m.n_bytes, True) * self.tau_ns

    def rta_msgs(self) -> list[Msg]:
        return [Msg(m.name, m.can_id, self.C_ns(m), round(m.T_ms * 1e6), round(m.T_ms * 1e6), round(m.J_ms * 1e6))
                for m in self.msgs]

    def load(self) -> float:
        return utilisation(self.rta_msgs())


def message_table() -> list[CanMsg]:
    d = data.load("messages")
    nodes = d["nodes"]
    jit = d["defaults"]["jitter_ms"]
    out = []
    for m in d["messages"]:
        sa = nodes[m["sa"]]
        cid = make_id(m["prio"], m["pgn"], sa, m.get("da", 0xFF))
        out.append(CanMsg(m["name"], cid, m["bytes"], float(m["period_ms"]), float(jit[m["domain"]]), m["domain"],
                          bool(m["e2e"]), m["sa"]))
    ids = [m.can_id for m in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate CAN identifiers in messages.yaml")
    return out


def add_growth(msgs: list[CanMsg], growth: float) -> list[CanMsg]:
    n = math.ceil(growth * len(msgs))
    ordered = sorted(msgs, key=lambda m: m.can_id)
    picks = ordered[::3][:n]
    if len(picks) < n:
        picks += [m for m in ordered if m not in picks][: n - len(picks)]
    used = {m.can_id for m in msgs}
    grown = list(msgs)
    for m in picks:
        cid = m.can_id + 1
        while cid in used:
            cid += 1
        used.add(cid)
        grown.append(CanMsg("G_" + m.name, cid, m.n_bytes, m.T_ms, m.J_ms, m.domain, m.e2e, m.sa))
    return grown


def build_arch(name: str, growth: bool = False, gw_jitter_ms: float = 0.0,
               periods: dict[str, float] | None = None) -> tuple[dict, list[Bus]]:
    a = data.load("architectures")
    arch = a["architectures"][name]
    table = message_table()
    for m in table:
        if periods and m.name in periods:
            m.T_ms = float(periods[m.name])
    move = arch.get("move", {})
    buses = []
    for b in arch["buses"]:
        bus = Bus(b["id"], b["kind"], b["bps"], b.get("data_bps", 0), b.get("dual", False))
        for m in table:
            if m.domain == "gw" and arch.get("vcu_on_ethernet"):
                continue           # the VCU reads the backbone directly
            target = move.get(m.name)
            if (target == bus.id) or (target is None and m.domain in b["domains"]):
                if m.domain == "gw":
                    m = CanMsg(m.name, m.can_id, m.n_bytes, m.T_ms,
                               m.J_ms + gw_jitter_ms + a["gateway_task_ms"]["value"], m.domain, m.e2e, m.sa)
                bus.msgs.append(m)
        if growth:
            bus.msgs = add_growth(bus.msgs, a["growth"]["value"])
        buses.append(bus)
    return arch, buses


@dataclass
class BusReport:
    bus: str
    bps: int
    kind: str
    n: int
    load: float
    results: dict[str, Result]
    worst_slack: float
    misses: list[str]


def analyse_bus(bus: Bus, err: ErrorModel = NO_ERRORS) -> BusReport:
    res = analyse(bus.rta_msgs(), bus.tau_ns, err)
    misses = [k for k, r in res.items() if not r.schedulable]
    slacks = [r.slack for r in res.values() if r.R is not None]
    ws = min(slacks) if slacks and not misses else (min(slacks) if slacks else float("-inf"))
    return BusReport(bus.id, bus.bps, bus.kind, len(bus.msgs), bus.load(), res, ws, misses)


def error_model(bus: Bus) -> ErrorModel:
    """Burst of 2 errors plus one error every 10 ms on every bus (assumed EMI environment, DOSSIER 8.3)."""
    e = data.load("environment")["can_error_model"]
    return ErrorModel(e["n_burst"]["value"], round(e["t_err_ms"]["value"] * 1e6))
