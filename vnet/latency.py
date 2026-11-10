"""End-to-end latency of cause-effect chains across tasks, CAN buses and the Ethernet backbone.

Tasks are periodic and not synchronised with each other or with the buses, so each reader samples the newest
value once per period. For a chain sensor-task -> message -> controller-task -> message -> actuator, the worst
data age at the actuator is ([W2016] Principle 31.6: queue + processing + transport time; DOSSIER 8.6)

    L = R_1 + T_ctrl + R_2 + T_act

where R_i is the worst-case response time of message i measured from the release of its sending task (its
queuing jitter J covers the sender's execution), T_ctrl the controller period (a fresh value can just miss an
activation) and T_act the actuator's sampling period. Ethernet hops use the network-calculus bound plus the
sender jitter and a fixed protocol-stack latency.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import data, ethernet, network
from .rta import ErrorModel


@dataclass
class ChainResult:
    chain: str
    arch: str
    hops: list[tuple[str, float]]     # (label, ms)
    total_ms: float
    budget_ms: float

    @property
    def ok(self) -> bool:
        return self.total_ms <= self.budget_ms


def chain_defs() -> dict:
    return data.load("chains")


def _can_R(arch: str, growth: bool, errors: bool, periods: dict[str, float] | None = None) -> dict[str, float]:
    a = data.load("architectures")["architectures"][arch]
    eth = ethernet.build(a["ethernet_mbps"], bywire=a.get("bywire_on_ethernet", False))
    gw_j = ethernet.bounds(eth).get("GW_CAN", 0.0) * 1e3
    _, buses = network.build_arch(arch, growth=growth, gw_jitter_ms=gw_j, periods=periods)
    out: dict[str, float] = {}
    for b in buses:
        err = network.error_model(b) if errors else ErrorModel()
        rep = network.analyse_bus(b, err)
        for k, r in rep.results.items():
            out.setdefault(k, float("inf") if r.R is None else r.R / 1e6)
    return out


def evaluate(arch: str, growth: bool = True, errors: bool = True, budgets: dict[str, float] | None = None,
             task_periods: dict[str, float] | None = None) -> list[ChainResult]:
    a = data.load("architectures")["architectures"][arch]
    defs = chain_defs()
    periods = None
    if task_periods and "VCU" in task_periods:      # the VCU's inputs and outputs follow its period
        periods = {k: task_periods["VCU"] for k in ("INS_DYN", "STEER_CMD", "BRAKE_CMD", "DRIVER_IN", "SAS1_STAT", "SAS2_STAT")}
    R = _can_R(arch, growth, errors, periods)
    eth = ethernet.build(a["ethernet_mbps"], bywire=a.get("bywire_on_ethernet", False))
    eb = {k: v * 1e3 for k, v in ethernet.bounds(eth).items()}
    stack = defs["ethernet_stack_ms"]["value"]
    jit = data.load("messages")["defaults"]["jitter_ms"]["bw"]
    out = []
    for name, c in defs["chains"].items():
        hops: list[tuple[str, float]] = []
        for h in c["hops"]:
            if "task" in h:
                hops.append((h["task"], float((task_periods or {}).get(h["task"], h["period_ms"]))))
                continue
            msg = h["msg"]
            if a.get("bywire_on_ethernet") and h.get("eth"):
                hops.append((h["eth"] + " (Ethernet)", eb[h["eth"]] + jit + stack))
            elif a.get("vcu_on_ethernet") and h.get("eth_if_vcu"):
                hops.append((h["eth_if_vcu"] + " (Ethernet)", eb[h["eth_if_vcu"]] + jit + stack))
            else:
                hops.append((msg, R.get(msg, float("inf"))))
        total = sum(v for _, v in hops)
        budget = (budgets or {}).get(name, c.get("budget_ms", float("inf")))
        out.append(ChainResult(name, arch, hops, total, budget))
    return out
