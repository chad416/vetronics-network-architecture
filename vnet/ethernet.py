"""Switched Ethernet backbone: worst-case delay by network calculus and a packet-level simulation.

Each flow crosses a sequence of output ports (the source's own link, then the switch output ports). A port
of rate C serves IEEE 802.1Q priority classes (PCP 7 highest) with non-pre-emptive strict priority, FIFO
inside a class. By [LEBOUDEC2001] Proposition 1.3.4 (non-pre-emptive priority node, p. 21) generalised to
several classes, class k gets a rate-latency service curve with

    R_k = C - sum_{j higher} r_j,   T_k = (sum_{j higher} b_j + L_max,lower) / R_k

and, by Theorem 1.4.2 (delay bound, p. 23), the class aggregate with total burst B_k waits at most
D_k = T_k + B_k / R_k at that port. A flow's burst grows to b + r D_k at the next port (Theorem 1.4.3,
output flow). A store-and-forward switch adds its fixed forwarding latency; the frame's own transmission is
inside B_k. Arrival curves are token buckets (r, b) in wire bytes, with b at least one frame.
The simulation (`simulate`) runs the same flows through the same ports packet by packet; its maxima must
stay below the bounds.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

WIRE_OVERHEAD = 8 + 14 + 4 + 4 + 12   # preamble+SFD, MAC header, 802.1Q tag, FCS, inter-frame gap (bytes)
MAX_PAYLOAD = 1500


@dataclass
class Flow:
    name: str
    path: list[str]            # output ports in order
    pcp: int
    period_s: float = 0.0      # periodic: one message every period
    msg_bytes: int = 0         # application bytes per message (fragmented into <= 1500 B payloads)
    rate_bps: float = 0.0      # bursty (video, bulk): long-term rate
    burst_bytes: int = 0       # bursty: largest back-to-back burst at the source (e.g. an I-frame)
    deadline_s: float = 0.0
    jitter_s: float = 0.0

    def frames(self, nbytes: int) -> list[int]:
        out, left = [], nbytes
        while left > 0:
            p = min(MAX_PAYLOAD, left)
            out.append(max(p, 42) + WIRE_OVERHEAD)
            left -= p
        return out

    def token_bucket(self) -> tuple[float, float, int]:
        """(r bytes/s, b bytes, max frame bytes) on the wire."""
        if self.period_s > 0:
            fr = self.frames(self.msg_bytes)
            wire = sum(fr)
            b = wire * (1 + self.jitter_s / self.period_s)
            return wire / self.period_s, b, max(fr)
        fr = self.frames(self.burst_bytes)
        wire_burst = sum(fr)
        r = self.rate_bps / 8 * (1 + WIRE_OVERHEAD / MAX_PAYLOAD)
        return r, wire_burst, max(fr)


@dataclass
class Network:
    ports: dict[str, float]               # port -> rate in bits/s
    switch_latency_s: float = 4e-6
    flows: list[Flow] = field(default_factory=list)


def bounds(net: Network) -> dict[str, float]:
    """End-to-end delay bound per flow (s), by propagating bursts port by port in path order."""
    burst = {f.name: f.token_bucket()[1] for f in net.flows}
    rate = {f.name: f.token_bucket()[0] for f in net.flows}
    lmax = {f.name: f.token_bucket()[2] for f in net.flows}
    total = {f.name: 0.0 for f in net.flows}
    hop = {f.name: 0 for f in net.flows}
    done: set[str] = set()
    remaining = len(net.flows)
    for _ in range(64):
        # ports whose every flow has arrived (all upstream hops done) can be evaluated
        progress = False
        for port, C in net.ports.items():
            here = [f for f in net.flows if hop[f.name] < len(f.path) and f.path[hop[f.name]] == port]
            if not here:
                continue
            waiting = [f for f in net.flows if port in f.path[hop[f.name]:] and f not in here]
            if waiting:
                continue
            cap = C / 8.0
            allf = [f for f in net.flows if port in f.path]
            new_burst = {}
            for f in here:
                hi = [g for g in allf if g.pcp > f.pcp]
                same = [g for g in allf if g.pcp == f.pcp]
                lo = [g for g in allf if g.pcp < f.pcp]
                R = cap - sum(rate[g.name] for g in hi)
                if R <= sum(rate[g.name] for g in same):
                    total[f.name] = float("inf")
                    D = float("inf")
                else:
                    T = (sum(burst[g.name] for g in hi) + max((lmax[g.name] for g in lo), default=0)) / R
                    D = T + sum(burst[g.name] for g in same) / R
                    total[f.name] += D + (net.switch_latency_s if hop[f.name] > 0 else 0.0)
                new_burst[f.name] = burst[f.name] + rate[f.name] * D
            for f in here:
                burst[f.name] = new_burst[f.name]
                hop[f.name] += 1
                if hop[f.name] == len(f.path):
                    done.add(f.name)
            progress = True
        if len(done) == remaining:
            break
        if not progress:
            raise ValueError("cyclic port dependency; feed-forward topologies only")
    return total


def simulate(net: Network, t_end: float = 2.0, seed: int = 0, gop: int = 30, i_ratio: float = 10.0,
             aligned: bool = False) -> dict[str, float]:
    """Packet-level store-and-forward simulation; returns the max end-to-end delay per flow (s).
    Periodic flows start at random phases with random jitter; bursty flows send video frames at 30 fps
    with one I-frame (i_ratio x a P-frame, capped at burst_bytes) every `gop` frames, or bulk bursts.
    `aligned`: every source starts at t = 0 with its largest burst (the phasing the bound assumes)."""
    rng = random.Random(seed)
    ev: list[tuple] = []          # (time, seq, kind, flow, frame_bytes, hop, t0)
    seq = 0
    fl = {f.name: f for f in net.flows}
    for f in net.flows:
        if f.period_s > 0:
            t = 0.0 if aligned else rng.uniform(0, f.period_s)
            while t < t_end:
                tj = t + (0.0 if aligned else rng.uniform(0, f.jitter_s))
                for fr in f.frames(f.msg_bytes):
                    heapq.heappush(ev, (tj, seq, "arr", f.name, fr, 0, tj))
                    seq += 1
                t += f.period_s
        else:
            fps = 30.0
            per = f.rate_bps / 8 / fps
            p_size = per * gop / (gop - 1 + i_ratio)
            i_size = f.burst_bytes if aligned else min(f.burst_bytes, p_size * i_ratio)
            t, k = (0.0, 0) if aligned else (rng.uniform(0, 1 / fps), rng.randrange(gop))
            while t < t_end:
                size = int(i_size if k % gop == 0 else p_size)
                for fr in f.frames(size):
                    heapq.heappush(ev, (t, seq, "arr", f.name, fr, 0, t))
                    seq += 1
                t += 1 / fps
                k += 1
    busy_until = {p: 0.0 for p in net.ports}
    queues: dict[str, list] = {p: [] for p in net.ports}
    worst = {f.name: 0.0 for f in net.flows}

    def try_start(port: str, now: float):
        nonlocal seq
        if busy_until[port] > now + 1e-15 or not queues[port]:
            return
        _, _, name, fr, h, t0 = heapq.heappop(queues[port])
        done = now + fr * 8 / net.ports[port]
        busy_until[port] = done
        heapq.heappush(ev, (done, seq, "dep", name, fr, h, t0, port))
        seq += 1

    while ev:
        e = heapq.heappop(ev)
        now, kind, name, fr, h, t0 = e[0], e[2], e[3], e[4], e[5], e[6]
        f = fl[name]
        if kind == "arr":
            port = f.path[h]
            heapq.heappush(queues[port], (-f.pcp, e[1], name, fr, h, t0))
            try_start(port, now)
        else:
            port = e[7]
            if h + 1 < len(f.path):
                nxt = now + net.switch_latency_s
                heapq.heappush(ev, (nxt, seq, "arr", name, fr, h + 1, t0))
                seq += 1
            else:
                worst[name] = max(worst[name], now - t0)
            try_start(port, now)
    return worst


def build(rate_mbps: float, bywire: bool = False, video_scale: float = 1.0) -> Network:
    """Backbone of data/ethernet.yaml at `rate_mbps` on every link; `bywire` adds the A4 control plane."""
    from . import data
    d = data.load("ethernet")
    recs = list(d["flows"]) + (list(d["bywire_flows"]) if bywire else [])
    flows = []
    ports: dict[str, float] = {}
    for r in recs:
        for p in r["path"]:
            ports[p] = rate_mbps * 1e6
        if "period_ms" in r:
            flows.append(Flow(r["name"], list(r["path"]), r["pcp"], period_s=r["period_ms"] / 1e3, msg_bytes=r["bytes"],
                              deadline_s=r["deadline_ms"] / 1e3, jitter_s=0.001))
        else:
            flows.append(Flow(r["name"], list(r["path"]), r["pcp"], rate_bps=r["rate_mbps"] * 1e6 * video_scale,
                              burst_bytes=int(r["burst_kB"] * 1000 * video_scale), deadline_s=r["deadline_ms"] / 1e3))
    return Network(ports, d["switch_latency_us"]["value"] * 1e-6, flows)
