"""Discrete-event simulation of one CAN bus at frame and bit level.

The bus is idle or transmitting. At every arbitration point (bus idle and at least one message queued) the
queued message with the lowest identifier wins, which is what bitwise arbitration on a wired-AND bus does
when every node offers its highest-priority pending frame ([DAVIS2007] section 2.2 model). A frame occupies
the bus for its exact stuffed length (built bit by bit by `canframe`) or, in worst-case mode, for the
[DAVIS2007] Eq. (1) length. An error hitting a frame aborts it: the bus carries the error frame (31 bit
times worst case, [DAVIS2007] section 2.1.2) and the frame re-enters arbitration.

Used two ways: (1) the critical-instant scenario of each message, which must reproduce the analysis
exactly in worst-case mode; (2) random phasing, jitter and payloads, whose maxima must stay below the
analytical bound.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

from .canframe import on_bus_bits, worst_case_bits


@dataclass
class SimMsg:
    name: str
    can_id: int           # 29-bit (extended) or 11-bit; lower wins arbitration
    n_bytes: int
    T: int                # ns
    J: int = 0            # ns, max queuing jitter
    D: int = 0            # ns
    extended: bool = True
    offset: int = 0       # ns, first initiating event
    events: list[int] | None = None   # explicit initiating events (overrides offset/T)
    queues: list[int] | None = None   # explicit queue times matching `events`


@dataclass
class SimResult:
    max_R: dict[str, int] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)
    busy_ns: int = 0
    t_end: int = 0
    errors: int = 0
    frames: list[tuple[int, int, str]] = field(default_factory=list)   # (start, end, name) when traced

    @property
    def load(self) -> float:
        return self.busy_ns / self.t_end if self.t_end else 0.0


def simulate(msgs: list[SimMsg], tau: int, t_end: int, *, worst_case: bool = False, seed: int = 0,
             error_times: list[int] | None = None, trace: bool = False, blocker: SimMsg | None = None) -> SimResult:
    """Run the bus until t_end (ns). `blocker`: a frame that starts at t = -1 ns (critical-instant setup)."""
    rng = random.Random(seed)
    res = SimResult(t_end=t_end)
    releases: list[tuple[int, int, int, str]] = []    # (queue_time, can_id, event_time, name)
    by_name = {m.name: m for m in msgs}
    for m in msgs:
        res.max_R[m.name] = 0
        res.count[m.name] = 0
        if m.events is not None:
            for e, q in zip(m.events, m.queues or m.events):
                heapq.heappush(releases, (q, m.can_id, e, m.name))
            continue
        k = 0
        while True:
            e = m.offset + k * m.T
            if e > t_end:
                break
            q = e + (rng.randint(0, m.J) if m.J else 0)
            heapq.heappush(releases, (q, m.can_id, e, m.name))
            k += 1

    def length(m: SimMsg) -> int:
        if worst_case:
            return worst_case_bits(m.n_bytes, m.extended) * tau
        data = bytes(rng.getrandbits(8) for _ in range(m.n_bytes))
        return on_bus_bits(m.can_id, data, m.extended) * tau

    errs = sorted(error_times or [])
    ei = 0
    pending: list[tuple[int, int, str]] = []          # heap (can_id, event_time, name)
    t = 0
    if blocker is not None:
        t = -1 + length(blocker)
        res.busy_ns += t + 1
    while True:
        while releases and releases[0][0] <= t:
            q, cid, e, name = heapq.heappop(releases)
            heapq.heappush(pending, (cid, e, name))
        if not pending:
            if not releases or releases[0][0] > t_end:
                break
            t = releases[0][0]
            continue
        cid, e, name = heapq.heappop(pending)
        m = by_name[name]
        dur = length(m)
        start, end = t, t + dur
        while ei < len(errs) and errs[ei] < start:
            ei += 1
        if ei < len(errs) and errs[ei] < end:
            hit = errs[ei]
            ei += 1
            res.errors += 1
            t = hit + 31 * tau                        # aborted frame + worst-case error frame
            res.busy_ns += t - start
            heapq.heappush(pending, (cid, e, name))
            if trace:
                res.frames.append((start, t, name + "!err"))
            continue
        t = end
        res.busy_ns += dur
        if trace:
            res.frames.append((start, end, name))
        r = end - e
        res.count[name] += 1
        if r > res.max_R[name]:
            res.max_R[name] = r
        if t > t_end and not releases:
            break
    res.t_end = max(t_end, t)
    return res


def critical_instant(msgs: list[SimMsg], target: str, tau: int, horizon: int, worst_case: bool = True) -> int:
    """Max response of `target` in its critical-instant scenario ([DAVIS2007] section 3): the longest
    lower-priority frame starts 1 ns before t = 0; every message of priority >= target is queued at 0 with
    its first instance carrying full jitter, later instances queued with zero jitter."""
    ms = sorted(msgs, key=lambda m: m.can_id)
    tgt = next(m for m in ms if m.name == target)
    hep = [m for m in ms if m.can_id <= tgt.can_id]
    lp = [m for m in ms if m.can_id > tgt.can_id]
    blocker = max(lp, key=lambda m: worst_case_bits(m.n_bytes, m.extended)) if lp else None
    run = []
    for m in hep:
        ev, qu = [], []
        k = 0
        while True:
            e = k * m.T - m.J
            if e > horizon:
                break
            ev.append(e)
            qu.append(0 if k == 0 else e)
            k += 1
        run.append(SimMsg(m.name, m.can_id, m.n_bytes, m.T, m.J, m.D, m.extended, events=ev, queues=qu))
    if blocker is not None:
        blocker = SimMsg(blocker.name + "#blk", blocker.can_id, blocker.n_bytes, blocker.T, extended=blocker.extended)
    r = simulate(run, tau, horizon, worst_case=worst_case, blocker=blocker)
    return r.max_R[target]
