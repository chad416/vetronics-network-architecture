"""Worst-case response-time analysis of CAN (fixed-priority, non-pre-emptive).

Implements, with integer nanoseconds so every ceiling is exact:
- the original analysis of Tindell and Burns as restated in [DAVIS2007] Eq. (6)-(7) ("legacy"), kept to
  show the flaw;
- the revised exact analysis of [DAVIS2007] Eq. (8), (10)-(13), with the error model of Eq. (18)-(20);
- the sufficient test of [DAVIS2007] Eq. (16).

Error model: F(t) = n_burst + ceil(t / t_err) errors in any interval t (t_err = 0 means only the burst),
each costing 31 bit times of error frame plus the retransmission of the longest frame of priority >= m
([DAVIS2007] Eq. 18). A message is identified by its priority (lower number = higher priority; on CAN, the
lower identifier).
"""
from __future__ import annotations

from dataclasses import dataclass, field

NS = 1_000_000_000


@dataclass(frozen=True)
class Msg:
    name: str
    prio: int          # unique; lower = higher priority (= lower CAN identifier)
    C: int             # worst-case transmission time, ns (includes stuffing and intermission)
    T: int             # period or minimum inter-arrival, ns
    D: int             # deadline relative to the initiating event, ns
    J: int = 0         # queuing jitter, ns


@dataclass(frozen=True)
class ErrorModel:
    n_burst: int = 0
    t_err: int = 0     # ns between sporadic errors; 0 = none

    def F(self, t: int) -> int:
        if self.t_err <= 0:
            return self.n_burst
        return self.n_burst + -(-t // self.t_err)


@dataclass
class Result:
    name: str
    R: int | None                  # worst-case response time, ns (None = unbounded / not converged)
    D: int
    Q: int = 1                     # instances checked in the busy period
    busy: int = 0
    schedulable: bool = False
    per_instance: list[int] = field(default_factory=list)

    @property
    def slack(self) -> float:
        return float("nan") if self.R is None else (self.D - self.R) / self.D


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _sorted(msgs: list[Msg]) -> list[Msg]:
    ms = sorted(msgs, key=lambda m: m.prio)
    if len({m.prio for m in ms}) != len(ms):
        raise ValueError("CAN priorities (identifiers) must be unique")
    return ms


def utilisation(msgs: list[Msg]) -> float:
    return sum(m.C / m.T for m in msgs)


def _E(err: ErrorModel, t: int, tau: int, cmax_hep: int) -> int:
    return (31 * tau + cmax_hep) * err.F(t) if (err.n_burst or err.t_err) else 0


NO_ERRORS = ErrorModel()


def analyse(msgs: list[Msg], tau: int, err: ErrorModel = NO_ERRORS, limit: int | None = None) -> dict[str, Result]:
    """Exact worst-case response times per [DAVIS2007] section 3 (with Eq. 19-20 when err is given)."""
    ms = _sorted(msgs)
    out: dict[str, Result] = {}
    for i, m in enumerate(ms):
        hp, lp = ms[:i], ms[i + 1:]
        hep = ms[: i + 1]
        B = max((k.C for k in lp), default=0)
        cmax = max(k.C for k in hep)
        cap = limit if limit is not None else 1000 * max(k.T for k in ms)
        u = sum(k.C / k.T for k in hep)
        if u >= 1.0:
            out[m.name] = Result(m.name, None, m.D, schedulable=False)
            continue
        # busy period, Eq. (19)
        t = m.C
        while True:
            nt = _E(err, t, tau, cmax) + B + sum(_ceil(t + k.J, k.T) * k.C for k in hep)
            if nt == t or nt > cap:
                break
            t = nt
        if nt > cap:
            out[m.name] = Result(m.name, None, m.D, schedulable=False)
            continue
        Q = _ceil(t + m.J, m.T)
        Rs, w_prev, ok = [], None, True
        for q in range(Q):
            w = B + q * m.C if w_prev is None else w_prev + m.C
            while True:
                nw = (_E(err, w + m.C, tau, cmax) + B + q * m.C
                      + sum(_ceil(w + k.J + tau, k.T) * k.C for k in hp))
                if nw == w or nw > cap:
                    break
                w = nw
            if nw > cap:
                ok = False
                break
            w_prev = w
            Rs.append(m.J + w - q * m.T + m.C)
        R = max(Rs) if ok else None
        out[m.name] = Result(m.name, R, m.D, Q, t, R is not None and R <= m.D, Rs)
    return out


def analyse_legacy(msgs: list[Msg], tau: int) -> dict[str, Result]:
    """The original analysis ([DAVIS2007] Eq. 7): first instance only. Optimistic in general."""
    ms = _sorted(msgs)
    out = {}
    for i, m in enumerate(ms):
        hp, lp = ms[:i], ms[i + 1:]
        B = max((k.C for k in lp), default=0)
        w = B
        cap = 1000 * max(k.T for k in ms)
        while True:
            nw = B + sum(_ceil(w + k.J + tau, k.T) * k.C for k in hp)
            if nw == w or nw > cap:
                break
            w = nw
        R = None if nw > cap else m.J + w + m.C
        out[m.name] = Result(m.name, R, m.D, 1, 0, R is not None and R <= m.D, [R] if R else [])
    return out


def analyse_sufficient(msgs: list[Msg], tau: int) -> dict[str, Result]:
    """Sufficient (not necessary) test, [DAVIS2007] Eq. (16): blocking max(B_m, C_m)."""
    ms = _sorted(msgs)
    out = {}
    for i, m in enumerate(ms):
        hp, lp = ms[:i], ms[i + 1:]
        B = max(max((k.C for k in lp), default=0), m.C)
        w = B
        cap = 1000 * max(k.T for k in ms)
        while True:
            nw = B + sum(_ceil(w + k.J + tau, k.T) * k.C for k in hp)
            if nw == w or nw > cap:
                break
            w = nw
        R = None if nw > cap else m.J + w + m.C
        out[m.name] = Result(m.name, R, m.D, 1, 0, R is not None and R <= m.D, [R] if R else [])
    return out


def breakdown_utilisation(n_intermediate: int) -> float:
    """[DAVIS2007] section 4.2: lowest utilisation at which the legacy analysis is known to fail, for the
    constructed set H, n x I, X with C_H = C_X = 55 bit, C_I = 135 bit, C_L = 0."""
    ch = cx = 55
    ci = 135
    n = n_intermediate
    return 2 * cx / (3 * ch + 2 * n * ci + 2 * cx) + 2 * ch / (2 * ch + 2 * n * ci + cx)
