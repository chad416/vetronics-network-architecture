"""Network fault behaviour: CAN fault confinement timing, babbling-idiot impact, and the dual-channel
redundancy run on python-can virtual buses with the C redundancy manager and bus guardian in the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import data
from .canframe import worst_case_bits
from .rta import Msg, analyse


# --- ISO 11898-1 / CAN 2.0 fault confinement ---------------------------------------------------------
@dataclass
class ConfinementResult:
    frames_to_passive: int
    frames_to_bus_off: int
    t_bus_off_ms: float
    t_recover_min_ms: float
    t_recover_app_ms: float


def confinement(bps: int, n_bytes: int = 8) -> ConfinementResult:
    """A transmitter whose every frame fails (e.g. its transceiver cannot drive dominant): TEC +8 per error
    [CAN20 rule 3], error passive at 128, bus-off at 256 [rules 9-10]. Each attempt costs at most the frame
    plus a 31-bit error frame. Recovery needs 128 x 11 recessive bits [rule 12] plus the application delay."""
    fc = data.values("environment", "fault_confinement")
    tau_ms = 1e3 / bps
    attempt = (worst_case_bits(n_bytes) + 31) * tau_ms
    n_pass = -(-fc["error_passive_at"] // fc["tec_tx_error"])
    n_off = -(-fc["bus_off_at"] // fc["tec_tx_error"])
    rec = fc["recovery_sequences"] * fc["recovery_bits_each"] * tau_ms
    return ConfinementResult(n_pass, n_off, n_off * attempt, rec, rec + fc["app_recovery_delay_ms"])


def tec_walk(error_prob: float, n_frames: int, seed: int = 0) -> list[int]:
    """TEC trajectory of a transmitter with independent per-frame error probability (rules 3 and 7)."""
    import random
    fc = data.values("environment", "fault_confinement")
    rng = random.Random(seed)
    tec, out = 0, []
    for _ in range(n_frames):
        if rng.random() < error_prob:
            tec += fc["tec_tx_error"]
        else:
            tec = max(0, tec + fc["tec_tx_ok"])
        out.append(tec)
        if tec >= fc["bus_off_at"]:
            break
    return out


def babbler_threshold(msgs: list[Msg], tau: int, lo_ms: float = 0.05, hi_ms: float = 100.0) -> float:
    """Shortest period (ms) at which a node babbling at the highest identifier still leaves every message
    schedulable: what a transmit guardian must enforce. Bisection on the RTA."""
    C = worst_case_bits(8) * tau
    base = {k for k, r in analyse(msgs, tau).items() if r.schedulable}

    def ok(p_ms: float) -> bool:
        b = Msg("BABBLER", -1, C, round(p_ms * 1e6), round(p_ms * 1e6))
        res = analyse([b] + msgs, tau)
        return all(res[k].schedulable for k in base)
    if not ok(hi_ms):
        return float("inf")
    lo, hi = lo_ms, hi_ms
    while hi - lo > 0.01:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


# --- dual-channel run on python-can virtual buses ---------------------------------------------------
@dataclass
class RedundancyRun:
    t_ms: list[int] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    age_ms: list[int] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    guard_tripped_at: int | None = None
    lost_detected_at: int | None = None


def redundancy_run(period_ms: int = 10, t_end_ms: int = 1000, babble_at: int = 250, open_a_at: int = 500,
                   corrupt_b_at: int = 750, timeout_periods: int | None = None, seed: int = 0) -> RedundancyRun:
    """Steer command sent every period on channels A and B (python-can virtual buses). Faults:
    - babble_at: a node on A starts sending identifier 0 back-to-back; its bus guardian (C) latches it off;
    - open_a_at: channel A harness open, nothing more arrives on A;
    - corrupt_b_at: channel B frames arrive with one flipped bit (EMI) from then on.
    The actuator side feeds every received copy to the C redundancy manager and ticks it every 1 ms."""
    import random

    import can

    from .e2e import BusGuardian, RedundancyManager, Sender

    env = data.values("environment", "e2e")
    tp = timeout_periods or env["timeout_periods"]
    rng = random.Random(seed)
    data_id = 0x0EF0
    tx = Sender(data_id)
    rm = RedundancyManager(data_id, timeout_ms=tp * period_ms, max_delta=env["max_delta_counter"])
    guard = BusGuardian(rate_per_s=1000 // period_ms, depth=2, trip_after=20)
    names = (f"bwA-{seed}-{id(rm)}", f"bwB-{seed}-{id(rm)}")
    vcu = [can.Bus(interface="virtual", channel=n, receive_own_messages=False) for n in names]
    sas = [can.Bus(interface="virtual", channel=n, receive_own_messages=False) for n in names]
    out = RedundancyRun()
    try:
        for t in range(t_end_ms):
            babbling = babble_at <= t and not guard.tripped
            if t == babble_at:
                out.events.append((t, "A: babbling node starts"))
            if babbling:
                for _ in range(5):                       # ~5 frames per ms is what a 500 kbit/s bus can carry
                    if guard.allow(t):
                        vcu[0].send(can.Message(arbitration_id=0, data=bytes(8), is_extended_id=True))
                if guard.tripped and out.guard_tripped_at is None:
                    out.guard_tripped_at = t
                    out.events.append((t, "A: guardian latches the babbler off"))
            if t == open_a_at:
                out.events.append((t, "A: harness open"))
            if t == corrupt_b_at:
                out.events.append((t, "B: EMI corrupts every frame"))
            if t % period_ms == 0:
                angle = int(1000 * rng.uniform(-1, 1)) & 0xFFFF
                pdu = tx.protect(bytes([0, 0, angle & 0xFF, angle >> 8, 0, 0, 0, 0]))
                a_blocked = babbling or t >= open_a_at
                if not a_blocked:
                    vcu[0].send(can.Message(arbitration_id=0x04EFFF27, data=pdu, is_extended_id=True))
                bpdu = bytearray(pdu)
                if t >= corrupt_b_at:
                    bpdu[rng.randrange(8)] ^= 1 << rng.randrange(8)
                vcu[1].send(can.Message(arbitration_id=0x04EFFF27, data=bytes(bpdu), is_extended_id=True))
            for ch in (0, 1):
                while True:
                    m = sas[ch].recv(timeout=0)
                    if m is None:
                        break
                    if m.arbitration_id == 0x04EFFF27:
                        rm.on_frame(ch, bytes(m.data), t)
            st = rm.tick(t)
            if st == "LOST" and out.lost_detected_at is None:
                out.lost_detected_at = t
                out.events.append((t, "both channels lost: safe-stop request"))
            out.t_ms.append(t)
            out.state.append(st)
            out.age_ms.append(min(rm.age(t), 10_000))
    finally:
        for b in vcu + sas:
            b.shutdown()
    return out
