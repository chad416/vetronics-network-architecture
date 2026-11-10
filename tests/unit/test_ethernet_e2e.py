"""Ethernet bounds, the C E2E library through ctypes, and fault confinement."""
import random

import pytest

from vnet import e2e, ethernet, faults
from vnet.ethernet import WIRE_OVERHEAD, Flow, Network


@pytest.mark.req("MS-02")
def test_two_class_bound_by_hand():
    C = 100e6
    net = Network({"P": C}, 0.0, [Flow("H", ["P"], 6, period_s=0.01, msg_bytes=200),
                                   Flow("L", ["P"], 0, rate_bps=1e6, burst_bytes=1500)])
    b = ethernet.bounds(net)
    lh = 200 + WIRE_OVERHEAD
    ll = 1500 + WIRE_OVERHEAD
    assert abs(b["H"] - (ll + lh) / (C / 8)) < 1e-12          # Prop. 1.3.4 + Thm 1.4.2: (l_max of L + b_H) / C
    rH = lh / 0.01
    assert abs(b["L"] - (lh + ll) / (C / 8 - rH)) < 1e-12


@pytest.mark.req("MS-02", "SR-20")
def test_simulation_never_exceeds_bound():
    net = ethernet.build(1000, bywire=True)
    b = ethernet.bounds(net)
    for kw in ({"seed": 3}, {"aligned": True}):
        s = ethernet.simulate(net, t_end=0.5, **kw)
        for k in b:
            assert s[k] <= b[k] + 1e-9


def _py_crc8(d, c=0):
    for x in d:
        c ^= x
        for _ in range(8):
            c = ((c << 1) ^ 0x1D) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


@pytest.mark.req("SR-52")
def test_c_crc_matches_reference_and_catalogue():
    assert e2e.crc8(b"123456789") == 0x37                      # CRC-8/GSM-A check value [CRCCAT]
    rng = random.Random(2)
    for _ in range(200):
        d = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 8)))
        assert e2e.crc8(d) == _py_crc8(d)


@pytest.mark.req("SR-52")
def test_e2e_detects_corruption_repetition_masquerade():
    tx, rx, wrong = e2e.Sender(0x0EF0), e2e.Receiver(0x0EF0), e2e.Receiver(0x0EF1)
    p = tx.protect(bytes([0, 0, 1, 2, 3, 4, 5, 6]))
    assert rx.check(p) == "INITIAL" and wrong.check(p) == "ERROR"
    assert rx.check(p) == "REPEATED"
    q = bytearray(tx.protect(bytes([0, 0, 9, 9, 9, 9, 9, 9])))
    q[5] ^= 0x10
    assert rx.check(bytes(q)) == "ERROR"
    counters = {tx.protect(bytes(8))[1] & 0xF for _ in range(30)}
    assert counters == set(range(15))


@pytest.mark.req("SR-50", "SR-51")
def test_redundancy_manager_states():
    tx, rm = e2e.Sender(5), e2e.RedundancyManager(5, timeout_ms=30)
    for k in range(20):
        p = tx.protect(bytes(8))
        rm.on_frame(0, p, 10 * k)
        rm.on_frame(1, p, 10 * k + 1)
    assert rm.tick(195) == "DUAL" and rm.s.duplicates == 20
    for k in range(20, 30):
        rm.on_frame(1, tx.protect(bytes(8)), 10 * k)
    assert rm.tick(295) == "SINGLE_B" and rm.age(295) <= 10
    assert rm.tick(335) == "LOST"


@pytest.mark.req("SR-53")
def test_guardian_latches_babbler():
    g = e2e.BusGuardian(rate_per_s=100, depth=2, trip_after=20)
    assert sum(g.allow(t) for t in range(0, 1000, 10)) >= 99 and not g.tripped
    for _ in range(50):
        g.allow(1000)
    assert g.tripped and not g.allow(5000)


@pytest.mark.req("SR-54")
def test_fault_confinement_timing():
    c = faults.confinement(500000)
    assert c.frames_to_passive == 16 and c.frames_to_bus_off == 32
    assert abs(c.t_recover_min_ms - 128 * 11 * 0.002) < 1e-9
    assert max(faults.tec_walk(0.0, 1000)) == 0
    assert len(faults.tec_walk(1.0, 1000)) == 32
