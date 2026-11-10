"""CAN frame, analysis and simulator against the published results."""
import random

import pytest

from vnet import cansim, network
from vnet.canframe import crc15, on_bus_bits, stuff, worst_case_bits
from vnet.j1939 import make_id, parse_id, tp_packets
from vnet.rta import ErrorModel, Msg, analyse, analyse_legacy, analyse_sufficient, breakdown_utilisation

MS = 1_000_000


def _bits(s: bytes):
    return [int(x) for ch in s for x in format(ch, "08b")]


@pytest.mark.req("MS-01")
def test_crc15_check_value():
    assert crc15(_bits(b"123456789")) == 0x059E          # CRC-15/CAN check value [CRCCAT]


@pytest.mark.req("MS-01")
def test_stuffing_rule():
    assert stuff([0] * 5) == [0] * 5 + [1]
    assert stuff([1] * 10) == [1] * 5 + [0] + [1] * 5 + [0]
    assert stuff([0, 0, 0, 0, 0, 1, 1, 1, 1]) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]   # stuff bit starts a new run


@pytest.mark.req("MS-01")
def test_frame_length_bound_davis_eq1():
    assert worst_case_bits(8, True) == 160 and worst_case_bits(8, False) == 135 and worst_case_bits(7, False) == 125
    rng = random.Random(0)
    for _ in range(3000):
        n = rng.randint(0, 8)
        ext = rng.random() < 0.5
        cid = rng.getrandbits(29 if ext else 11)
        d = bytes(rng.getrandbits(8) for _ in range(n))
        assert on_bus_bits(cid, d, ext) <= worst_case_bits(n, ext)
    # the minimum is the unstuffed frame: 47 + 8n (standard) / 67 + 8n (extended) including 3-bit IFS
    assert on_bus_bits(0x555, bytes([0x55] * 8), False) >= 47 + 64


@pytest.mark.req("MS-01", "SR-12")
def test_davis_table2_example():
    tau = 8000
    msgs = [Msg("A", 1, MS, 2_500_000, 2_500_000), Msg("B", 2, MS, 3_500_000, 3_250_000), Msg("C", 3, MS, 3_500_000, 3_250_000)]
    old = analyse_legacy(msgs, tau)
    new = analyse(msgs, tau)
    assert [old[k].R for k in "ABC"] == [2 * MS, 3 * MS, 3 * MS]          # published legacy results
    assert new["C"].R == 3_500_000 and new["C"].busy == 7 * MS and new["C"].Q == 2
    assert not new["C"].schedulable and old["C"].schedulable           # the flaw: legacy says schedulable
    suf = analyse_sufficient(msgs, tau)
    assert all(suf[k].R >= new[k].R for k in "ABC")


@pytest.mark.req("MS-01", "SR-12")
def test_davis_section41_construction_and_breakdown():
    tau = 1000
    ch, ci, cx, cl, n = 55, 135, 80, 40, 2
    TH = (cl + 2 * ch + 2 * n * ci + cx) * tau // 2
    TX = (cl + 3 * ch + 2 * n * ci + 2 * cx) * tau // 2
    TI = 100 * TX
    JI = TI - (cl + 2 * ch + n * ci + cx) * tau
    msgs = [Msg("H", 1, ch * tau, TH, TH)] + [Msg(f"I{i}", 2 + i, ci * tau, TI, TI, JI) for i in range(n)] + \
           [Msg("X", 10, cx * tau, TX, TX), Msg("L", 11, cl * tau, 1000 * TX, 1000 * TX)]
    r = analyse(msgs, tau)["X"]
    assert r.Q == 2 and r.per_instance[1] > r.per_instance[0]
    assert abs(r.R - TX) <= 2 * tau                                          # "only just schedulable, R_X = T_X"
    pub = {1: 45.5, 3: 21.4, 8: 9.2, 23: 3.4, 98: 0.82}
    for k, v in pub.items():
        assert abs(100 * breakdown_utilisation(k) - v) < 0.06


@pytest.mark.req("SR-10")
def test_error_model_only_lengthens():
    _, buses = network.build_arch("A2", growth=True)
    b = buses[0]
    a = analyse(b.rta_msgs(), b.tau_ns)
    e = analyse(b.rta_msgs(), b.tau_ns, ErrorModel(2, 10 * MS))
    assert all(e[k].R >= a[k].R for k in a)


@pytest.mark.req("MS-02")
def test_critical_instant_reproduces_analysis():
    _, buses = network.build_arch("A2", growth=True)
    b = buses[0]
    rep = network.analyse_bus(b)
    sm = [cansim.SimMsg(m.name, m.can_id, m.n_bytes, round(m.T_ms * MS), round(m.J_ms * MS)) for m in b.msgs]
    for m in sm:
        ci = cansim.critical_instant(sm, m.name, b.tau_ns, 200 * MS)
        R = rep.results[m.name].R
        assert ci <= R
        assert R - ci <= 2 * worst_case_bits(8) * b.tau_ns                 # within two frames of the bound


@pytest.mark.req("MS-02")
def test_random_phasing_below_bound():
    _, buses = network.build_arch("A1")
    b = buses[0]
    rep = network.analyse_bus(b)
    rng = random.Random(5)
    sm = [cansim.SimMsg(m.name, m.can_id, m.n_bytes, round(m.T_ms * MS), round(m.J_ms * MS), offset=rng.randrange(round(m.T_ms * MS)))
          for m in b.msgs]
    r = cansim.simulate(sm, b.tau_ns, 400 * MS, seed=1)
    for k, v in r.max_R.items():
        if rep.results[k].R is not None:
            assert v <= rep.results[k].R


def test_simulated_error_lengthens_frame():
    m = [cansim.SimMsg("A", 1, 8, 10 * MS)]
    a = cansim.simulate(m, 2000, 9 * MS, worst_case=True)
    e = cansim.simulate(m, 2000, 9 * MS, worst_case=True, error_times=[100_000])
    assert e.max_R["A"] - a.max_R["A"] == 100_000 + 31 * 2000 and e.errors == 1


@pytest.mark.req("SR-10")
def test_j1939_identifier():
    cid = make_id(3, 0xF004, 0x00)
    assert cid == 0x0CF00400                                  # EEC1 from engine #1, priority 3
    p = parse_id(cid)
    assert (p.priority, p.pgn, p.sa, p.da) == (3, 0xF004, 0, None)
    cid1 = make_id(6, 0xEA00, 0xF9, da=0x00)                  # PDU1 request to address 0
    assert parse_id(cid1).pgn == 0xEA00 and parse_id(cid1).da == 0
    assert make_id(0, 0xFFFF, 0xFF) < make_id(1, 0x0000, 0x00)   # priority bits dominate arbitration
    assert tp_packets(1785) == 255
    with pytest.raises(ValueError):
        make_id(3, 0xEA05, 1)
