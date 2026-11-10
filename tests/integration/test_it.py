"""Integration tests: the requirements of DOSSIER section 5 on the recommended architecture (A2) and the
baselines, through the analysis, the simulators, python-can virtual buses and the C code."""
import can
import cantools
import pytest

from vnet import aoa, data, ethernet, faults, latency, network, rma, studies
from vnet.e2e import Sender


@pytest.fixture(scope="module")
def r5():
    return studies.r5_loops(verify_speeds=())


@pytest.mark.it("IT01")
@pytest.mark.req("SR-10", "SR-11")
def test_a2_schedulable_with_growth_and_errors():
    _, buses = network.build_arch("A2", growth=True)
    for b in buses:
        rep = network.analyse_bus(b, network.error_model(b))
        assert not rep.misses, (b.id, rep.misses)
        assert rep.load <= data.load("architectures")["max_design_load"]["value"]


@pytest.mark.it("IT02")
@pytest.mark.req("SR-01", "SR-03", "SR-04")
def test_a2_chains_within_budget(r5):
    b = studies.budgets_from(r5)
    assert 20 < b["YAW"] < 30
    for c in latency.evaluate("A2", budgets=b):
        if c.chain != "ACT_CENTRAL":
            assert c.ok, (c.chain, c.total_ms, c.budget_ms)


@pytest.mark.it("IT03")
@pytest.mark.req("SR-02", "SR-14")
def test_baselines_fail_where_expected(r5):
    _, buses = network.build_arch("A0")
    rep = network.analyse_bus(buses[0])
    assert rep.load > 0.95 and len(rep.misses) >= 5               # by-wire at 100 Hz does not fit 250 kbit/s
    b = studies.budgets_from(r5)
    a0 = {c.chain: c for c in latency.evaluate("A0", budgets=b)}
    assert not a0["ACT_CENTRAL"].ok                               # a central position loop fails on the legacy bus
    a1 = {c.chain: c for c in latency.evaluate("A1", budgets=b)}
    assert a1["ACT_CENTRAL"].total_ms > 0.85 * a1["ACT_CENTRAL"].budget_ms   # and is marginal even at 500 kbit/s
    _, buses = network.build_arch("A1", growth=True, gw_jitter_ms=1.0)
    assert "NAV_POS" in network.analyse_bus(buses[0]).misses     # gatewayed loop data inherits jitter and misses


@pytest.mark.it("IT04")
@pytest.mark.req("SR-30", "SR-32", "SR-50", "SR-51", "SR-53")
def test_dual_channel_run_on_virtual_buses():
    run = faults.redundancy_run(babble_at=250, open_a_at=500, corrupt_b_at=750)
    st = dict(zip(run.t_ms, run.state))
    assert st[240] == "DUAL" and st[400] == "DUAL"                 # the babbler was latched off before A timed out
    assert run.guard_tripped_at is not None and run.guard_tripped_at - 250 < 10
    assert st[600] == "SINGLE_B"
    healthy = [a for t, s, a in zip(run.t_ms, run.state, run.age_ms) if s in ("DUAL", "SINGLE_B") and 20 < t < 750]
    assert max(healthy) <= 11                                      # one period + jitter while a channel is healthy
    assert run.lost_detected_at is not None and 750 < run.lost_detected_at <= 750 + 30 + 1


@pytest.mark.it("IT05")
@pytest.mark.req("SR-53")
def test_guardian_rate_is_safe():
    _, buses = network.build_arch("A2", growth=True)
    b = buses[0]
    thr = faults.babbler_threshold(b.rta_msgs(), b.tau_ns)
    assert thr < 10.0                     # a node held to its own 10 ms rate cannot starve the others
    _, buses = network.build_arch("A0")
    assert faults.babbler_threshold(buses[0].rta_msgs(), buses[0].tau_ns) == float("inf")


@pytest.mark.it("IT06")
@pytest.mark.req("SR-52")
def test_dbc_and_virtual_bus_round_trip():
    db = cantools.database.load_file(str(data.ROOT / "docs" / "vehicle.dbc"))
    m = db.get_message_by_name("STEER_CMD")
    assert m.frame_id == 0x04EFFF27 and m.is_extended_frame
    payload = m.encode({"E2E_CRC": 0, "E2E_Counter": 0, "Mode": 1, "AngleAx1": -12.5, "AngleAx2": -8.2, "RateLimit": 25.0})
    pdu = Sender(0x0EF0).protect(bytes(payload))
    a = can.Bus(interface="virtual", channel="it06", receive_own_messages=False)
    b = can.Bus(interface="virtual", channel="it06", receive_own_messages=False)
    try:
        a.send(can.Message(arbitration_id=m.frame_id, data=pdu, is_extended_id=True))
        rx = b.recv(timeout=1.0)
    finally:
        a.shutdown()
        b.shutdown()
    dec = db.decode_message(rx.arbitration_id, rx.data)
    assert abs(dec["AngleAx1"] + 12.5) < 0.01 and dec["Mode"] == 1


@pytest.mark.it("IT07")
@pytest.mark.req("SR-20", "SR-21")
def test_backbone_deadlines():
    net = ethernet.build(1000)
    b = ethernet.bounds(net)
    for f in net.flows:
        assert b[f.name] <= f.deadline_s, f.name
    net100 = ethernet.build(100)
    b100 = ethernet.bounds(net100)
    assert any(b100[f.name] > f.deadline_s for f in net100.flows)   # 100 Mbit/s cannot carry video + C4I


@pytest.mark.it("IT08")
@pytest.mark.req("MS-03")
def test_sources_resolve():
    assert data.check_sources() == []


@pytest.mark.it("IT09")
@pytest.mark.req("SR-13")
def test_physical_layer_limits():
    r = studies.r11_phy()
    harness = r["harness_m"]
    j = r["j1939"]
    assert harness <= j["J1939_14"]["max_bus_m"] and harness <= j["J1939_11"]["max_bus_m"]
    lim = {x["bps"]: x["arbitration_limit_m"] for x in r["rows"]}
    assert lim[500000] > harness and 30 < lim[1000000] < 45        # the classic "40 m at 1 Mbit/s"
    _, buses = network.build_arch("A2")
    for b in buses:
        assert len({m.sa for m in b.msgs}) <= 30


@pytest.mark.it("IT10")
@pytest.mark.req("SR-30", "SR-31")
def test_trade_study_and_reliability():
    p1, p2 = rma.path_reliability("A1"), rma.path_reliability("A2")
    assert p2["F_mission"] < p1["F_mission"] / 10
    assert p2["F_mission"] / 12 < 1e-7                              # per hour, below the SIL 3 continuous band
    r3, r9 = studies.r3_rta(), studies.r9_rma()
    r6 = studies.r6_chains(studies.r5_loops(verify_speeds=()))
    r7 = studies.r7_ethernet()
    r10 = studies.r10_aoa(r3, r6, r7, r9)
    assert r10["screen"]["A0"] and r10["screen"]["A1"] and not r10["screen"]["A2"]
    assert r10["leader"] == "A2"
    assert set(aoa.weights()) == set(r10["weights"])
