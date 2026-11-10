"""Lateral dynamics and the steering loops."""
import numpy as np
import pytest

from vnet import data, dynamics


@pytest.fixture(scope="module")
def veh():
    return dynamics.build(data.load("vehicle"))


def P(k):
    return data.values("vehicle", k)


@pytest.mark.req("SR-01")
def test_vehicle_understeers_and_is_stable(veh):
    assert 0.5 < dynamics.understeer_gradient_deg_per_g(veh) < 6
    for kmh in (20, 60, 120):
        A, _ = dynamics.state_matrices(veh, kmh / 3.6)
        assert np.all(np.linalg.eigvals(A).real < 0)
    assert 4.0 < dynamics.effective_wheelbase(veh) < veh.wheelbase


@pytest.mark.req("SR-01")
def test_yaw_delay_tolerance_shrinks_with_speed(veh):
    w = dynamics.freq_grid()
    t = [dynamics.margins(dynamics.loop_yaw(veh, k / 3.6, P("loop_yaw"), w), w).t30_s for k in (20, 60, 100, 120)]
    assert all(a > b for a, b in zip(t, t[1:]))
    assert 0.03 < t[-1] < 0.08


@pytest.mark.req("SR-02")
def test_actuator_loop_margin(veh):
    w = dynamics.freq_grid()
    m = dynamics.margins(dynamics.loop_actuator(P("loop_actuator"), w), w)
    assert 55 < m.pm_deg < 70
    assert abs(m.dm_s - np.radians(m.pm_deg) / m.wc) < 1e-9


@pytest.mark.req("MS-04")
def test_time_domain_confirms_delay_margin(veh):
    w = dynamics.freq_grid()
    u = 100 / 3.6
    m = dynamics.margins(dynamics.loop_yaw(veh, u, P("loop_yaw"), w), w)
    s = dynamics.unstable_delay(lambda **k: dynamics.simulate_yaw(veh, u, P("loop_yaw"), period_s=0.001, t_end=8, **k),
                                "r", 0.1, hi=0.5, tol=0.004)
    assert abs(s - m.dm_s) / m.dm_s < 0.04
    ok = dynamics.simulate_yaw(veh, u, P("loop_yaw"), delay_s=0.02, period_s=0.01)
    assert not ok["diverged"] and ok["late_osc"] < 0.05
