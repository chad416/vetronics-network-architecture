"""Lateral dynamics of a multi-axle wheeled vehicle and the three steering loops that close over the network.

Linear single-track ("bicycle") model generalised to n axles: every axle i at x_i (m, positive ahead of the
centre of gravity) has cornering stiffness C_i (N/rad, both tyres) and steer gain k_i (road-wheel angle per
command). With lateral velocity v, yaw rate r, forward speed u and small angles [WONG2008 ch. 5; RAJAMANI2012
ch. 2; CHALMERS2013 section 2]:

    m (v' + u r) = sum_i C_i (k_i d - (v + x_i r) / u)
    Iz r'        = sum_i x_i C_i (k_i d - (v + x_i r) / u)

The loops (DOSSIER section 8.2):
- L1 the steering-actuator position loop, only if it were closed across the network (central controller);
- L2 a yaw-rate stabilisation loop in the vehicle control unit (yaw rate from the inertial unit, steer
  correction to the actuators);
- L3 a path-following loop (pure pursuit [COULTER1992]) for remote or semi-autonomous driving.
For each, the open-loop frequency response gives the phase margin, the crossover and the delay margin
(PM / w_c); network latency enters as e^{-sT}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Vehicle:
    mass: float                  # kg
    iz: float                    # kg m^2
    x: tuple[float, ...]         # axle positions from the CG, m (+ ahead)
    C: tuple[float, ...]         # axle cornering stiffness, N/rad
    k: tuple[float, ...]         # steer gain per axle (0 = not steered)
    act_wn: float                # closed-loop actuator natural frequency, rad/s
    act_zeta: float

    @property
    def wheelbase(self) -> float:
        return max(self.x) - min(self.x)


def build(d: dict) -> Vehicle:
    """From data/vehicle.yaml: equal-ish axle loads, cornering stiffness = coefficient x axle load, steered
    axles reduced by the steering-compliance factor, second steered axle by the Ackermann ratio about the
    centre of the unsteered bogie."""
    pos = np.array(d["axle_pos_from_front_m"]["value"], float)
    share = np.array(d["axle_load_share"]["value"], float)
    m = d["mass_kg"]["value"]
    xcg = float((pos * share).sum() / share.sum())
    x = tuple(float(xcg - p) for p in pos)
    fz = m * G * share / share.sum()
    steered = d["steered_axles"]["value"]
    comp = d["steer_compliance_factor"]["value"]
    coef = d["cornering_coeff_per_rad"]["value"]
    C = tuple(float(coef * f * (comp if i in steered else 1.0)) for i, f in enumerate(fz))
    unst = [p for i, p in enumerate(pos) if i not in steered]
    xref = float(np.mean(unst))
    k = tuple(float((xref - p) / (xref - pos[0])) if i in steered else 0.0 for i, p in enumerate(pos))
    kgyr = d["yaw_radius_of_gyration_m"]["value"]
    return Vehicle(m, m * kgyr ** 2, x, C, k, 2 * math.pi * d["actuator_bandwidth_hz"]["value"],
                   d["actuator_damping"]["value"])


def state_matrices(v: Vehicle, u: float) -> tuple[np.ndarray, np.ndarray]:
    """x = [v, r], input = steer command (rad, first axle)."""
    C, x, k = np.array(v.C), np.array(v.x), np.array(v.k)
    a11 = -C.sum() / (v.mass * u)
    a12 = -(C * x).sum() / (v.mass * u) - u
    a21 = -(C * x).sum() / (v.iz * u)
    a22 = -(C * x * x).sum() / (v.iz * u)
    b1 = (C * k).sum() / v.mass
    b2 = (C * k * x).sum() / v.iz
    return np.array([[a11, a12], [a21, a22]]), np.array([b1, b2])


def understeer_gradient_deg_per_g(v: Vehicle, u: float = 20.0) -> float:
    """Steady-state steer angle per lateral acceleration minus the kinematic part, deg/g."""
    A, B = state_matrices(v, u)
    ss = -np.linalg.solve(A, B)          # [v, r] per rad of steer
    r = ss[1]
    ay = u * r                            # lateral acceleration per rad (steady state)
    # kinematic steer for this path: delta_kin = L_eff * r / u; L_eff from a very low speed run
    A0, B0 = state_matrices(v, 0.5)
    r0 = (-np.linalg.solve(A0, B0))[1] / 0.5   # yaw rate per (rad * m/s) at low speed = 1 / L_eff
    l_eff = 1.0 / r0
    delta_per_ay = 1.0 / ay
    kin_per_ay = l_eff / u ** 2
    return math.degrees((delta_per_ay - kin_per_ay) * G)


def yaw_gain_ss(v: Vehicle, u: float) -> float:
    """Steady-state yaw rate per rad of steer, 1/s."""
    A, B = state_matrices(v, u)
    return float((-np.linalg.solve(A, B))[1])


def effective_wheelbase(v: Vehicle) -> float:
    """Kinematic wheelbase of the multi-axle vehicle: u / (yaw-rate gain) at walking speed."""
    return 0.5 / yaw_gain_ss(v, 0.5)


def actuator_tf(v: Vehicle, s: np.ndarray) -> np.ndarray:
    wn, z = v.act_wn, v.act_zeta
    return wn ** 2 / (s ** 2 + 2 * z * wn * s + wn ** 2)


def yaw_tf(v: Vehicle, u: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lateral-velocity and yaw-rate responses to the steer angle of axle 1."""
    A, B = state_matrices(v, u)
    out_v, out_r = [], []
    for si in s:
        x = np.linalg.solve(si * np.eye(2) - A, B)
        out_v.append(x[0])
        out_r.append(x[1])
    return np.array(out_v), np.array(out_r)


@dataclass
class Margins:
    wc: float          # crossover, rad/s
    pm_deg: float      # phase margin without network delay
    dm_s: float        # delay margin = PM / wc
    t30_s: float       # delay that leaves 30 deg phase margin
    gm_db: float


def margins(L: np.ndarray, w: np.ndarray) -> Margins:
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))
    idx = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if len(idx) == 0:
        return Margins(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    best = None
    for i in idx:
        f = (0 - math.log(mag[i])) / (math.log(mag[i + 1]) - math.log(mag[i]))
        wc = math.exp(math.log(w[i]) + f * (math.log(w[i + 1]) - math.log(w[i])))
        p = ph[i] + f * (ph[i + 1] - ph[i])
        pm = math.degrees(p) + 180.0
        pm = (pm + 180.0) % 360.0 - 180.0
        dm = math.radians(pm) / wc
        if best is None or dm < best.dm_s:
            best = Margins(wc, pm, dm, math.radians(pm - 30.0) / wc, float("nan"))
    j = np.where((ph[:-1] > -math.pi) & (ph[1:] <= -math.pi))[0]
    if len(j):
        best.gm_db = float(-20 * math.log10(mag[j[0]]))
    return best


# --- the three loops -------------------------------------------------------------------------------
def loop_actuator(p: dict, w: np.ndarray) -> np.ndarray:
    """L1: valve-controlled cylinder (integrator with valve lag) under a P position controller,
    L = Kp K / (s (tau s + 1))."""
    s = 1j * w
    return p["kp_times_kv"] / (s * (p["valve_tau_s"] * s + 1))


def loop_yaw(v: Vehicle, u: float, p: dict, w: np.ndarray) -> np.ndarray:
    """L2: PI yaw-rate controller around actuator + vehicle. The gain is scheduled with speed so that the
    loop always crosses over at the design bandwidth wc (rad/s); what changes with speed is the phase the
    vehicle's own yaw dynamics contribute (they lose damping as speed rises)."""
    s = 1j * w
    _, r = yaw_tf(v, u, s)
    plant = (1 + 1 / (p["ti_s"] * s)) * actuator_tf(v, s) * r
    kp = 1.0 / abs(_plant_yaw(v, u, p, p["wc_design_rad_s"]))
    return kp * plant


def _plant_yaw(v: Vehicle, u: float, p: dict, wc: float) -> complex:
    s = np.array([1j * wc])
    _, r = yaw_tf(v, u, s)
    return complex(((1 + 1 / (p["ti_s"] * s)) * actuator_tf(v, s) * r)[0])


def loop_path(v: Vehicle, u: float, p: dict, w: np.ndarray) -> np.ndarray:
    """L3: pure pursuit, curvature = 2 e / Ld^2 with e the lateral error at the look-ahead Ld = max(Ld_min,
    t_la u); steer = L_eff * curvature. Linearised: e = y + Ld psi, y' = v + u psi, psi' = r."""
    s = 1j * w
    vv, r = yaw_tf(v, u, s)
    psi = r / s
    y = (vv + u * psi) / s
    ld = max(p["ld_min_m"], p["t_lookahead_s"] * u)
    l_eff = effective_wheelbase(v)
    return (2 * l_eff / ld ** 2) * (y + ld * psi) * actuator_tf(v, s)


def freq_grid() -> np.ndarray:
    return np.logspace(-2, 3, 6000)


# --- time domain: closed-loop lane change with a real transport delay and sample-and-hold -------------
def simulate_path(v: Vehicle, u: float, p: dict, delay_s: float, period_s: float = 0.01, t_end: float = 12.0,
                  dt: float = 0.0005, lane_m: float = 3.5) -> dict:
    """Pure pursuit tracking a 3.5 m lateral step (smoothed) at speed u. The steer command is computed every
    `period_s`, then delayed by `delay_s` (network + task latency) before the actuator sees it."""
    A, B = state_matrices(v, u)
    n = int(t_end / dt)
    xs = np.zeros(2)
    psi = y = 0.0
    a_pos = a_vel = 0.0
    wn, z = v.act_wn, v.act_zeta
    ld = max(p["ld_min_m"], p["t_lookahead_s"] * u)
    l_eff = effective_wheelbase(v)
    buf: list[tuple[float, float]] = []      # (time it reaches the actuator, command)
    cmd_now = 0.0
    next_sample = 0.0
    ys, ts, rs = [], [], []
    rate_lim = math.radians(p.get("rate_limit_deg_s", 40.0))
    for i in range(n):
        t = i * dt
        if t >= next_sample - 1e-12:
            x_ahead = u * t + ld
            yref = lane_m * 0.5 * (1 - math.cos(math.pi * min(1.0, max(0.0, (x_ahead - 20.0) / 60.0))))
            e = yref - (y + ld * psi)
            delta = l_eff * 2 * e / ld ** 2
            buf.append((t + delay_s, max(-0.6, min(0.6, delta))))
            next_sample += period_s
        while buf and buf[0][0] <= t + 1e-12:
            cmd_now = buf.pop(0)[1]
        acc = wn * wn * (cmd_now - a_pos) - 2 * z * wn * a_vel
        a_vel += acc * dt
        a_vel = max(-rate_lim, min(rate_lim, a_vel))
        a_pos += a_vel * dt
        dx = A @ xs + B * a_pos
        xs = xs + dx * dt
        psi += xs[1] * dt
        y += (xs[0] + u * psi) * dt
        if i % 20 == 0:
            ys.append(y)
            ts.append(t)
            rs.append(xs[1])
    ys_a = np.array(ys)
    late = ys_a[len(ys_a) // 2:]
    return {"t": np.array(ts), "y": ys_a, "r": np.array(rs),
            "overshoot_m": float(ys_a.max() - lane_m),
            "late_osc_m": float(late.max() - late.min()),
            "diverged": bool(np.abs(ys_a).max() > 20 or not np.isfinite(ys_a).all())}


def simulate_yaw(v: Vehicle, u: float, p: dict, delay_s: float, period_s: float = 0.01, t_end: float = 6.0,
                 dt: float = 0.0005, r_step: float = 0.1) -> dict:
    """Yaw-rate loop (L2) answering a 0.1 rad/s yaw-rate step, with the PI computed every `period_s` and its
    output delayed by `delay_s` before the actuator."""
    A, B = state_matrices(v, u)
    kp = 1.0 / abs(_plant_yaw(v, u, p, p["wc_design_rad_s"]))
    ti = p["ti_s"]
    wn, z = v.act_wn, v.act_zeta
    xs = np.zeros(2)
    a_pos = a_vel = integ = 0.0
    buf: list[tuple[float, float]] = []
    cmd = 0.0
    nxt = 0.0
    rs, ts = [], []
    for i in range(int(t_end / dt)):
        t = i * dt
        if t >= nxt - 1e-12:
            e = r_step - xs[1]
            integ += e * period_s
            buf.append((t + delay_s, max(-0.6, min(0.6, kp * (e + integ / ti)))))
            nxt += period_s
        while buf and buf[0][0] <= t + 1e-12:
            cmd = buf.pop(0)[1]
        acc = wn * wn * (cmd - a_pos) - 2 * z * wn * a_vel
        a_vel += acc * dt
        a_pos += a_vel * dt
        xs = xs + (A @ xs + B * a_pos) * dt
        if i % 10 == 0:
            rs.append(xs[1])
            ts.append(t)
    r = np.array(rs)
    late = r[int(len(r) * 0.6):]
    return {"t": np.array(ts), "r": r, "overshoot": float(r.max() / r_step - 1.0),
            "late_osc": float((late.max() - late.min()) / r_step),
            "diverged": bool(not np.isfinite(r).all() or np.abs(r).max() > 10 * r_step)}


def unstable_delay(sim, key: str, ref: float, lo: float = 0.0, hi: float = 1.0, tol: float = 0.002, **kw) -> float:
    """Smallest added delay (s) at which the time-domain loop is no longer asymptotically stable: the error
    envelope in the last quarter of the run is not smaller than in the quarter before (or the run diverges).
    This is the simulated counterpart of the frequency-domain delay margin."""
    def bad(d):
        r = sim(delay_s=d, **kw)
        if r["diverged"]:
            return True
        e = np.abs(r[key] - ref)
        n = len(e)
        a1 = e[n // 2: 3 * n // 4].max()
        a2 = e[3 * n // 4:].max()
        return a2 > 0.01 * abs(ref) and a2 >= 0.98 * a1
    if not bad(hi):
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if bad(mid) else (mid, hi)
    return hi
