"""All studies of the dossier. Each returns a JSON-serialisable dict; `run_all` writes results/R*.json."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from . import aoa, cansim, data, dynamics, ethernet, faults, latency, network, rma
from .canframe import on_bus_bits, worst_case_bits
from .rta import ErrorModel, Msg, analyse, analyse_legacy, analyse_sufficient, breakdown_utilisation

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHS = ("A0", "A1", "A2", "A3", "A4")


def _ms(ns):
    return None if ns is None else ns / 1e6


# R1 ------------------------------------------------------------------------------------------------
def r1_benchmark() -> dict:
    """[DAVIS2007] Table 2 example and section 4.2 breakdown utilisations, reproduced."""
    tau = 8000                 # 125 kbit/s
    ms = 1_000_000
    msgs = [Msg("A", 1, ms, 2_500_000, 2_500_000), Msg("B", 2, ms, 3_500_000, 3_250_000), Msg("C", 3, ms, 3_500_000, 3_250_000)]
    new = analyse(msgs, tau)
    old = analyse_legacy(msgs, tau)
    suf = analyse_sufficient(msgs, tau)
    sim_msgs = [cansim.SimMsg(m.name, m.prio, 7, m.T, 0, m.D, extended=False) for m in msgs]
    sim = {m.name: cansim.critical_instant(sim_msgs, m.name, tau, 30 * ms) for m in msgs}
    published = {"legacy": {"A": 2.0, "B": 3.0, "C": 3.0}, "revised": {"C": 3.5}, "busy_C": 7.0, "Q_C": 2,
                 "breakdown_pct": {3: 45.5, 5: 21.4, 10: 9.2, 25: 3.4, 100: 0.82}}
    brk = {n + 2: round(100 * breakdown_utilisation(n), 2) for n in (1, 3, 8, 23, 98)}
    return {"frame_bits_7B_std": worst_case_bits(7, False), "published": published,
            "legacy_ms": {k: _ms(v.R) for k, v in old.items()}, "revised_ms": {k: _ms(v.R) for k, v in new.items()},
            "sufficient_ms": {k: _ms(v.R) for k, v in suf.items()},
            "busy_C_ms": _ms(new["C"].busy), "Q_C": new["C"].Q, "sim_critical_instant_ms": {k: _ms(v) for k, v in sim.items()},
            "breakdown_pct": brk}


# R2 ------------------------------------------------------------------------------------------------
def r2_frames(n: int = 20000, seed: int = 3) -> dict:
    rng = random.Random(seed)
    out = {}
    for ext in (False, True):
        for nb in (0, 2, 4, 8):
            lens = []
            for _ in range(n // 8):
                cid = rng.getrandbits(29 if ext else 11)
                d = bytes(rng.getrandbits(8) for _ in range(nb))
                lens.append(on_bus_bits(cid, d, ext))
            out[f"{'ext' if ext else 'std'}_{nb}B"] = {"bound": worst_case_bits(nb, ext), "max": max(lens),
                                                       "mean": float(np.mean(lens)), "p99": float(np.percentile(lens, 99))}
    # payload that reaches the bound: alternating 5-runs force a stuff bit every 4 bits after the first
    return out


# R3 ------------------------------------------------------------------------------------------------
def r3_rta() -> dict:
    out = {}
    for a in ARCHS:
        arch = data.load("architectures")["architectures"][a]
        gw_j = ethernet.bounds(ethernet.build(arch["ethernet_mbps"], bywire=arch.get("bywire_on_ethernet", False)))
        gw_j = gw_j.get("GW_CAN", 0.0) * 1e3
        rows = []
        for growth in (False, True):
            _, buses = network.build_arch(a, growth=growth, gw_jitter_ms=gw_j)
            for b in buses:
                for errs in (False, True):
                    rep = network.analyse_bus(b, network.error_model(b) if errs else ErrorModel())
                    bw = {k: _ms(r.R) for k, r in rep.results.items()
                          if k in ("STEER_CMD", "DRIVER_IN", "BRAKE_CMD", "INS_DYN", "SAS1_STAT", "NAV_POS")}
                    rows.append({"bus": b.id, "kind": b.kind, "bps": b.bps, "dual": b.dual, "growth": growth, "errors": errs,
                                 "n": rep.n, "load": rep.load, "misses": rep.misses, "worst_slack": rep.worst_slack,
                                 "R_ms": bw,
                                 "all": {k: [_ms(r.R), _ms(r.D)] for k, r in rep.results.items()}})
        out[a] = {"name": arch["name"], "rows": rows}
    return out


# R4 ------------------------------------------------------------------------------------------------
def _sim_set(bus: network.Bus) -> list[cansim.SimMsg]:
    return [cansim.SimMsg(m.name, m.can_id, m.n_bytes, round(m.T_ms * 1e6), round(m.J_ms * 1e6), round(m.T_ms * 1e6))
            for m in bus.msgs]


def r4_sim_vs_analysis(seeds: int = 6, t_end_ms: int = 2000) -> dict:
    out = {}
    for a, gi in (("A2", True), ("A1", False)):
        _, buses = network.build_arch(a, growth=gi)
        bus = buses[0]
        rep = network.analyse_bus(bus)
        sm = _sim_set(bus)
        ci = {m.name: cansim.critical_instant(sm, m.name, bus.tau_ns, 200_000_000) for m in sm}
        rand: dict[str, int] = {m.name: 0 for m in sm}
        for s in range(seeds):
            rng = random.Random(100 + s)
            for m in sm:
                m.offset = rng.randrange(0, m.T)
            r = cansim.simulate(sm, bus.tau_ns, t_end_ms * 1_000_000, seed=s)
            for k, v in r.max_R.items():
                rand[k] = max(rand[k], v)
        rows = []
        for m in sorted(sm, key=lambda x: x.can_id):
            R = rep.results[m.name].R
            rows.append({"msg": m.name, "bound_ms": _ms(R), "critical_instant_ms": _ms(ci[m.name]),
                         "random_max_ms": _ms(rand[m.name])})
        out[f"{a}_{bus.id}{'_growth' if gi else ''}"] = {"bps": bus.bps, "rows": rows}
    return out


# R5 ------------------------------------------------------------------------------------------------
def _params():
    d = data.load("vehicle")
    v = dynamics.build(d)
    P = {k: data.values("vehicle", k) for k in ("loop_actuator", "loop_yaw", "loop_path")}
    return d, v, P


def r5_loops(verify_speeds=(20, 100, 120)) -> dict:
    d, v, P = _params()
    w = dynamics.freq_grid()
    speeds = d["speeds_kmh"]["value"]
    m1 = dynamics.margins(dynamics.loop_actuator(P["loop_actuator"], w), w)
    rows = []
    for kmh in speeds:
        u = kmh / 3.6
        A, _ = dynamics.state_matrices(v, u)
        ev = np.linalg.eigvals(A)
        zeta = float(-ev[0].real / abs(ev[0]))
        m2 = dynamics.margins(dynamics.loop_yaw(v, u, P["loop_yaw"], w), w)
        m3 = dynamics.margins(dynamics.loop_path(v, u, P["loop_path"], w), w)
        rows.append({"kmh": kmh, "yaw_damping": zeta, "yaw_gain_ss": dynamics.yaw_gain_ss(v, u),
                     "yaw": m2.__dict__, "path": m3.__dict__})
    ver = []
    lin = dict(P["loop_path"], rate_limit_deg_s=1e4)
    for kmh in verify_speeds:
        u = kmh / 3.6
        m2 = dynamics.margins(dynamics.loop_yaw(v, u, P["loop_yaw"], w), w)
        s2 = dynamics.unstable_delay(lambda **k: dynamics.simulate_yaw(v, u, P["loop_yaw"], period_s=0.001, t_end=10, **k),
                                     "r", 0.1, hi=0.6, tol=0.004)
        m3 = dynamics.margins(dynamics.loop_path(v, u, P["loop_path"], w), w)
        s3 = dynamics.unstable_delay(lambda **k: dynamics.simulate_path(v, u, lin, period_s=0.001, t_end=40, dt=0.001, **k),
                                     "y", 3.5, hi=2.0, tol=0.02)
        ver.append({"kmh": kmh, "yaw_dm_ms": m2.dm_s * 1e3, "yaw_sim_ms": s2 * 1e3, "path_dm_ms": m3.dm_s * 1e3,
                    "path_sim_ms": s3 * 1e3})
    return {"vehicle": {"mass": v.mass, "iz": v.iz, "x": v.x, "C": v.C, "k": v.k, "wheelbase": v.wheelbase,
                        "l_eff": dynamics.effective_wheelbase(v), "Kus_deg_per_g": dynamics.understeer_gradient_deg_per_g(v)},
            "actuator_loop": m1.__dict__, "speeds": rows, "verification": ver}


def budgets_from(r5: dict) -> dict[str, float]:
    share = data.load("chains")["budget_share"]["value"]
    vmax = data.load("chains")["design_speed_kmh"]["value"]
    row = next(r for r in r5["speeds"] if r["kmh"] == vmax)
    return {"YAW": share * row["yaw"]["t30_s"] * 1e3, "ACT_CENTRAL": share * r5["actuator_loop"]["t30_s"] * 1e3,
            "PATH": share * row["path"]["t30_s"] * 1e3}


# R6 ------------------------------------------------------------------------------------------------
def r6_chains(r5: dict) -> dict:
    budgets = budgets_from(r5)
    out = {"budgets_ms": budgets, "archs": {}}
    for a in ARCHS:
        res = latency.evaluate(a, growth=True, errors=True, budgets=budgets)
        out["archs"][a] = [{"chain": c.chain, "total_ms": c.total_ms, "budget_ms": c.budget_ms, "ok": c.ok,
                            "hops": c.hops} for c in res]
    # how much of the yaw chain is the VCU period (A2)?
    a2 = next(c for c in out["archs"]["A2"] if c["chain"] == "YAW")
    out["A2_yaw_split_ms"] = {"bus": sum(v for k, v in a2["hops"] if k in ("INS_DYN", "STEER_CMD")),
                              "tasks": sum(v for k, v in a2["hops"] if k in ("VCU", "SAS"))}
    sweep, loads = {}, {}
    fast = ("INS_DYN", "STEER_CMD", "BRAKE_CMD", "DRIVER_IN", "SAS1_STAT", "SAS2_STAT")
    for a in ("A1", "A2", "A3"):
        sweep[a], loads[a] = {}, {}
        for per in (2, 5, 10, 20):
            c = next(x for x in latency.evaluate(a, True, True, budgets, {"VCU": per}) if x.chain == "YAW")
            sweep[a][str(per)] = c.total_ms
            _, buses = network.build_arch(a, growth=True, periods={k: per for k in fast})
            bw = buses[0]
            rep = network.analyse_bus(bw, network.error_model(bw))
            loads[a][str(per)] = {"load": rep.load, "misses": len(rep.misses)}
    out["vcu_period_sweep_yaw_ms"] = sweep
    out["vcu_period_sweep_bw_load"] = loads
    return out


# R7 ------------------------------------------------------------------------------------------------
def r7_ethernet() -> dict:
    out = {}
    for rate, bw in ((100, False), (1000, False), (1000, True)):
        net = ethernet.build(rate, bywire=bw)
        b = ethernet.bounds(net)
        s = ethernet.simulate(net, t_end=1.0, seed=1)
        al = ethernet.simulate(net, t_end=1.0, aligned=True)
        load = {}
        for p in net.ports:
            load[p] = sum(f.token_bucket()[0] for f in net.flows if p in f.path) * 8 / net.ports[p]
        key = f"{rate}M" + ("_bywire" if bw else "")
        out[key] = {"flows": [{"flow": f.name, "pcp": f.pcp, "bound_ms": b[f.name] * 1e3, "sim_ms": s[f.name] * 1e3,
                               "aligned_ms": al[f.name] * 1e3, "deadline_ms": f.deadline_s * 1e3} for f in net.flows],
                    "max_port_load": max(load.values()), "port_load": load}
    return out


# R8 ------------------------------------------------------------------------------------------------
def r8_faults() -> dict:
    conf = {bps: faults.confinement(bps).__dict__ for bps in (250000, 500000)}
    thr = {}
    for a in ("A1", "A2"):
        _, buses = network.build_arch(a, growth=True)
        b = buses[0]
        thr[a] = faults.babbler_threshold(b.rta_msgs(), b.tau_ns)
    run = faults.redundancy_run()
    segs, prev = [], None
    for t, s in zip(run.t_ms, run.state):
        if s != prev:
            segs.append([t, s])
            prev = s
    ages = {s: max([a for st, a in zip(run.state, run.age_ms) if st == s] or [0]) for s in set(run.state)}
    walk = {p: len(faults.tec_walk(p, 20000)) for p in (0.05, 0.12, 0.2)}
    return {"confinement": conf, "babbler_min_period_ms": thr, "redundancy_states": segs, "max_age_by_state_ms": ages,
            "events": run.events, "guard_tripped_at_ms": run.guard_tripped_at, "lost_detected_at_ms": run.lost_detected_at,
            "tec_frames_to_bus_off": walk,
            "timeline": {"t": run.t_ms[::2], "age": run.age_ms[::2], "state": run.state[::2]}}


# R9 ------------------------------------------------------------------------------------------------
def r9_rma() -> dict:
    out = {a: rma.path_reliability(a) for a in ARCHS}
    sweep = {}
    for a in ("A1", "A2"):
        sweep[a] = {f"x{s}": rma.path_reliability(a, scale=s)["F_mission"] for s in (0.3, 1, 3, 10)}
    beta = {f"{b}": rma.path_reliability("A2", beta=b)["F_mission"] for b in (0.0, 0.01, 0.02, 0.05, 0.1)}
    return {"paths": out, "scale_sweep": sweep, "beta_sweep_A2": beta}


# R10 -----------------------------------------------------------------------------------------------
def r10_aoa(r3: dict, r6: dict, r7: dict, r9: dict) -> dict:
    j = data.load("aoa")["judged"]
    screen = {}
    for a in ARCHS:
        arch = data.load("architectures")["architectures"][a]
        g = [r for r in r3[a]["rows"] if r["growth"] and r["errors"]]
        misses = sorted({m for r in g for m in r["misses"]})
        reasons = []
        if misses:
            more = "..." if len(misses) > 4 else ""
            reasons.append(f"{len(misses)} deadline misses with growth and errors ({', '.join(misses[:4])}{more})")
        if any(r["load"] > data.load("architectures")["max_design_load"]["value"] for r in g):
            reasons.append("bus load above 80 % with growth")
        if not arch.get("bywire_redundant"):
            reasons.append("a single network fault (short, open, babbling node) removes steering")
        chains = {c["chain"]: c for c in r6["archs"][a]}
        if not chains["YAW"]["ok"]:
            reasons.append("yaw-loop chain over budget")
        screen[a] = reasons
    alive = [a for a in ARCHS if not screen[a]]
    raw = {}
    for a in alive:
        g = [r for r in r3[a]["rows"] if r["growth"] and r["errors"]]
        yaw = next(c for c in r6["archs"][a] if c["chain"] == "YAW")
        arch = data.load("architectures")["architectures"][a]
        if arch.get("bywire_on_ethernet"):
            bw_load = r7["1000M_bywire"]["port_load"]["SWA>VCU"]
        else:
            bw_load = max(r["load"] for r in g if r["bus"] == "BW")
        cap = 1 - r7["1000M"]["max_port_load"]
        raw[a] = {"timing": 1 - yaw["total_ms"] / yaw["budget_ms"], "load": bw_load,
                  "reliability": r9["paths"][a]["F_mission"], "maturity": j["maturity"][a],
                  "integration": j["integration"][a] * j["integration_effort"][a], "cost": j["cost_eur"][a],
                  "mass": j["mass_kg"][a], "capacity": cap}
    util = aoa.utilities(raw)
    ratings = {k: float(v["rating"]) for k, v in data.load("aoa")["criteria"].items()}
    w = aoa.weights()
    f = aoa.fom(util, w)
    lead = max(f, key=f.get)
    sens = aoa.sensitivity(util, ratings)
    mc = aoa.monte_carlo(util, ratings, data.load("aoa")["monte_carlo_draws"])
    rivals = [a for a in f if a != lead]
    cross = {}
    for r in rivals:
        cross[r] = {k: aoa.crossover(util, ratings, k, r, lead) for k in ratings}
    return {"screen": screen, "raw": raw, "utility": util, "weights": w, "fom": f, "leader": lead,
            "sensitivity": [{"criterion": s["criterion"], "winner": s["winner"]} for s in sens], "monte_carlo": mc,
            "crossover_rating": cross}


# R11 -----------------------------------------------------------------------------------------------
def r11_phy() -> dict:
    p = data.load("architectures")["phy"]
    tp = p["propagation_ns_per_m"]["value"]
    tn = p["node_loop_delay_ns"]["value"]
    sp = p["sample_point"]["value"]
    rows = []
    for bps in (125000, 250000, 500000, 1000000):
        tbit = 1e9 / bps
        lmax = (sp * tbit - 2 * tn) / (2 * tp)
        rows.append({"bps": bps, "arbitration_limit_m": lmax})
    return {"rows": rows, "j1939": {k: v for k, v in p.items() if k.startswith("J1939")},
            "harness_m": data.load("architectures")["reference_harness_m"]["value"]}


# R12 -----------------------------------------------------------------------------------------------
def r12_uncertainty(n: int = 150, seed: int = 7, r6: dict | None = None) -> dict:
    """Monte Carlo over the uncertain vehicle and actuator inputs: the yaw-loop delay that keeps 30 deg of
    phase margin at 120 km/h, against the A2 yaw chain (with growth and errors)."""
    d = data.load("vehicle")
    rng = random.Random(seed)
    w = dynamics.freq_grid()
    py = data.values("vehicle", "loop_yaw")
    t30 = []
    for _ in range(n):
        dd = json.loads(json.dumps(d))
        dd["mass_kg"]["value"] = rng.uniform(20000, 33000)
        dd["cornering_coeff_per_rad"]["value"] = rng.uniform(4.0, 8.0)
        dd["steer_compliance_factor"]["value"] = rng.uniform(0.7, 1.0)
        dd["actuator_bandwidth_hz"]["value"] = rng.uniform(2.0, 5.0)
        dd["yaw_radius_of_gyration_m"]["value"] = rng.uniform(1.9, 2.4)
        v = dynamics.build(dd)
        p = dict(py, wc_design_rad_s=rng.uniform(5.0, 7.0))
        m = dynamics.margins(dynamics.loop_yaw(v, 120 / 3.6, p, w), w)
        t30.append(m.t30_s * 1e3)
    t = np.array(t30)
    out = {"n": n, "t30_ms": {"p5": float(np.percentile(t, 5)), "p50": float(np.median(t)), "min": float(t.min())},
           "samples": t.tolist(), "cases": {}}
    if r6:
        for a in ("A1", "A2", "A3"):
            for per in ("10", "5"):
                chain = r6["vcu_period_sweep_yaw_ms"][a][per]
                out["cases"][f"{a}_vcu{per}"] = {"chain_ms": chain, "P_within_t30": float((t >= chain).mean()),
                                                  "P_within_half_t30": float((t / 2 >= chain).mean())}
    return out


def run_all(out: Path = RESULTS, quick: bool = False) -> dict:
    out.mkdir(exist_ok=True)
    res = {}
    res["R1"] = r1_benchmark()
    res["R2"] = r2_frames(4000 if quick else 20000)
    res["R3"] = r3_rta()
    res["R4"] = r4_sim_vs_analysis(2 if quick else 6, 500 if quick else 2000)
    res["R5"] = r5_loops((100,) if quick else (20, 100, 120))
    res["R6"] = r6_chains(res["R5"])
    res["R7"] = r7_ethernet()
    res["R8"] = r8_faults()
    res["R9"] = r9_rma()
    res["R10"] = r10_aoa(res["R3"], res["R6"], res["R7"], res["R9"])
    res["R11"] = r11_phy()
    res["R12"] = r12_uncertainty(30 if quick else 150, r6=res["R6"])
    for k, v in res.items():
        (out / f"{k}.json").write_text(json.dumps(v, indent=1, default=_json_default), encoding="utf-8", newline="\n")
    return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return str(o)
