"""Command line: python -m vnet <command>."""
from __future__ import annotations

import argparse
import sys

from . import data, ethernet, faults, latency, network, studies
from .rta import ErrorModel


def _f(x, nd=2):
    return "  -  " if x is None else (f"{x:.{nd}f}" if x == x and x != float("inf") else "inf")


def cmd_bench(_a) -> None:
    r = studies.r1_benchmark()
    print("Davis et al. 2007, Table 2 (125 kbit/s, 3 messages, 1 ms each)")
    print(f"{'msg':4} {'legacy':>8} {'revised':>8} {'sim (critical instant)':>24}")
    for k in ("A", "B", "C"):
        print(f"{k:4} {r['legacy_ms'][k]:8.2f} {r['revised_ms'][k]:8.2f} {r['sim_critical_instant_ms'][k]:24.3f}")
    print(f"busy period of C {r['busy_C_ms']:.1f} ms, instances {r['Q_C']} (published: 7.0 ms, 2)")
    print("breakdown utilisation of the legacy analysis (published: 45.5 21.4 9.2 3.4 0.82 %):",
          " ".join(f"{n}:{v}%" for n, v in r["breakdown_pct"].items()))


def cmd_rta(a) -> None:
    _, buses = network.build_arch(a.arch, growth=a.growth)
    print(data.load("architectures")["architectures"][a.arch]["name"], "| growth" if a.growth else "", "| errors" if a.errors else "")
    for b in buses:
        rep = network.analyse_bus(b, network.error_model(b) if a.errors else ErrorModel())
        print(f"\n{b.id} {b.kind} {b.bps // 1000} kbit/s{' x2 (dual)' if b.dual else ''}: {rep.n} frames, "
              f"worst-case load {rep.load * 100:.1f} %, deadline misses {len(rep.misses)}")
        if a.verbose or b.id in ("BW", "J1939"):
            print(f"  {'message':12} {'ID':>10} {'T ms':>6} {'R ms':>7} {'slack':>6}")
            for m in sorted(b.msgs, key=lambda x: x.can_id)[: a.top]:
                r = rep.results[m.name]
                print(f"  {m.name:12} {m.can_id:#010x} {m.T_ms:6.0f} {_f(None if r.R is None else r.R / 1e6):>7} "
                      f"{('MISS' if not r.schedulable else f'{r.slack * 100:5.0f}%'):>6}")


def cmd_loops(_a) -> None:
    r = studies.r5_loops(verify_speeds=())
    v = r["vehicle"]
    print(f"8x8, {v['mass'] / 1000:.0f} t, effective wheelbase {v['l_eff']:.2f} m, understeer {v['Kus_deg_per_g']:.1f} deg/g")
    a = r["actuator_loop"]
    print(f"L1 position loop across the network: crossover {a['wc']:.1f} rad/s, PM {a['pm_deg']:.0f} deg, "
          f"delay for PM 30 deg {a['t30_s'] * 1e3:.0f} ms")
    print(f"{'km/h':>5} {'yaw zeta':>8} | {'L2 wc':>6} {'PM':>4} {'DM ms':>6} {'T30 ms':>7} | "
          f"{'L3 wc':>6} {'PM':>4} {'T30 ms':>7}")
    for s in r["speeds"]:
        y, p = s["yaw"], s["path"]
        print(f"{s['kmh']:5d} {s['yaw_damping']:8.2f} | {y['wc']:6.2f} {y['pm_deg']:4.0f} {y['dm_s'] * 1e3:6.0f} {y['t30_s'] * 1e3:7.0f} | "
              f"{p['wc']:6.2f} {p['pm_deg']:4.0f} {p['t30_s'] * 1e3:7.0f}")


def cmd_chains(a) -> None:
    r5 = studies.r5_loops(verify_speeds=())
    b = studies.budgets_from(r5)
    for arch in ([a.arch] if a.arch else list(studies.ARCHS)):
        print(f"\n{arch}: {data.load('architectures')['architectures'][arch]['name']}")
        for c in latency.evaluate(arch, budgets=b):
            hops = " + ".join(f"{h} {v:.1f}" for h, v in c.hops)
            print(f"  {c.chain:13} {c.total_ms:6.1f} ms  budget {c.budget_ms:6.1f}  {'ok' if c.ok else 'OVER'}   ({hops})")


def cmd_ether(a) -> None:
    net = ethernet.build(a.rate, bywire=a.bywire)
    b = ethernet.bounds(net)
    s = ethernet.simulate(net, t_end=1.0)
    al = ethernet.simulate(net, t_end=1.0, aligned=True)
    print(f"backbone {a.rate} Mbit/s{' + by-wire planes' if a.bywire else ''}: bound (network calculus) vs simulation")
    print(f"  {'flow':12} {'PCP':>3} {'bound ms':>9} {'sim ms':>8} {'aligned':>8} {'deadline':>8}")
    for f in net.flows:
        print(f"  {f.name:12} {f.pcp:3d} {_f(b[f.name] * 1e3, 3):>9} {s[f.name] * 1e3:8.3f} "
              f"{al[f.name] * 1e3:8.3f} {f.deadline_s * 1e3:8.0f}")


def cmd_faults(_a) -> None:
    for bps in (250000, 500000):
        c = faults.confinement(bps)
        print(f"{bps // 1000} kbit/s: error passive after {c.frames_to_passive} failed frames, bus-off after "
              f"{c.frames_to_bus_off} ({c.t_bus_off_ms:.1f} ms); recovery >= {c.t_recover_min_ms:.2f} ms + application delay")
    run = faults.redundancy_run()
    print("\ndual-channel run (python-can virtual buses, C redundancy manager and guardian):")
    for t, e in run.events:
        print(f"  t={t:4d} ms  {e}")
    prev = None
    for t, s in zip(run.t_ms, run.state):
        if s != prev:
            print(f"  t={t:4d} ms  state {s}")
            prev = s


def cmd_all(a) -> None:
    studies.run_all(quick=a.quick)
    print("results written to results/R1..R12.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vnet", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bench", help="reproduce the published CAN benchmark").set_defaults(fn=cmd_bench)
    p = sub.add_parser("rta", help="worst-case response times of one architecture")
    p.add_argument("--arch", default="A2", choices=studies.ARCHS)
    p.add_argument("--growth", action="store_true")
    p.add_argument("--errors", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--top", type=int, default=16)
    p.set_defaults(fn=cmd_rta)
    sub.add_parser("loops", help="steering-loop margins vs speed").set_defaults(fn=cmd_loops)
    p = sub.add_parser("chains", help="end-to-end chains vs budgets")
    p.add_argument("--arch", default=None, choices=studies.ARCHS)
    p.set_defaults(fn=cmd_chains)
    p = sub.add_parser("ether", help="Ethernet backbone bounds vs simulation")
    p.add_argument("--rate", type=int, default=1000)
    p.add_argument("--bywire", action="store_true")
    p.set_defaults(fn=cmd_ether)
    sub.add_parser("faults", help="fault confinement and the dual-channel run").set_defaults(fn=cmd_faults)
    p = sub.add_parser("all", help="every study -> results/*.json")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_all)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
