"""Figures for README and DOSSIER from results/*.json (real runs) and from live CLI output.

python tools/make_figures.py   -> docs/img/*.png
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vnet import data, dynamics  # noqa: E402

IMG = ROOT / "docs" / "img"
C = {"A0": "#9e9e9e", "A1": "#c77c2e", "A2": "#1f6fb2", "A3": "#2a9d8f", "A4": "#8e5cb5"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})


def R(k):
    return json.loads((ROOT / "results" / f"{k}.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=130)
    plt.close(fig)
    print("wrote", name)


def fig_loops():
    r5 = R("R5")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sp = [s["kmh"] for s in r5["speeds"]]
    ax[0].plot(sp, [s["yaw"]["t30_s"] * 1e3 for s in r5["speeds"]], "o-", c=C["A2"], label="L2 yaw loop: delay for PM 30 deg")
    ax[0].plot(sp, [s["yaw"]["dm_s"] * 1e3 for s in r5["speeds"]], "o--", c=C["A2"], alpha=0.5, label="L2 delay margin (PM 0)")
    ax[0].axhline(r5["actuator_loop"]["t30_s"] * 1e3, c="#c0392b", ls="-", label="L1 position loop across the bus: PM 30 deg")
    ax[0].plot(sp, [s["path"]["t30_s"] * 1e3 for s in r5["speeds"]], "s-", c=C["A3"], label="L3 path following: PM 30 deg")
    for v in r5["verification"]:
        ax[0].plot(v["kmh"], v["yaw_sim_ms"], "kx", ms=8)
    ax[0].plot([], [], "kx", label="time-domain instability onset")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("speed, km/h")
    ax[0].set_ylabel("tolerable added delay, ms")
    ax[0].set_title("How much latency the steering loops tolerate")
    ax[0].legend(fontsize=7)
    v = dynamics.build(data.load("vehicle"))
    w = dynamics.freq_grid()
    py = data.values("vehicle", "loop_yaw")
    for kmh, ls in ((20, ":"), (120, "-")):
        L = dynamics.loop_yaw(v, kmh / 3.6, py, w)
        ax[1].semilogx(w, np.degrees(np.unwrap(np.angle(L))), ls, c=C["A2"], label=f"phase, {kmh} km/h")
    ax[1].axvline(py["wc_design_rad_s"], c="k", lw=0.8)
    ax[1].axhline(-180, c="#c0392b", lw=0.8)
    ax[1].set_xlim(0.3, 100)
    ax[1].set_ylim(-300, 0)
    ax[1].set_xlabel("frequency, rad/s")
    ax[1].set_ylabel("open-loop phase, deg")
    ax[1].set_title("L2 at crossover 6 rad/s: speed eats phase")
    ax[1].legend(fontsize=7)
    u = 120 / 3.6
    for d, c in ((0.018, C["A2"]), (0.06, "#e09f3e"), (0.12, "#c0392b")):
        s = dynamics.simulate_yaw(v, u, py, delay_s=d, period_s=0.01, t_end=4)
        ax[2].plot(s["t"], s["r"] / 0.1, c=c, label=f"loop delay {d * 1e3:.0f} ms")
    ax[2].set_xlabel("time, s")
    ax[2].set_ylabel("yaw rate / command")
    ax[2].set_title("Yaw-rate step at 120 km/h, 10 ms controller")
    ax[2].legend(fontsize=7)
    save(fig, "loops.png")


def fig_rta():
    r3 = R("R3")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    labels, loads, cols, miss = [], [], [], []
    for a, d in r3.items():
        for row in d["rows"]:
            if row["growth"] and row["errors"] and not (row["bus"] == "BW" and False):
                labels.append(f"{a}\n{row['bus']} {row['bps'] // 1000}k{' FD' if row['kind'] == 'canfd' else ''}")
                loads.append(row["load"] * 100)
                cols.append(C[a])
                miss.append(len(row["misses"]))
    x = np.arange(len(labels))
    ax[0].bar(x, loads, color=cols)
    for i, (l, m) in enumerate(zip(loads, miss)):
        ax[0].text(i, l + 2, f"{m} miss" if m else "ok", ha="center", fontsize=7, color="#c0392b" if m else "k")
    ax[0].axhline(80, c="k", ls="--", lw=0.8)
    ax[0].text(len(labels) - 0.5, 82, "80 % design ceiling", ha="right", fontsize=7)
    ax[0].set_xticks(x, [lb.replace(chr(10), " ") for lb in labels], fontsize=7, rotation=45, ha="right")
    ax[0].set_ylabel("worst-case bus load, %")
    ax[0].set_title("Every CAN bus with +30 % growth and the error model")
    for a, mk in (("A0", "o"), ("A2", "s")):
        row = next(r for r in r3[a]["rows"] if r["growth"] and r["errors"] and r["bus"] in ("J1939", "BW"))
        T = [v[1] for v in row["all"].values() if v[1] <= 120]
        Rr = [v[0] if v[0] is not None else 200 for v in row["all"].values() if v[1] <= 120]
        ax[1].scatter(T, Rr, marker=mk, c=C[a], label=f"{a} {row['bus']} {row['bps'] // 1000}k", alpha=0.8)
    ax[1].plot([5, 120], [5, 120], "k--", lw=0.8, label="R = D")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("deadline (= period), ms")
    ax[1].set_ylabel("worst-case response time, ms (200 = unbounded)")
    ax[1].set_title("Response time vs deadline, revised analysis [Davis 2007]")
    ax[1].legend(fontsize=7)
    save(fig, "rta.png")


def fig_sim():
    r4, r1 = R("R4"), R("R1")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    k = next(iter(r4))
    rows = r4[k]["rows"]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.27, [r["bound_ms"] for r in rows], 0.27, color=C["A2"], label="analysis bound")
    ax[0].bar(x, [r["critical_instant_ms"] for r in rows], 0.27, color="#7fb3d5", label="simulation, critical instant")
    ax[0].bar(x + 0.27, [r["random_max_ms"] for r in rows], 0.27, color="#d5d8dc", label="simulation, random phasing (max)")
    ax[0].set_xticks(x, [r["msg"] for r in rows], rotation=60, fontsize=7)
    ax[0].set_ylabel("response time, ms")
    ax[0].set_title(f"Bit-level simulator vs analysis: {k.replace('_', ' ')}")
    ax[0].legend(fontsize=7)
    m = ["A", "B", "C"]
    xx = np.arange(3)
    ax[1].bar(xx - 0.25, [r1["legacy_ms"][i] for i in m], 0.25, color="#9e9e9e", label="legacy analysis")
    ax[1].bar(xx, [r1["revised_ms"][i] for i in m], 0.25, color=C["A2"], label="revised analysis")
    ax[1].bar(xx + 0.25, [r1["sim_critical_instant_ms"][i] for i in m], 0.25, color="#7fb3d5", label="simulation")
    ax[1].axhline(3.25, c="#c0392b", lw=0.8)
    ax[1].text(2.4, 3.3, "D = 3.25", ha="right", fontsize=7, color="#c0392b")
    ax[1].set_xticks(xx, m)
    ax[1].set_title("Published example reproduced")
    ax[1].legend(fontsize=7, loc="upper left")
    save(fig, "sim_vs_analysis.png")


def fig_chains():
    r6 = R("R6")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    archs = list(r6["archs"])
    for i, a in enumerate(archs):
        c = next(x for x in r6["archs"][a] if x["chain"] == "YAW")
        bottom = 0
        for (h, v) in c["hops"]:
            v = min(v, 60)
            task = h in ("VCU", "SAS")
            ax[0].bar(i, v, bottom=bottom, color="#e5e7e9" if task else C[a], edgecolor="k", lw=0.4)
            if v > 1.5:
                ax[0].text(i, bottom + v / 2, f"{h.split(' ')[0]}\n{v:.1f}", ha="center", va="center", fontsize=6)
            bottom += v
    ax[0].axhline(r6["budgets_ms"]["YAW"], c="#c0392b", ls="--")
    ax[0].text(4.4, r6["budgets_ms"]["YAW"] + 0.5, f"budget {r6['budgets_ms']['YAW']:.1f} ms", ha="right", fontsize=7, color="#c0392b")
    ax[0].set_xticks(range(len(archs)), archs)
    ax[0].set_ylabel("worst-case latency, ms")
    ax[0].set_title("Yaw-loop chain INS -> VCU -> steering (growth + errors)")
    sw, ld = r6["vcu_period_sweep_yaw_ms"], r6["vcu_period_sweep_bw_load"]
    for a in sw:
        pers = [p for p in sw[a] if sw[a][p] != "inf" and sw[a][p] != float("inf")]
        ax[1].plot([float(p) for p in pers], [sw[a][p] for p in pers], "o-", c=C[a], label=a)
        for p in pers:
            if ld[a][p]["misses"]:
                ax[1].plot(float(p), sw[a][p], "x", c="#c0392b", ms=10)
    ax[1].plot([], [], "x", c="#c0392b", label="by-wire bus has misses")
    ax[1].axhline(r6["budgets_ms"]["YAW"], c="#c0392b", ls="--")
    ax[1].set_xlabel("VCU control period (and by-wire message period), ms")
    ax[1].set_ylabel("yaw chain, ms")
    ax[1].set_title("The task period, not the bus, dominates the chain")
    ax[1].legend(fontsize=7)
    save(fig, "chains.png")


def fig_ethernet():
    r7 = R("R7")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 1.2]})
    rows = r7["1000M_bywire"]["flows"]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.27, [r["bound_ms"] for r in rows], 0.27, color=C["A4"], label="network-calculus bound")
    ax[0].bar(x, [r["aligned_ms"] for r in rows], 0.27, color="#c39bd3", label="simulation, aligned bursts")
    ax[0].bar(x + 0.27, [r["sim_ms"] for r in rows], 0.27, color="#e8daef", label="simulation, random phasing")
    ax[0].scatter(x, [r["deadline_ms"] for r in rows], marker="_", s=200, c="#c0392b", label="deadline", zorder=5)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, [r["flow"] for r in rows], rotation=60, fontsize=7)
    ax[0].set_ylabel("end-to-end delay, ms")
    ax[0].set_title("1 Gbit/s backbone + by-wire planes: bound vs simulation")
    ax[0].legend(fontsize=7)
    names = ["CAM1", "DVE", "INS_VCU", "GW_CAN"]
    for i, rate in enumerate(("100M", "1000M")):
        d = {r["flow"]: r for r in r7[rate]["flows"]}
        ax[1].bar(np.arange(4) + (i - 0.5) * 0.35, [min(d[n]["bound_ms"], 200) for n in names], 0.35,
                  color=["#d98880", C["A2"]][i], label=f"{rate}bit/s")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(4), names, fontsize=7)
    ax[1].set_title("100 Mbit/s cannot carry the video")
    ax[1].legend(fontsize=7)
    save(fig, "ethernet.png")


def fig_redundancy():
    r8 = R("R8")
    t = np.array(r8["timeline"]["t"])
    age = np.array(r8["timeline"]["age"])
    st = r8["timeline"]["state"]
    lvl = {"INIT": 0, "DUAL": 3, "SINGLE_A": 2, "SINGLE_B": 2, "LOST": 1}
    fig, ax = plt.subplots(2, 1, figsize=(12, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax[0].step(t, [lvl[s] for s in st], where="post", c=C["A2"])
    ax[0].set_yticks([1, 2, 3], ["LOST\n(safe stop)", "SINGLE", "DUAL"])
    ax[0].set_title("Steering command on two CAN channels: python-can virtual buses, C redundancy manager and guardian")
    ax[1].plot(t, np.minimum(age, 60), c=C["A2"])
    ax[1].axhline(30, c="#c0392b", ls="--", lw=0.8)
    ax[1].text(5, 32, "timeout 3 periods", fontsize=7, color="#c0392b")
    ax[1].set_ylabel("age of used set-point, ms")
    ax[1].set_xlabel("time, ms")
    for i, (tt, e) in enumerate(r8["events"]):
        for a in ax:
            a.axvline(tt, c="k", lw=0.6, ls=":")
        ax[1].text(tt + 4, 54 - 9 * (i % 3) - 4, e, fontsize=7, rotation=0, bbox={"fc": "white", "ec": "none", "pad": 0.5})
    ax[1].set_ylim(0, 60)
    save(fig, "redundancy.png")


def fig_aoa_rma():
    r10, r9, r12 = R("R10"), R("R9"), R("R12")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    fo = r10["fom"]
    ax[0].bar(list(fo), list(fo.values()), color=[C[a] for a in fo])
    for i, a in enumerate(fo):
        ax[0].text(i, fo[a] + 0.01, f"{fo[a]:.3f}\nwins {100 * r10['monte_carlo'][a]:.0f} %", ha="center", fontsize=7)
    ax[0].set_ylim(0, 0.85)
    ax[0].set_title("Figure of merit (A0, A1 screened out)\nand share of 2000 random weightings won")
    p = r9["paths"]
    ax[1].bar(list(p), [p[a]["F_mission"] / r9_hours() for a in p], color=[C[a] for a in p])
    ax[1].set_yscale("log")
    ax[1].axhline(1e-7, c="#c0392b", ls="--", lw=0.8)
    ax[1].text(4.4, 1.2e-7, "1e-7 /h", ha="right", fontsize=7, color="#c0392b")
    ax[1].set_ylabel("network-caused loss of steering, per hour")
    ax[1].set_title("Steer-command path (beta = 5 %)")
    s = np.array(r12["samples"])
    ax[2].hist(s, bins=25, color="#d6eaf8", edgecolor=C["A2"])
    for key, ls in (("A2_vcu10", "-"), ("A2_vcu5", "--"), ("A3_vcu5", ":")):
        c = r12["cases"][key]
        ax[2].axvline(c["chain_ms"], c="k", ls=ls, label=f"{key.replace('_vcu', ', VCU ')} ms: P = {c['P_within_t30']:.2f}")
    ax[2].set_xlabel("yaw-loop delay for PM 30 deg at 120 km/h, ms")
    ax[2].set_title(f"{r12['n']} vehicles/actuators sampled")
    ax[2].legend(fontsize=7)
    save(fig, "aoa_rma_mc.png")


def r9_hours():
    return data.load("rma")["mission_h"]["value"]


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def box(x, y, t, c="#ffffff"):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.25), 1.0, 0.5, fc=c, ec="k", lw=0.8))
        ax.text(x, y, t, ha="center", va="center", fontsize=7)

    buses = [(5.55, "BW-A  500 kbit/s (J1939-14, E2E)", C["A2"]), (4.45, "BW-B  500 kbit/s, separate route", C["A2"]),
             (3.0, "PT  250 kbit/s (J1939-11)", C["A1"]), (2.0, "BODY  250 kbit/s (J1939-15)", "#7f8c8d")]
    for y, t, c in buses:
        ax.plot([0.3, 11.7], [y, y], c=c, lw=2.2)
        ax.text(0.3, y + 0.12, t, fontsize=7, color=c)
    for x, t in ((1.5, "DCU\nhand wheel"), (3.0, "INS"), (4.5, "SAS1"), (6.0, "SAS2"), (7.5, "EBS")):
        box(x, 5.0, t, "#eaf2f8")
        ax.plot([x, x], [4.75, 4.7], c="k", lw=0.8)
    box(9.5, 3.8, "VCU\n(by-wire computer)", "#d4e6f1")
    for y in (5.55, 4.45, 3.0, 2.0):
        ax.plot([9.5, 9.5], [4.05, y], c="k", lw=0.6)
    for x, t in ((2.0, "Engine"), (3.6, "Transmission")):
        box(x, 2.55, t)
    for x, t in ((5.2, "CTIS"), (6.6, "PMC / SSPC"), (8.0, "BCM / CBRN"), (11.0, "HUMS")):
        box(x, 1.55, t)
    ax.plot([0.3, 11.7], [0.6, 0.6], c=C["A4"], lw=3)
    ax.text(0.3, 0.72, "Open-architecture Ethernet backbone 1 Gbit/s, 802.1Q priorities (C4I, video, navigation)",
            fontsize=7, color=C["A4"])
    for x, t in ((1.5, "4 SA cameras"), (3.0, "Driver's vision"), (4.5, "C4I computer"), (6.0, "Radios"), (7.5, "Crew displays"),
                 (11.0, "Nav / INS")):
        box(x, 0.25 + 0.0, t, "#f5eef8")
    ax.plot([9.5, 9.5], [3.55, 0.6], c=C["A4"], lw=1.2)
    ax.text(9.6, 1.0, "VCU reads navigation\ndirectly (no gateway)", fontsize=6.5)
    ax.set_title("Recommended architecture A2: partitioned J1939 with dual by-wire channels and an Ethernet backbone", fontsize=9)
    save(fig, "architecture.png")


def cli_png(name: str, args: list[str], title: str):
    from vnet.__main__ import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(args)
    text = f"$ python -m vnet {' '.join(args)}\n" + buf.getvalue()
    lines = text.rstrip().splitlines()
    h = 0.17 * len(lines) + 0.4
    fig = plt.figure(figsize=(11.5, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 0.98, text, va="top", ha="left", family="monospace", fontsize=7.6, color="#e6e6e6")
    fig.savefig(IMG / name, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name, f"({title})")


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_loops()
    fig_rta()
    fig_sim()
    fig_chains()
    fig_ethernet()
    fig_redundancy()
    fig_aoa_rma()
    cli_png("cli_bench.png", ["bench"], "benchmark")
    cli_png("cli_rta.png", ["rta", "--arch", "A0"], "legacy bus")
    cli_png("cli_loops.png", ["loops"], "loops")
    cli_png("cli_chains.png", ["chains", "--arch", "A2"], "chains")
    cli_png("cli_faults.png", ["faults"], "faults")



if __name__ == "__main__":
    main()
