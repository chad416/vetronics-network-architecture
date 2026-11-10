"""Generate the J1939 DBC of the message set (docs/vehicle.dbc) with cantools; `--check` fails if stale.

By-wire frames carry the E2E header (CRC-8 in byte 0, counter in the low nibble of byte 1); the by-wire
groups with a defined layout get their engineering signals, the rest a raw 48-bit payload.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cantools
from cantools.database.can import Database, Message, Node, Signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vnet.network import message_table  # noqa: E402

LAYOUT = {
    "STEER_CMD": [("Mode", 12, 4, 1, 0, False, ""), ("AngleAx1", 16, 16, 0.01, 0, True, "deg"),
                  ("AngleAx2", 32, 16, 0.01, 0, True, "deg"), ("RateLimit", 48, 16, 0.1, 0, False, "deg/s")],
    "INS_DYN": [("YawRate", 16, 16, 0.002, 0, True, "rad/s"), ("AccLat", 32, 16, 0.001, 0, True, "m/s2"),
                ("AccLon", 48, 16, 0.001, 0, True, "m/s2")],
    "SAS1_STAT": [("Fault", 12, 4, 1, 0, False, ""), ("Angle", 16, 16, 0.01, 0, True, "deg"),
                  ("RackForce", 32, 16, 1, 0, True, "N"), ("Temp", 48, 8, 1, -40, False, "degC")],
    "SAS2_STAT": [("Fault", 12, 4, 1, 0, False, ""), ("Angle", 16, 16, 0.01, 0, True, "deg"),
                  ("RackForce", 32, 16, 1, 0, True, "N"), ("Temp", 48, 8, 1, -40, False, "degC")],
}


def build() -> Database:
    msgs = []
    table = message_table()
    nodes = sorted({m.sa for m in table})
    for m in table:
        sigs = []
        if m.e2e:
            sigs += [Signal("E2E_CRC", 0, 8, byte_order="little_endian"),
                     Signal("E2E_Counter", 8, 4, byte_order="little_endian")]
            spec = LAYOUT.get(m.name)
            if spec:
                for n, start, length, scale, off, signed, unit in spec:
                    sigs.append(Signal(n, start, length, byte_order="little_endian", is_signed=signed,
                                       conversion=cantools.database.conversion.BaseConversion.factory(scale, off),
                                       unit=unit or None))
            else:
                sigs.append(Signal("Payload", 16, 8 * m.n_bytes - 16, byte_order="little_endian"))
        else:
            sigs.append(Signal("Payload", 0, 8 * m.n_bytes, byte_order="little_endian"))
        for sg in sigs:
            sg.receivers = [n for n in nodes if n != m.sa]
        msgs.append(Message(frame_id=m.can_id, name=m.name, length=m.n_bytes, signals=sigs, is_extended_frame=True,
                            senders=[m.sa], cycle_time=int(m.T_ms)))
    return Database(messages=msgs, nodes=[Node(n) for n in nodes])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build().as_dbc_string().replace(chr(13) + chr(10), chr(10))
    p = ROOT / "docs" / "vehicle.dbc"
    if a.check:
        ok = p.exists() and p.read_text(encoding="utf-8") == text
        print("docs/vehicle.dbc is current" if ok else "docs/vehicle.dbc is out of date: run python tools/make_dbc.py")
        return 0 if ok else 1
    p.write_text(text, encoding="utf-8", newline="\n")
    db = cantools.database.load_file(str(p))
    print(f"wrote {p.relative_to(ROOT)}: {len(db.messages)} messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
