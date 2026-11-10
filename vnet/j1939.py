"""SAE J1939-21 identifiers and the transport protocol, as far as the timing analysis needs them.

29-bit identifier = priority (3) | EDP (1) | DP (1) | PDU format PF (8) | PDU specific PS (8) | source address SA (8).
PF < 240: PDU1, PS is the destination address and not part of the PGN. PF >= 240: PDU2 (broadcast), PS is the
group extension and part of the PGN [J1939-21 via MILCAN-COS, FMS2010]. Priority 0 is the highest; because the
priority bits are the most significant identifier bits, J1939 priority dominates arbitration, then PGN, then SA.

Transport protocol (multi-packet, up to 1785 bytes = 255 x 7): TP.CM (PGN 60416, 0xEC00) and TP.DT
(PGN 60160, 0xEB00), each 8 data bytes; one TP.DT carries a sequence number and 7 data bytes. BAM sends the
packets 50-200 ms apart; CMDT uses RTS/CTS flow control [J1939TP].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PGN_TP_CM = 0xEC00
PGN_TP_DT = 0xEB00
TP_MAX_BYTES = 1785
BAM_GAP_MIN_S = 0.050
BAM_GAP_MAX_S = 0.200


def make_id(priority: int, pgn: int, sa: int, da: int = 0xFF) -> int:
    if not 0 <= priority <= 7 or not 0 <= sa <= 0xFF:
        raise ValueError("priority 0-7, source address 0-255")
    dp = (pgn >> 16) & 0x3
    pf = (pgn >> 8) & 0xFF
    ps = pgn & 0xFF
    if pf < 240:
        if ps != 0:
            raise ValueError("PDU1 PGN must have PS = 0; the destination goes into PS")
        ps = da & 0xFF
    return (priority << 26) | (dp << 24) | (pf << 16) | (ps << 8) | sa


@dataclass(frozen=True)
class J1939Id:
    priority: int
    pgn: int
    sa: int
    da: int | None


def parse_id(can_id: int) -> J1939Id:
    priority = (can_id >> 26) & 0x7
    dp = (can_id >> 24) & 0x3
    pf = (can_id >> 16) & 0xFF
    ps = (can_id >> 8) & 0xFF
    sa = can_id & 0xFF
    if pf < 240:
        return J1939Id(priority, (dp << 16) | (pf << 8), sa, ps)
    return J1939Id(priority, (dp << 16) | (pf << 8) | ps, sa, None)


def tp_packets(n_bytes: int) -> int:
    if not 9 <= n_bytes <= TP_MAX_BYTES:
        raise ValueError("TP carries 9-1785 bytes (8 or fewer go in one frame)")
    return math.ceil(n_bytes / 7)


def bam_duration_s(n_bytes: int, gap_s: float = BAM_GAP_MIN_S) -> float:
    """Time from TP.CM_BAM to the last TP.DT at a fixed inter-packet gap: the payload is usable only then."""
    return tp_packets(n_bytes) * gap_s
