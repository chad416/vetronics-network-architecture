"""CAN frames at bit level: classical (ISO 11898-1 / CAN 2.0) base and extended data frames, and CAN FD lengths.

The classical frame is built bit by bit (SOF, arbitration, control, data, CRC-15), bit-stuffed exactly as
the specification requires (a complement bit after five equal bits, stuff bits counted in the run), and
closed with the fixed tail (CRC delimiter, ACK slot, ACK delimiter, 7-bit EOF, 3-bit intermission).
The worst-case length is Davis et al. 2007 Eq. (1) [DAVIS2007 p. 245]; a test checks that no payload
exceeds it and that the bound is reached.
"""
from __future__ import annotations

import math

CRC15_POLY = 0x4599   # x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1 [CAN20 Part A p. 13]
TAIL_BITS = 1 + 1 + 1 + 7 + 3   # CRC delimiter, ACK slot, ACK delimiter, EOF, intermission (IFS)


def crc15(bits: list[int]) -> int:
    """CRC-15 of the CAN 2.0 specification over a destuffed bit stream (SOF .. end of data)."""
    crc = 0
    for b in bits:
        nxt = b ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if nxt:
            crc ^= CRC15_POLY
    return crc


def _bits(value: int, n: int) -> list[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def frame_bits(can_id: int, data: bytes, extended: bool = True) -> list[int]:
    """Destuffed bits from SOF to the end of the CRC sequence (the stuffed region)."""
    if not 0 <= len(data) <= 8:
        raise ValueError("classical CAN carries 0-8 data bytes")
    b = [0]                                           # SOF (dominant)
    if extended:
        if not 0 <= can_id < 1 << 29:
            raise ValueError("29-bit identifier expected")
        b += _bits(can_id >> 18, 11)                  # base identifier ID28..ID18
        b += [1, 1]                                   # SRR (recessive), IDE (recessive)
        b += _bits(can_id & 0x3FFFF, 18)              # extended identifier ID17..ID0
        b += [0, 0, 0]                                # RTR (data frame), r1, r0
    else:
        if not 0 <= can_id < 1 << 11:
            raise ValueError("11-bit identifier expected")
        b += _bits(can_id, 11)
        b += [0, 0, 0]                                # RTR, IDE (dominant), r0
    b += _bits(len(data), 4)                          # DLC
    for byte in data:
        b += _bits(byte, 8)                           # MSB first [CAN20 Part A p. 13]
    return b + _bits(crc15(b), 15)


def stuff(bits: list[int]) -> list[int]:
    """Insert a complement bit after every run of five equal bits; stuff bits start new runs."""
    out: list[int] = []
    run_bit, run = None, 0
    for x in bits:
        out.append(x)
        if x == run_bit:
            run += 1
        else:
            run_bit, run = x, 1
        if run == 5:
            s = 1 - x
            out.append(s)
            run_bit, run = s, 1
    return out


def on_bus_bits(can_id: int, data: bytes, extended: bool = True) -> int:
    """Exact number of bit times the frame occupies, including stuff bits and the 3-bit intermission."""
    return len(stuff(frame_bits(can_id, data, extended))) + TAIL_BITS


def worst_case_bits(n_bytes: int, extended: bool = True) -> int:
    """Davis et al. 2007 Eq. (1): (g + 8s + 13 + floor((g + 8s - 1)/4)), g = 54 extended, 34 base."""
    g = 54 if extended else 34
    return g + 8 * n_bytes + 13 + (g + 8 * n_bytes - 1) // 4


# --- CAN FD (ISO 11898-1:2015 frame format; used by J1939-22) -------------------------------------
FD_DLC = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


def fd_payload(n: int) -> int:
    """Smallest CAN FD data-field size that holds n bytes."""
    for s in FD_DLC:
        if s >= n:
            return s
    raise ValueError("CAN FD carries at most 64 bytes")


def fd_worst_case_time(n_bytes: int, nominal_bps: float, data_bps: float, extended: bool = True) -> float:
    """Upper bound of a CAN FD frame duration in seconds (bit-rate switch on).

    Arbitration phase at the nominal rate: SOF, identifier, SRR/IDE (extended), RRS, FDF, res, BRS.
    Data phase at the data rate: ESI, DLC, data, then the stuff count (4 bits) and the CRC (17 bits up to
    16 data bytes, 21 above) with one fixed stuff bit per 4 bits plus one in front, and the CRC delimiter.
    Dynamic stuffing (worst case one per 4 bits after the first, as in [DAVIS2007] Eq. 1) applies from SOF to
    the end of the data field; each phase is bounded separately and one extra stuff bit covers the
    boundary. The tail (ACK slot, ACK delimiter, EOF 7, intermission 3) is at the nominal rate.
    """
    s = fd_payload(n_bytes)
    arb = (1 + 11 + 1 + 1 + 18 + 1 + 1 + 1 + 1) if extended else (1 + 11 + 1 + 1 + 1 + 1)
    arb_st = arb + (arb - 1) // 4 + 1
    dat = 1 + 4 + 8 * s
    dat_st = dat + (dat - 1) // 4
    crc_len = 17 if s <= 16 else 21
    crc_field = 4 + crc_len + math.ceil((4 + crc_len) / 4) + 1   # stuff count + CRC + fixed stuff bits
    data_phase = dat_st + crc_field + 1                            # + CRC delimiter
    tail = 1 + 1 + 7 + 3
    return (arb_st + tail) / nominal_bps + data_phase / data_bps
