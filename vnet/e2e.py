"""ctypes binding to the C library c/e2e_guard.c (the code that would run in the by-wire ECUs).

`load()` builds the shared library on first use with $CC, gcc, cc or (on Windows) `python -m ziglang cc`.
"""
from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "c" / "e2e_guard.c"
BUILD = ROOT / "build"
LIBNAME = "e2e_guard.dll" if sys.platform == "win32" else "libe2e_guard.so"

RM_STATES = {0: "INIT", 1: "DUAL", 2: "SINGLE_A", 3: "SINGLE_B", 4: "LOST"}
E2E_STATUS = {0: "OK", 1: "OKSOMELOST", 2: "REPEATED", 3: "WRONGSEQUENCE", 4: "ERROR", 5: "INITIAL"}


class E2ETx(ct.Structure):
    _fields_ = [("data_id", ct.c_uint16), ("counter", ct.c_uint8)]


class E2ERx(ct.Structure):
    _fields_ = [("data_id", ct.c_uint16), ("max_delta", ct.c_uint8), ("last", ct.c_uint8), ("synced", ct.c_bool)]


class RM(ct.Structure):
    _fields_ = [("rx", E2ERx * 2), ("last_ok_ms", ct.c_uint32 * 2), ("seen", ct.c_bool * 2), ("timeout_ms", ct.c_uint32),
                ("value", ct.c_uint8 * 8), ("len", ct.c_uint8), ("value_counter", ct.c_uint8), ("have_value", ct.c_bool),
                ("value_ms", ct.c_uint32), ("state", ct.c_int), ("accepted", ct.c_uint32), ("duplicates", ct.c_uint32),
                ("disagree", ct.c_uint32), ("rejected", ct.c_uint32 * 2)]


class Guard(ct.Structure):
    _fields_ = [("tokens_milli", ct.c_uint32), ("depth_milli", ct.c_uint32), ("rate_per_s", ct.c_uint32),
                ("last_ms", ct.c_uint32), ("violations", ct.c_uint32), ("trip_after", ct.c_uint32),
                ("consecutive", ct.c_uint32), ("tripped", ct.c_bool)]


def _compiler() -> list[str] | None:
    cc = os.environ.get("CC")
    if cc:
        return cc.split()
    for c in ("gcc", "cc", "clang"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "cc"]
    except ImportError:
        return None


def build_lib(force: bool = False) -> Path:
    out = BUILD / LIBNAME
    if out.exists() and not force and out.stat().st_mtime >= max(SRC.stat().st_mtime, SRC.with_suffix(".h").stat().st_mtime):
        return out
    cc = _compiler()
    if cc is None:
        raise RuntimeError("no C compiler found (set CC, install gcc, or pip install ziglang)")
    BUILD.mkdir(exist_ok=True)
    flags = ["-std=c99", "-O2", "-Wall", "-Wextra", "-shared"] + ([] if sys.platform == "win32" else ["-fPIC"])
    subprocess.run(cc + flags + ["-o", str(out), str(SRC)], check=True)
    return out


_LIB = None


def load():
    global _LIB
    if _LIB is None:
        lib = ct.CDLL(str(build_lib()))
        u8p = ct.POINTER(ct.c_uint8)
        lib.e2e_crc8.argtypes = [u8p, ct.c_uint, ct.c_uint8]
        lib.e2e_crc8.restype = ct.c_uint8
        lib.e2e_tx_init.argtypes = [ct.POINTER(E2ETx), ct.c_uint16]
        lib.e2e_protect.argtypes = [ct.POINTER(E2ETx), u8p, ct.c_uint]
        lib.e2e_rx_init.argtypes = [ct.POINTER(E2ERx), ct.c_uint16, ct.c_uint8]
        lib.e2e_check.argtypes = [ct.POINTER(E2ERx), u8p, ct.c_uint]
        lib.e2e_check.restype = ct.c_int
        lib.rm_init.argtypes = [ct.POINTER(RM), ct.c_uint16, ct.c_uint32, ct.c_uint8]
        lib.rm_on_frame.argtypes = [ct.POINTER(RM), ct.c_uint, u8p, ct.c_uint, ct.c_uint32]
        lib.rm_on_frame.restype = ct.c_int
        lib.rm_tick.argtypes = [ct.POINTER(RM), ct.c_uint32]
        lib.rm_tick.restype = ct.c_int
        lib.rm_value_age.argtypes = [ct.POINTER(RM), ct.c_uint32]
        lib.rm_value_age.restype = ct.c_uint32
        lib.guard_init.argtypes = [ct.POINTER(Guard), ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32]
        lib.guard_allow.argtypes = [ct.POINTER(Guard), ct.c_uint32]
        lib.guard_allow.restype = ct.c_bool
        _LIB = lib
    return _LIB


def _buf(b: bytes | bytearray):
    return (ct.c_uint8 * len(b)).from_buffer_copy(bytes(b))


def crc8(data: bytes, start: int = 0) -> int:
    return load().e2e_crc8(_buf(data), len(data), start)


class Sender:
    def __init__(self, data_id: int):
        self.s = E2ETx()
        load().e2e_tx_init(ct.byref(self.s), data_id)

    def protect(self, payload: bytes) -> bytes:
        b = _buf(payload)
        load().e2e_protect(ct.byref(self.s), b, len(payload))
        return bytes(b)


class Receiver:
    def __init__(self, data_id: int, max_delta: int = 2):
        self.s = E2ERx()
        load().e2e_rx_init(ct.byref(self.s), data_id, max_delta)

    def check(self, pdu: bytes) -> str:
        return E2E_STATUS[load().e2e_check(ct.byref(self.s), _buf(pdu), len(pdu))]


class RedundancyManager:
    def __init__(self, data_id: int, timeout_ms: int, max_delta: int = 2):
        self.s = RM()
        load().rm_init(ct.byref(self.s), data_id, timeout_ms, max_delta)

    def on_frame(self, ch: int, pdu: bytes, now_ms: int) -> str:
        return E2E_STATUS[load().rm_on_frame(ct.byref(self.s), ch, _buf(pdu), len(pdu), now_ms)]

    def tick(self, now_ms: int) -> str:
        return RM_STATES[load().rm_tick(ct.byref(self.s), now_ms)]

    def age(self, now_ms: int) -> int:
        return load().rm_value_age(ct.byref(self.s), now_ms)

    @property
    def value(self) -> bytes:
        return bytes(self.s.value[: self.s.len])


class BusGuardian:
    def __init__(self, rate_per_s: int, depth: int, trip_after: int, now_ms: int = 0):
        self.s = Guard()
        load().guard_init(ct.byref(self.s), rate_per_s, depth, trip_after, now_ms)

    def allow(self, now_ms: int) -> bool:
        return bool(load().guard_allow(ct.byref(self.s), now_ms))

    @property
    def tripped(self) -> bool:
        return bool(self.s.tripped)
