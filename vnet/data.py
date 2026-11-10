"""Load the YAML data set (data/*.yaml) and check that every non-assumed number resolves to a reference.

Every engineering value in data/ is a mapping with `value` and `basis` (sourced / derived / assumed) or a
record with `basis` and, unless assumed, `src: [KEY, ...]` keys that must exist in data/references.yaml.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILES = ("vehicle", "messages", "architectures", "environment", "ethernet", "chains", "rma", "aoa", "fmea")
BASES = {"sourced", "derived", "assumed"}


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    return yaml.safe_load((DATA / f"{name}.yaml").read_text(encoding="utf-8"))


def references() -> dict:
    return load("references")


def values(name: str, key: str) -> dict:
    """Flatten {k: {value: v, ...}} to {k: v}."""
    return {k: (x["value"] if isinstance(x, dict) and "value" in x else x) for k, x in load(name)[key].items()}


def check_sources() -> list[str]:
    refs = references()
    bad: list[str] = []
    for name in FILES:
        doc = load(name)

        def walk(x, path, name=name):
            if isinstance(x, dict):
                b = x.get("basis")
                if b is not None and b not in BASES:
                    bad.append(f"{name}:{path}: basis '{b}' not in {sorted(BASES)}")
                if b == "sourced" and not x.get("src"):
                    bad.append(f"{name}:{path}: sourced value without src")
                for k in x.get("src", []) or []:
                    if k not in refs:
                        bad.append(f"{name}:{path}: unknown source {k}")
                for k, v in x.items():
                    walk(v, f"{path}.{k}", name)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]", name)
        walk(doc, "")
    return bad
