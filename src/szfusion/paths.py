"""Locations of data and outputs. Read from configs/paths.yaml (git-ignored; see configs/paths.example.yaml), each
key overridable by the environment variable SZF_<KEY>. No other module contains a path (RULES.md 5)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
KEYS = ("tusz_root", "chbmit_root", "work_dir", "dev_run_dir")


@lru_cache(maxsize=1)
def _cfg():
    f = REPO / "configs" / "paths.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else {}


def get(key: str) -> Path:
    if key not in KEYS:
        raise KeyError(key)
    v = os.environ.get(f"SZF_{key.upper()}") or _cfg().get(key)
    if not v:
        raise RuntimeError(f"path '{key}' is not set: add it to configs/paths.yaml or set SZF_{key.upper()}")
    return Path(v)


def work(*parts) -> Path:
    """A location under work_dir (created on demand)."""
    p = get("work_dir").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
