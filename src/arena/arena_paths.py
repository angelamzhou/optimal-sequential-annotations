"""Shared paths and small I/O helpers for the LMArena pipeline.

All LMArena inputs and outputs live under one data directory, ARENA_DATA (environment variable;
default: <repo>/data/arena). It holds the public inputs you rebuild from Hugging Face
(arena140k.parquet, arena_rm_scores.parquet, arena_text_hash/; see README), the derived session
manifest written by build_arena_manifest.py, and the Monte Carlo run directories.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ARENA_DATA", REPO / "data" / "arena"))


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_once(path: Path, value) -> None:
    """Write JSON once; refuse to overwrite a different artifact."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to overwrite different artifact: {path.name}")
    else:
        path.write_text(text)
