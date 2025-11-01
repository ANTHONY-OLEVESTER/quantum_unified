#!/usr/bin/env python3
"""Regenerate Phase-II universality figures from the cached plot data."""

from __future__ import annotations

import pickle
from pathlib import Path

from phase2_plot_utils import plot_phase2_results


def main() -> None:
    cache_path = Path("phase2_plot_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Plot cache '{cache_path}' not found. Run phase2.py with --sweep first."
        )
    with cache_path.open("rb") as fh:
        cache = pickle.load(fh)
    saved = plot_phase2_results(cache, Path("."))
    pngs = [path.name for path in saved if path.suffix == ".png"]
    if pngs:
        print("Regenerated figures:", ", ".join(pngs))


if __name__ == "__main__":
    main()
