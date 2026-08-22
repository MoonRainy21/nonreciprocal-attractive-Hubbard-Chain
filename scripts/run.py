#!/usr/bin/env python3
"""Run one config-defined numerical study; plotting is intentionally separate."""

from __future__ import annotations

import argparse
from pathlib import Path

from nhbdg.studies import load_config, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", default="all", choices=["all", "fig2", "conditioning", "green", "fig3", "fig4"])
    parser.add_argument("--L", type=float, action="append", default=[], help="Optional chain-length filter for a remaining-study run.")
    parser.add_argument("--g", type=float, action="append", default=[], help="Optional nonreciprocity filter for a remaining-study run.")
    parser.add_argument("--U", type=float, action="append", default=[], help="Optional interaction filter for a remaining-study run.")
    args = parser.parse_args()
    run(load_config(args.config), args.study, {"L": set(args.L), "g": set(args.g), "U": set(args.U)})


if __name__ == "__main__":
    main()
