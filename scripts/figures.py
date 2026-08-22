#!/usr/bin/env python3
"""Create numerical figures from processed tables; this script never runs HFB."""

from __future__ import annotations

import argparse
from pathlib import Path

from nhbdg.figures import make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", default="all", choices=["all", "fig02", "fig03", "fig04", "figS1", "figS2", "figS3"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    make(root / "data" / "processed" / "figure_data", root / "figures", args.figure)


if __name__ == "__main__":
    main()
