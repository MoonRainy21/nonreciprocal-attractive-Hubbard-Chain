#!/usr/bin/env python3
"""Remove only generated numerical outputs before a fresh reproduction run."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "figures", ROOT / "logs")


def main() -> None:
    """Empty the four Git-ignored output directories and recreate them."""

    for directory in OUTPUTS:
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    print("[CLEAN] removed generated raw data, processed data, figures, and logs", flush=True)


if __name__ == "__main__":
    main()
