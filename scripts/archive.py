#!/usr/bin/env python3
"""Create a checksummed publication archive from a completed clean run."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "nhbdg-reproducibility.tar.gz"
INCLUDE = ("configs", "src", "scripts", "tests", "data/raw", "data/processed", "figures")


def main() -> None:
    status_path = ROOT / "data" / "processed" / "figure_data" / "production_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("final_status") != "COMPLETE" or not status.get("provenance_complete"):
        raise SystemExit("Refusing to archive an incomplete or dirty production run.")
    files = sorted(
        path
        for name in INCLUDE
        for path in (ROOT / name).rglob("*")
        if path.is_file()
    )
    manifest = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest_path = ROOT / "artifacts" / "SHA256SUMS.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT))
        archive.add(manifest_path, arcname=manifest_path.relative_to(ROOT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
