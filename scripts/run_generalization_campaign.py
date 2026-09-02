#!/usr/bin/env python3
"""Run independent generalization cases in parallel until a wall-clock deadline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generalization.yaml")
    parser.add_argument("--deadline", required=True, help="Local ISO timestamp, e.g. 2026-08-31T09:45:00")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--python", default=str(ROOT / ".venv312/bin/python"))
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="U:FILLING",
        help="Run only this interaction/filling case; may be supplied repeatedly.",
    )
    parser.add_argument(
        "--resume-missing",
        type=Path,
        metavar="COVERAGE_CSV",
        help="Run only slices with no accepted trials and no recorded Hermitian-seed failure.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    deadline = datetime.fromisoformat(args.deadline).timestamp()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cases = [(float(case["U"]), float(case["filling"])) for case in config["studies"]["generalization"]["cases"]]
    if args.case:
        selected = set()
        for value in args.case:
            try:
                U_text, filling_text = value.split(":", maxsplit=1)
                selected.add((float(U_text), float(filling_text)))
            except ValueError as error:
                raise SystemExit(f"invalid --case {value!r}; expected U:FILLING") from error
        unknown = selected.difference(cases)
        if unknown:
            raise SystemExit(f"requested cases are absent from the config: {sorted(unknown)}")
        cases = [case for case in cases if case in selected]
    branches = [
        (int(branch["L"]), float(branch["g"]))
        for branch in config["studies"]["generalization"]["branches"]
    ]
    jobs = [(U, filling, L, g) for U, filling in cases for L, g in branches]
    if args.resume_missing:
        with args.resume_missing.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        missing = {
            (float(row["U"]), float(row["filling"]), int(row["L"]), float(row["g"]))
            for row in rows
            if int(row["accepted_trials"]) == 0
            and row["hermitian_seed_failed"].strip().lower() != "true"
        }
        jobs = [job for job in jobs if job in missing]
    log_dir = ROOT / "logs/generalization"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "campaign_state.json"
    running: dict[subprocess.Popen[bytes], tuple[float, float, int, float, object, float]] = {}
    completed: list[dict[str, object]] = []

    def save_state() -> None:
        state = {
            "deadline": args.deadline,
            "completed": completed,
            "running": [
                {"U": U, "filling": filling, "L": L, "g": g, "started_at": started}
                for U, filling, L, g, _, started in running.values()
            ],
            "remaining": [
                {"U": U, "filling": filling, "L": L, "g": g}
                for U, filling, L, g in jobs
            ],
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        while jobs or running:
            while jobs and len(running) < args.workers and time.time() < deadline:
                U, filling, L, g = jobs.pop(0)
                log_path = log_dir / f"U{U:g}_n{filling:g}_L{L}_g{g:g}.log"
                handle = log_path.open("wb")
                command = [
                    args.python,
                    str(ROOT / "scripts/run.py"),
                    "--config", str(args.config),
                    "--study", "generalization",
                    "--U", str(U),
                    "--filling", str(filling),
                    "--L", str(L),
                    "--g", str(g),
                ]
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "src"),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[process] = (U, filling, L, g, handle, time.time())
                print(f"[START] U={U:g} filling={filling:g} L={L} g={g:g} pid={process.pid}", flush=True)
                save_state()

            for process, (U, filling, L, g, handle, started) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                completed.append({
                    "U": U,
                    "filling": filling,
                    "L": L,
                    "g": g,
                    "return_code": return_code,
                    "wall_seconds": time.time() - started,
                })
                del running[process]
                print(f"[DONE] U={U:g} filling={filling:g} L={L} g={g:g} rc={return_code}", flush=True)
                save_state()

            if time.time() >= deadline:
                for process, (U, filling, L, g, handle, started) in list(running.items()):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    handle.close()
                    completed.append({
                        "U": U,
                        "filling": filling,
                        "L": L,
                        "g": g,
                        "return_code": process.returncode,
                        "wall_seconds": time.time() - started,
                        "stopped_at_deadline": True,
                    })
                    del running[process]
                save_state()
                break
            time.sleep(5)
    finally:
        for process, (_, _, _, _, handle, _) in running.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            handle.close()
        save_state()


if __name__ == "__main__":
    main()
