#!/usr/bin/env python3
"""Summarize a deadline-limited generalization campaign from saved raw records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.process import _collapse_quality, _matched_gl_quality


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generalization.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    rows: list[dict] = []
    for metadata_file in sorted((ROOT / "data/raw/generalization").glob("*/metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("config_hash") != config_hash:
            continue
        directory = metadata_file.parent
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        observables = json.loads((directory / "observables.json").read_text(encoding="utf-8"))
        physical = metadata["physical"]
        rows.append({
            "run_id": metadata["run_id"],
            "kind": metadata["kind"],
            "L": int(physical["L"]),
            "U": float(physical["U"]),
            "filling": float(physical["filling"]),
            "g": float(physical["g"]),
            "target_g": float(metadata.get("target_g", physical["g"])),
            "lambda": float(physical["lambda"]),
            "chi": float(metadata.get("chi", physical["lambda"] * np.exp(physical["g"] * physical["L"]))),
            "status": status["status"],
            "message": status["message"],
            "accepted": bool(metadata.get("accepted", True)),
            "s_min": float(metadata.get("s_min", status.get("branch_overlap", np.nan))),
            "metric_violation": float(observables.get("metric_violation", np.nan)),
        })

    data = pd.DataFrame(rows)
    if data.empty:
        raise SystemExit("No generalization records match the configuration hash.")

    processed = ROOT / "data/processed/generalization"
    processed.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed / "run_manifest.csv", index=False)

    collapse_frames = []
    matched_frames = []
    coverage_rows = []
    cases = config["studies"]["generalization"]["cases"]
    branches = config["studies"]["generalization"]["branches"]
    for case in cases:
        U, filling = float(case["U"]), float(case["filling"])
        case_data = data[np.isclose(data["U"], U) & np.isclose(data["filling"], filling)]
        collapse = _collapse_quality(case_data)
        if not collapse.empty:
            collapse.insert(0, "filling", filling)
            collapse.insert(0, "U", U)
            collapse_frames.append(collapse)
        matched = _matched_gl_quality(case_data)
        if not matched.empty:
            matched.insert(0, "filling", filling)
            matched.insert(0, "U", U)
            matched_frames.append(matched)
        for branch in branches:
            L, g = int(branch["L"]), float(branch["g"])
            subset = case_data[(case_data["L"] == L) & np.isclose(case_data["target_g"], g)]
            valid = subset[(subset["kind"] == "branch_trial") & (subset["status"] == "SUCCESS") & subset["accepted"]]
            coverage_rows.append({
                "U": U,
                "filling": filling,
                "L": L,
                "g": g,
                "gL": L * g,
                "saved_records": len(subset),
                "accepted_trials": len(valid),
                "max_lambda": float(valid["lambda"].max()) if not valid.empty else np.nan,
                "hermitian_seed_failed": bool((subset["kind"] == "hermitian_seed_failed").any()),
            })

    collapse = pd.concat(collapse_frames, ignore_index=True) if collapse_frames else pd.DataFrame()
    matched = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    collapse.to_csv(processed / "collapse_quality.csv", index=False)
    matched.to_csv(processed / "matched_gl_quality.csv", index=False)
    coverage.to_csv(processed / "branch_coverage.csv", index=False)

    comparison = {}
    for (U, filling), group in collapse.groupby(["U", "filling"]):
        scores = group.set_index("coordinate")["mean_log10_std"].to_dict()
        comparison[f"U={U:g},filling={filling:g}"] = {
            "lambda_scatter": scores.get("lambda"),
            "chi_scatter": scores.get("chi"),
            "chi_improves_collapse": bool(scores.get("chi", np.inf) < scores.get("lambda", -np.inf)),
        }
    summary = {
        "raw_records": len(data),
        "cases_with_records": int(data[["U", "filling"]].drop_duplicates().shape[0]),
        "expected_cases": len(cases),
        "branches_with_records": int((coverage["saved_records"] > 0).sum()),
        "expected_branches": len(cases) * len(branches),
        "hermitian_seed_failures": int((data["kind"] == "hermitian_seed_failed").sum()),
        "collapse_by_case": comparison,
        "matched_gL_comparisons": len(matched),
    }
    (processed / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
