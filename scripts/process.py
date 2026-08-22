#!/usr/bin/env python3
"""Turn raw runs from one YAML configuration into processed figure tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_hashes = {config_hash, *map(str, config.get("reuse_config_hashes", []))}
    processed = ROOT / "data" / "processed"
    figure_data = processed / "figure_data"
    processed.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)

    rows, profiles, green_rows, spectra = [], [], [], []
    for metadata_file in sorted((ROOT / "data" / "raw").glob("*/*/metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("config_hash") not in source_hashes or not _within_scope(metadata, config):
            continue
        directory = metadata_file.parent
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        observables = json.loads((directory / "observables.json").read_text(encoding="utf-8"))
        physical = metadata["physical"]
        row = {
            "run_id": metadata["run_id"], "study": metadata["study"], "kind": metadata["kind"],
            "config_hash": config_hash, "git_commit": metadata["git_commit"],
            "source_config_hash": metadata["config_hash"],
            "L": physical["L"], "U": physical["U"], "g": physical["g"],
            "q": physical["g"] * (physical["L"] - 1), "lambda": physical["lambda"],
            "chi": metadata.get("chi", physical["lambda"] * np.exp(physical["g"] * physical["L"])),
            "status": status["status"], "message": status["message"],
            "field_residual": status["field_residual"], "density_residual": status["density_residual"],
            "number_residual": status["number_residual"], "s_min": metadata.get("s_min", status["branch_overlap"]),
            "mu_evaluations": status["mu_evaluations"], "total_scf_iterations": status["total_scf_iterations"],
            "wall_seconds": status["wall_seconds"], "accepted": metadata.get("accepted", True),
            "requested": metadata.get("requested", True), "reason": metadata.get("reason", ""),
            "raw_directory": str(directory.relative_to(ROOT)), **observables,
        }
        for key, value in metadata.items():
            if key not in {"run_id", "study", "kind", "config_hash", "git_commit", "physical", "accepted", "requested", "reason", "s_min", "chi"}:
                row[key] = value
        rows.append(row)
        state_file = directory / "state.npz"
        if state_file.exists():
            state = np.load(state_file)
            for j, (plus, minus, up, down) in enumerate(zip(state["delta_plus"], state["delta_minus"], state["n_up"], state["n_down"])):
                profiles.append({"run_id": row["run_id"], "study": row["study"], "kind": row["kind"], "L": row["L"], "U": row["U"], "g": row["g"], "lambda": row["lambda"], "accepted": row["accepted"], "j": j, "delta_plus_abs": abs(plus), "delta_minus_abs": abs(minus), "P_real": (plus * minus).real, "P_imag": (plus * minus).imag, "density": up + down})
            for index, value in enumerate(state["eigenvalues"]):
                spectra.append({"run_id": row["run_id"], "study": row["study"], "kind": row["kind"], "L": row["L"], "U": row["U"], "g": row["g"], "lambda": row["lambda"], "accepted": row["accepted"], "mode": index, "E_real": value.real, "E_imag": value.imag})
        green_file = directory / "green.csv"
        if green_file.exists():
            frequency = pd.read_csv(green_file)
            frequency.insert(0, "run_id", row["run_id"])
            frequency.insert(1, "q", row["q"])
            frequency.insert(2, "g", row["g"])
            frequency.insert(3, "L", row["L"])
            green_rows.extend(frequency.to_dict("records"))

    data = pd.DataFrame(rows)
    if data.empty:
        raise SystemExit("No raw runs match this config hash.")
    data.to_csv(processed / "run_manifest.csv", index=False)
    data[(data["status"] != "SUCCESS") | (~data["accepted"].astype(bool))].to_csv(processed / "failed_runs.csv", index=False)
    pd.DataFrame(profiles).to_csv(figure_data / "profiles.csv", index=False)
    pd.DataFrame(green_rows).to_csv(figure_data / "green_frequency.csv", index=False)
    pd.DataFrame(spectra).to_csv(figure_data / "spectra.csv", index=False)
    for study in ("fig2", "conditioning", "green", "fig3", "fig4"):
        frame = data[data["study"] == study]
        frame.to_csv(processed / f"{study}.csv", index=False)
        frame.to_csv(figure_data / f"{study}.csv", index=False)
    thresholds = _thresholds(data)
    thresholds.to_csv(processed / "crossover_thresholds.csv", index=False)
    thresholds.to_csv(figure_data / "thresholds.csv", index=False)
    production_status = _production_status(data, thresholds, config)
    (figure_data / "production_status.json").write_text(json.dumps(production_status, indent=2), encoding="utf-8")
    snapshots = _fig4_snapshots(data)
    snapshots.to_csv(figure_data / "fig4_snapshots.csv", index=False)
    manifest = {"config_hash": config_hash, "source_config_hashes": sorted(source_hashes), "figures": {"fig02_obc_covariance": ["figure_data/fig2.csv", "figure_data/profiles.csv"], "fig03_weak_link_crossover": ["figure_data/fig3.csv", "figure_data/thresholds.csv"], "fig04_pbc_endpoint": ["figure_data/fig4.csv", "figure_data/profiles.csv", "figure_data/spectra.csv", "figure_data/fig4_snapshots.csv"], "figS1_conditioning": ["figure_data/conditioning.csv"], "figS2_green_covariance": ["figure_data/green.csv", "figure_data/green_frequency.csv"], "figS3_branch_quality": ["figure_data/fig3.csv", "figure_data/fig4.csv"]}}
    (processed / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PROCESS] {len(data)} raw rows -> {processed}")


def _thresholds(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = data[(data["kind"] == "branch_trial") & (data["status"] == "SUCCESS") & data["accepted"].astype(bool) & (data["s_min"] >= 0.7)]
    for keys, group in valid.groupby(["study", "L", "g", "U"]):
        for label, column, threshold in (("metric_1e-4", "metric_violation", 1.0e-4), ("metric_1e-3", "metric_violation", 1.0e-3), ("metric_1e-2", "metric_violation", 1.0e-2), ("pair", "delta_P_bulk", 1.0e-2), ("gamma", "gamma_max_over_t", 1.0e-4)):
            lam = _crossing(group, column, threshold)
            rows.append({"study": keys[0], "L": keys[1], "g": keys[2], "U": keys[3], "quantity": label, "threshold": threshold, "lambda_c": lam, "chi_c": lam * np.exp(keys[2] * keys[1]) if np.isfinite(lam) else np.nan})
    return pd.DataFrame(rows, columns=["study", "L", "g", "U", "quantity", "threshold", "lambda_c", "chi_c"])


def _production_status(data: pd.DataFrame, thresholds: pd.DataFrame, config: dict) -> dict:
    """State which main figures have every required, branch-valid input."""

    fig3_branches = config["studies"]["fig3"].get("branches", [])
    missing_fig3 = []
    for item in fig3_branches:
        subset = thresholds[
            (thresholds["study"] == "fig3")
            & (thresholds["L"] == int(item["L"]))
            & np.isclose(thresholds["g"], float(item["g"]))
            & (thresholds["quantity"].isin(["metric_1e-4", "metric_1e-3", "metric_1e-2"]))
        ]
        if len(subset) != 3 or not np.isfinite(subset["lambda_c"]).all():
            missing_fig3.append({"L": int(item["L"]), "g": float(item["g"])})

    fig4_valid = data[(data["study"] == "fig4") & (data["kind"] == "branch_trial") & (data["status"] == "SUCCESS") & data["accepted"].astype(bool) & (data["s_min"] >= 0.7) & np.isclose(data["lambda"], 1.0)]
    missing_fig4 = [int(L) for L in config["studies"]["fig4"]["L"] if fig4_valid[fig4_valid["L"] == int(L)].empty]
    return {
        "fig03_complete": not missing_fig3,
        "fig03_missing_metric_crossings": missing_fig3,
        "fig04_complete": not missing_fig4,
        "fig04_missing_pbc_endpoints": missing_fig4,
        "final_status": "COMPLETE" if not missing_fig3 and not missing_fig4 else "INCOMPLETE",
    }


def _within_scope(metadata: dict, config: dict) -> bool:
    """Keep only raw runs compatible with the final YAML parameter matrix."""

    study, kind, physical = metadata["study"], metadata["kind"], metadata["physical"]
    if study not in config["studies"]:
        return False
    L, U, g = int(physical["L"]), float(physical["U"]), float(physical["g"])
    section = config["studies"][study]
    close = lambda first, second: np.isclose(first, second, rtol=0.0, atol=1.0e-12)
    if study == "fig2":
        return L in section["L"] and close(U, section["U"]) and any(close(g * (L - 1), q) for q in section["q"])
    if study == "conditioning":
        if kind.startswith("high_precision"):
            return L == int(section["high_precision_L"]) and close(U, section["U"]) and any(close(g * (L - 1), q) for q in section["high_precision_q"])
        return L == int(section["L"]) and close(U, section["U"]) and any(close(g * (L - 1), q) for q in section["q"])
    if study == "green":
        return L == int(section["L"]) and close(U, section["U"]) and any(close(g * (L - 1), q) for q in section["q"])
    if study == "fig3":
        branches = section["branches"] if "branches" in section else [{"L": length, "g": value} for length in section["L"] for value in section["g"]]
        return close(U, section["U"]) and any(L == int(item["L"]) and close(g, item["g"]) for item in branches)
    if study == "fig4":
        if L not in section["L"] or not close(U, section["U"]):
            return False
        # Hermitian seeds and bandwidth controls deliberately have g=0.
        return kind in {"bandwidth_control", "hermitian_seed_direct", "cross_u_seed", "cross_u_reference", "hermitian_seed_failed"} or close(g, section["g"])
    return False


def _fig4_snapshots(data: pd.DataFrame) -> pd.DataFrame:
    """Choose OBC/near/above/PBC L=40 snapshots from valid accepted states."""

    valid = data[(data["study"] == "fig4") & (data["status"] == "SUCCESS") & (data["accepted"].astype(bool)) & (data["L"] == 40)]
    rows: list[dict] = []
    obc = valid[valid["kind"] == "obc_seed"]
    endpoint = valid[(valid["kind"] == "branch_trial") & np.isclose(valid["lambda"], 1.0)]
    branch = valid[(valid["kind"] == "branch_trial") & (valid["lambda"] > 0.0) & (valid["s_min"] >= 0.7)]
    selections = [("OBC", obc), ("PBC", endpoint)]
    for label, threshold in (("near", 1.0e-3), ("above", 1.0e-2)):
        finite = branch[np.isfinite(branch["metric_violation"]) & (branch["metric_violation"] > 0.0)]
        if not finite.empty:
            selections.append((label, finite.loc[[((np.log(finite["metric_violation"]) - np.log(threshold)).abs()).idxmin()]]))
    for label, frame in selections:
        if not frame.empty:
            row = frame.iloc[0].to_dict()
            row["snapshot"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def _crossing(group: pd.DataFrame, column: str, threshold: float) -> float:
    if column not in group:
        return float("nan")
    ordered = group.sort_values("lambda")
    x, y = ordered["lambda"].to_numpy(float), ordered[column].to_numpy(float)
    valid = (x > 0.0) & np.isfinite(y)
    x, y = x[valid], y[valid]
    crossed = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    if not len(crossed):
        return float("nan")
    i = int(crossed[0])
    return float(np.exp(np.log(x[i]) + (threshold - y[i]) * (np.log(x[i + 1]) - np.log(x[i])) / (y[i + 1] - y[i])))


if __name__ == "__main__":
    main()
