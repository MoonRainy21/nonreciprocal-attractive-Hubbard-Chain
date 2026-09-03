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

    rows, profiles, green_rows, spectra, filling_rows = [], [], [], [], []
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
            "git_dirty": metadata.get("git_dirty", True),
            "git_diff_sha256": metadata.get("git_diff_sha256", "UNKNOWN"),
            "source_config_hash": metadata["config_hash"],
            "L": physical["L"], "U": physical["U"], "g": physical["g"],
            "q": physical["g"] * (physical["L"] - 1), "lambda": physical["lambda"],
            "chi": metadata.get("chi", physical["lambda"] * np.exp(physical["g"] * physical["L"])),
            "status": status["status"], "message": status["message"],
            "field_residual": status["field_residual"], "density_residual": status["density_residual"],
            "number_residual": status["number_residual"], "s_min": metadata.get("s_min", status["branch_overlap"]),
            "max_density_imaginary": status.get("max_density_imaginary", np.nan),
            "total_density_imaginary": status.get("total_density_imaginary", np.nan),
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
            if row["study"] == "branch_audit" and row["kind"] == "filling_audit":
                filling_rows.append(
                    {
                        "run_id": row["run_id"],
                        "L": row["L"],
                        "g": row["g"],
                        "mu": row["mu"],
                        "audit_location": row.get("audit_location", ""),
                        "audit_index": row.get("audit_index", np.nan),
                        "achieved_filling": float(np.mean(state["n_up"] + state["n_down"])),
                    }
                )
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
    pd.DataFrame(filling_rows).to_csv(figure_data / "filling_audit.csv", index=False)
    # A physical branch has one canonical destination.  The full Fig. 4
    # continuation is canonical for U=2, g=0.05; Fig. 3 retains only g=0.10.
    canonical_fig3 = pd.concat([
        data[(data["study"] == "fig3") & ~np.isclose(data["g"], 0.05)],
        data[(data["study"] == "fig4") & np.isclose(data["g"], 0.05)].assign(study="fig3"),
    ], ignore_index=True)
    branch_ids = {(24, 0.05): "U2_L24_g0p05", (40, 0.05): "U2_L40_g0p05", (24, 0.10): "U2_L24_g0p10", (40, 0.10): "U2_L40_g0p10"}
    canonical_fig3["branch_id"] = [branch_ids.get((int(L), round(float(g), 2)), "") for L, g in zip(canonical_fig3["L"], canonical_fig3["g"])]
    for study in ("fig2", "conditioning", "green", "fig3", "fig4", "branch_audit"):
        frame = canonical_fig3 if study == "fig3" else data[data["study"] == study]
        frame.to_csv(processed / f"{study}.csv", index=False)
        frame.to_csv(figure_data / f"{study}.csv", index=False)
    threshold_data = pd.concat([data[data["study"].isin(["fig2", "conditioning", "green", "fig4"])], canonical_fig3], ignore_index=True)
    thresholds = _thresholds(threshold_data)
    thresholds.to_csv(processed / "crossover_thresholds.csv", index=False)
    thresholds.to_csv(figure_data / "thresholds.csv", index=False)
    threshold_sensitivity = _threshold_sensitivity(threshold_data)
    threshold_sensitivity.to_csv(processed / "threshold_sensitivity.csv", index=False)
    threshold_sensitivity.to_csv(figure_data / "threshold_sensitivity.csv", index=False)
    gamma_brackets = _gamma_brackets(canonical_fig3)
    gamma_brackets.to_csv(processed / "gamma_brackets.csv", index=False)
    gamma_brackets.to_csv(figure_data / "gamma_brackets.csv", index=False)
    collapse_quality = _collapse_quality(data[data["study"] == "branch_audit"])
    collapse_quality.to_csv(processed / "collapse_quality.csv", index=False)
    collapse_quality.to_csv(figure_data / "collapse_quality.csv", index=False)
    matched_gl_quality = _matched_gl_quality(data[data["study"] == "branch_audit"])
    matched_gl_quality.to_csv(processed / "matched_gl_quality.csv", index=False)
    matched_gl_quality.to_csv(figure_data / "matched_gl_quality.csv", index=False)
    paired_audit_verdict = _paired_audit_verdict(processed / "paired_route_audit.json")
    production_status = _production_status(data, thresholds, collapse_quality, matched_gl_quality, config, paired_audit_verdict)
    (figure_data / "production_status.json").write_text(json.dumps(production_status, indent=2), encoding="utf-8")
    snapshots = _fig4_snapshots(data)
    snapshots.to_csv(figure_data / "fig4_snapshots.csv", index=False)
    manifest = {"config_hash": config_hash, "source_config_hashes": sorted(source_hashes), "figures": {"fig02_obc_covariance": ["figure_data/fig2.csv", "figure_data/profiles.csv"], "fig03_weak_link_crossover": ["figure_data/fig3.csv", "figure_data/branch_audit.csv", "figure_data/collapse_quality.csv", "figure_data/matched_gl_quality.csv"], "fig04_pbc_endpoint": ["figure_data/fig4.csv", "figure_data/profiles.csv", "figure_data/spectra.csv", "figure_data/fig4_snapshots.csv"], "figS1_conditioning": ["figure_data/conditioning.csv"], "figS2_green_covariance": ["figure_data/green.csv", "figure_data/green_frequency.csv"], "figS3_branch_quality": ["figure_data/fig3.csv"], "figS4_projector_filling_audit": ["figure_data/branch_audit.csv", "figure_data/filling_audit.csv"]}, "audits": {"branch_and_filling": ["figure_data/branch_audit.csv"], "collapse_quality": ["figure_data/collapse_quality.csv"], "matched_gl_quality": ["figure_data/matched_gl_quality.csv"], "threshold_sensitivity": ["figure_data/threshold_sensitivity.csv"]}}
    (processed / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PROCESS] {len(data)} raw rows -> {processed}")


def _thresholds(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = data[(data["kind"] == "branch_trial") & (data["status"] == "SUCCESS") & data["accepted"].astype(bool) & (data["s_min"] >= 0.7)]
    for keys, group in valid.groupby(["study", "L", "g", "U"]):
        for label, column, threshold in (("metric_1e-4", "metric_violation", 1.0e-4), ("metric_1e-3", "metric_violation", 1.0e-3), ("metric_1e-2", "metric_violation", 1.0e-2), ("pair", "delta_P_bulk", 1.0e-2), ("gamma", "gamma_max_over_t", 1.0e-4)):
            lam = _crossing(group, column, threshold)
            rows.append({
                "study": keys[0], "L": keys[1], "g": keys[2], "U": keys[3], "quantity": label,
                "threshold": threshold, "lambda_c": lam,
                "chi_c": lam * np.exp(keys[2] * keys[1]) if np.isfinite(lam) else np.nan,
                "status": "CROSSED" if np.isfinite(lam) else "NOT_REACHED",
                "max_chi": float(group["chi"].max()),
            })
    return pd.DataFrame(rows, columns=["study", "L", "g", "U", "quantity", "threshold", "lambda_c", "chi_c", "status", "max_chi"])


def _gamma_brackets(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = data[(data["kind"] == "branch_trial") & (data["status"] == "SUCCESS") & data["accepted"].astype(bool) & (data["s_min"] >= 0.7)]
    for (L, g, U), group in valid.groupby(["L", "g", "U"]):
        group = group.sort_values("lambda")
        below = group[group["gamma_max_over_t"] < 1e-4].tail(1)
        above = group[group["gamma_max_over_t"] >= 1e-4].head(1)
        rows.append({"L": L, "g": g, "U": U, "status": "BRACKETED" if not below.empty and not above.empty else "NOT_REACHED", "lambda_below": below["lambda"].iloc[0] if not below.empty else np.nan, "lambda_above": above["lambda"].iloc[0] if not above.empty else np.nan, "chi_below": below["chi"].iloc[0] if not below.empty else np.nan, "chi_above": above["chi"].iloc[0] if not above.empty else np.nan})
    return pd.DataFrame(rows)


def _threshold_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Compare the 1% pair threshold across centered bulk windows."""

    rows = []
    valid = data[
        (data["kind"] == "branch_trial")
        & (data["status"] == "SUCCESS")
        & data["accepted"].astype(bool)
        & (data["s_min"] >= 0.7)
    ]
    columns = (
        ("one_third", "delta_P_bulk_one_third"),
        ("one_half", "delta_P_bulk"),
        ("two_thirds", "delta_P_bulk_two_thirds"),
        ("full", "delta_P_full"),
    )
    for (study, L, g, U), group in valid.groupby(["study", "L", "g", "U"]):
        for window, column in columns:
            value = _crossing(group, column, 1.0e-2)
            rows.append({
                "study": study,
                "L": L,
                "g": g,
                "U": U,
                "window": window,
                "lambda_c": value,
                "chi_c": value * np.exp(g * L) if np.isfinite(value) else np.nan,
                "status": "CROSSED" if np.isfinite(value) else "NOT_REACHED",
            })
    return pd.DataFrame(rows)


def _paired_audit_verdict(path: Path) -> str:
    """Read the saved paired-route verdict without repeating the audit solves."""

    if not path.exists():
        return "MISSING"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("final_verdict", "MISSING"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"


def _collapse_quality(data: pd.DataFrame, grid_points: int = 101) -> pd.DataFrame:
    """Quantify curve-to-curve scatter on common logarithmic coordinates."""

    valid = data[
        (data["kind"] == "branch_trial")
        & (data["status"] == "SUCCESS")
        & data["accepted"].astype(bool)
        & (data["metric_violation"] > 0.0)
    ]
    rows = []
    for coordinate in ("lambda", "chi"):
        curves = []
        for _, group in valid.groupby(["L", "g"]):
            group = group[(group[coordinate] > 0.0)].sort_values(coordinate).drop_duplicates(coordinate)
            if len(group) >= 2:
                curves.append(group)
        if len(curves) < 2:
            continue
        lower = max(float(group[coordinate].min()) for group in curves)
        upper = min(float(group[coordinate].max()) for group in curves)
        if not 0.0 < lower < upper:
            continue
        grid = np.linspace(np.log10(lower), np.log10(upper), grid_points)
        interpolated = np.vstack([
            np.interp(
                grid,
                np.log10(group[coordinate].to_numpy(float)),
                np.log10(group["metric_violation"].to_numpy(float)),
            )
            for group in curves
        ])
        scatter = np.std(interpolated, axis=0)
        rows.append({
            "coordinate": coordinate,
            "branch_count": len(curves),
            "common_min": lower,
            "common_max": upper,
            "mean_log10_std": float(np.mean(scatter)),
            "max_log10_std": float(np.max(scatter)),
        })
    return pd.DataFrame(rows)


def _matched_gl_quality(data: pd.DataFrame, grid_points: int = 101) -> pd.DataFrame:
    """Compare distinct ``(L,g)`` branches with the same accumulated ``gL``."""

    valid = data[
        (data["kind"] == "branch_trial")
        & (data["status"] == "SUCCESS")
        & data["accepted"].astype(bool)
        & (data["chi"] > 0.0)
        & (data["metric_violation"] > 0.0)
    ]
    groups = {
        (int(L), float(g)): group.sort_values("chi").drop_duplicates("chi")
        for (L, g), group in valid.groupby(["L", "g"])
    }
    rows = []
    keys = sorted(groups)
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            if first == second or not np.isclose(first[0] * first[1], second[0] * second[1], atol=1.0e-12):
                continue
            left, right = groups[first], groups[second]
            lower = max(float(left["chi"].min()), float(right["chi"].min()))
            upper = min(float(left["chi"].max()), float(right["chi"].max()))
            if not 0.0 < lower < upper:
                continue
            grid = np.linspace(np.log10(lower), np.log10(upper), grid_points)
            left_values = np.interp(grid, np.log10(left["chi"]), np.log10(left["metric_violation"]))
            right_values = np.interp(grid, np.log10(right["chi"]), np.log10(right["metric_violation"]))
            difference = np.abs(left_values - right_values)
            rows.append({
                "L_first": first[0],
                "g_first": first[1],
                "L_second": second[0],
                "g_second": second[1],
                "gL": first[0] * first[1],
                "common_chi_min": lower,
                "common_chi_max": upper,
                "mean_abs_log10_difference": float(np.mean(difference)),
                "max_abs_log10_difference": float(np.max(difference)),
            })
    return pd.DataFrame(rows)


def _production_status(data: pd.DataFrame, thresholds: pd.DataFrame, collapse_quality: pd.DataFrame, matched_gl_quality: pd.DataFrame, config: dict, paired_audit_verdict: str) -> dict:
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
    collapse = collapse_quality.set_index("coordinate")["mean_log10_std"] if not collapse_quality.empty else pd.Series(dtype=float)
    collapse_improved = bool({"lambda", "chi"} <= set(collapse.index) and collapse["chi"] < collapse["lambda"])
    matched_gl_complete = not matched_gl_quality.empty
    dirty = data.get("git_dirty", pd.Series(True, index=data.index)).map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).lower() == "true"
    )
    provenance_complete = bool(
        not data.empty
        and "git_dirty" in data
        and not dirty.any()
        and (data["git_commit"] != "UNKNOWN").all()
    )
    missing_audits = []
    audit_config = config["studies"].get("branch_audit", {})
    for item in audit_config.get("branches", []):
        L, g = int(item["L"]), float(item["g"])
        branch = data[(data["study"] == "branch_audit") & (data["L"] == L) & np.isclose(data["g"], g)]
        forward = branch[(branch["kind"] == "branch_trial") & np.isclose(branch["lambda"], 1.0) & (branch["status"] == "SUCCESS") & branch["accepted"].astype(bool)]
        reverse = branch[(branch["kind"] == "reverse_obc_endpoint") & np.isclose(branch["lambda"], 0.0) & (branch["status"] == "SUCCESS")]
        filling = branch[(branch["kind"] == "filling_audit") & (branch["status"] == "SUCCESS")]
        locations = set(filling.get("audit_location", pd.Series(dtype=str)).dropna())
        filling_ok = (
            locations == {"obc", "middle", "pbc"}
            and not filling.empty
            and filling["audit_monotone"].astype(bool).all()
            and (filling["audit_target_crossings"] == 1).all()
        )
        reverse_ok = (
            not reverse.empty
            and float(reverse["pair_product_error_to_endpoint"].iloc[-1]) < 1.0e-7
            and float(reverse["density_error_to_endpoint"].iloc[-1]) < 1.0e-7
            and float(reverse["projector_error_to_endpoint"].iloc[-1]) < 1.0e-7
        )
        pbc_ok = (
            not forward.empty
            and float(forward["global_conjugacy_violation"].iloc[-1]) < 1.0e-8
            and float(forward["max_pair_product_imaginary"].iloc[-1]) < 1.0e-8
            and float(forward["projector_idempotency"].iloc[-1]) < 1.0e-8
            and float(forward["biorthogonality_error"].iloc[-1]) < 1.0e-8
        )
        if not pbc_ok or not reverse_ok or not filling_ok:
            missing_audits.append({"L": L, "g": g})
    return {
        "fig03_complete": not missing_fig3,
        "fig03_missing_metric_crossings": missing_fig3,
        "fig04_complete": not missing_fig4,
        "fig04_missing_pbc_endpoints": missing_fig4,
        "branch_audits_complete": not missing_audits,
        "missing_branch_audits": missing_audits,
        "collapse_quality_complete": collapse_improved,
        "collapse_mean_log10_std": collapse.to_dict(),
        "matched_gl_complete": matched_gl_complete,
        "provenance_complete": provenance_complete,
        "paired_route_audit_required": True,
        "paired_route_audit_verdict": paired_audit_verdict,
        "final_status": "COMPLETE" if not missing_fig3 and not missing_fig4 and not missing_audits and collapse_improved and matched_gl_complete and provenance_complete and paired_audit_verdict in {"PASS", "PASS_WITH_NOTE"} else "INCOMPLETE",
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
    if study == "branch_audit":
        return close(U, section["U"]) and any(
            L == int(item["L"]) and close(g, item["g"])
            for item in section["branches"]
        )
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
