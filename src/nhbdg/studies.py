"""Config-driven numerical studies and raw-state provenance writing."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .continuation import Branch, continue_branch
from .fixed_filling import audit_filling_curve, solve_fixed_filling
from .model import (
    Chain,
    Numerics,
    bdg_matrix,
    map_fields_from_hermitian,
    nambu_similarity,
    pbc_blocks,
)
from .observables import (
    METRIC_EPSILON,
    align_nambu_scale,
    complex_fraction,
    gamma_max,
    global_conjugacy_violation,
    green,
    green_covariance_error,
    metric_violation,
    min_spectrum_separation,
    pair_deformation,
    pair_deformation_window,
    particle_block,
    projector_diagnostics,
    relative_error,
    spectrum_distance,
    stripped_propagator,
)
from .solver import HFBState, MeanFieldSolver, occupied_projector

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    """Read one YAML input file without inserting study parameters in Python."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(config: dict[str, Any], study: str, filters: dict[str, set[float]] | None = None) -> None:
    """Run one named study, or every study in the paper configuration."""

    actions = {
        "fig2": run_fig2,
        "conditioning": run_conditioning,
        "green": run_green,
        "fig3": run_fig3,
        "fig4": run_fig4,
        "branch_audit": run_branch_audit,
        "generalization": run_generalization,
    }
    requested = [name for name in actions if name in config["studies"]] if study == "all" else [study]
    for name in requested:
        if name not in actions:
            raise ValueError(f"Unknown study {name!r}; choose {sorted(actions)} or 'all'.")
        print(f"[STUDY {name}] started", flush=True)
        actions[name](config, filters or {})
        print(f"[STUDY {name}] finished", flush=True)


def run_fig2(config: dict[str, Any], _: dict[str, set[float]]) -> None:
    """Study A/B: Hermitian HFB and OBC self-consistent covariance."""

    numeric, section = _numerics(config), config["studies"]["fig2"]
    for L in section["L"]:
        reference = _fixed_state(Chain(int(L), float(section["U"]), filling=float(config["common"]["filling"])), numeric, "hermitian")
        _save(config, "fig2", reference, "hermitian", {"q": 0.0})
        if not reference.converged:
            print(f"  [fig2] L={L}: Hermitian reference failed; OBC covariance points skipped", flush=True)
            continue
        for q in section["q"]:
            g = float(q) / (int(L) - 1)
            mapped = _map_obc(reference, g)
            raw_solver = MeanFieldSolver(mapped.chain, numeric, "raw")
            new_plus, new_minus = raw_solver.gap_map_once(mapped.mu, mapped.n_up, mapped.n_down, mapped.delta_plus, mapped.delta_minus)
            map_error = max(relative_error(new_plus, mapped.delta_plus), relative_error(new_minus, mapped.delta_minus))
            _save(config, "fig2", mapped, "mapped", {"q": float(q), "map_error": map_error})
            for representation in ("raw", "rescaled"):
                state = solve_fixed_filling(MeanFieldSolver(mapped.chain, numeric, representation), initial_state=mapped)
                comparison = _fig2_comparison(mapped, state)
                _save(
                    config, "fig2", state, representation,
                    {
                        "q": float(q),
                        "map_error": map_error,
                        **comparison,
                    },
                )


def run_conditioning(config: dict[str, Any], _: dict[str, set[float]]) -> None:
    """Study C: raw/rescaled conditioning and selected high-precision checks."""

    numeric, section = _numerics(config), config["studies"]["conditioning"]
    L, U = int(section["L"]), float(section["U"])
    reference = _fixed_state(Chain(L, U, filling=float(config["common"]["filling"])), numeric, "hermitian")
    _save(config, "conditioning", reference, "hermitian_reference", {"q": 0.0})
    if not reference.converged:
        print("  [conditioning] Hermitian reference failed; covariance scan skipped", flush=True)
        return
    reference_projector = occupied_projector(reference.eigensystem)
    for q in section["q"]:
        mapped = _map_obc(reference, float(q) / (L - 1))
        V = nambu_similarity(mapped.chain)
        inverse = np.diag(1.0 / np.diag(V))
        expected = inverse @ reference_projector @ V
        for representation in ("raw", "rescaled"):
            state = solve_fixed_filling(MeanFieldSolver(mapped.chain, numeric, representation), initial_state=mapped)
            valid = _comparable(state, reference)
            projector = occupied_projector(state.eigensystem)
            _save(
                config, "conditioning", state, representation,
                {
                    "q": float(q),
                    "covariance_error": relative_error(projector, expected) if valid else float("nan"),
                    "spectrum_error": _spectrum_error(state, reference),
                    "similarity_condition": float(np.linalg.cond(V)),
                    "right_condition": float(np.linalg.cond(state.eigensystem.right)) if state.eigensystem.right.size else float("inf"),
                },
            )
    for q in section["high_precision_q"]:
        small_L = int(section["high_precision_L"])
        small = _fixed_state(Chain(small_L, U, filling=float(config["common"]["filling"])), numeric, "hermitian")
        _save(config, "conditioning", small, "high_precision_reference", {"q": float(q)})
        if not small.converged:
            continue
        mapped = _map_obc(small, float(q) / (small_L - 1))
        raw = bdg_matrix(mapped.chain, mapped.mu, mapped.n_up, mapped.n_down, mapped.delta_plus, mapped.delta_minus)
        V = nambu_similarity(mapped.chain)
        expected = bdg_matrix(small.chain, small.mu, small.n_up, small.n_down, small.delta_plus, small.delta_minus)
        _save(config, "conditioning", mapped, "high_precision", {"q": float(q), "mp_similarity_error": _mp_similarity(raw, expected, V, int(section["precision"]))})


def run_green(config: dict[str, Any], _: dict[str, set[float]]) -> None:
    """Study D: OBC local covariance and nonlocal directional reweighting."""

    numeric, section = _numerics(config), config["studies"]["green"]
    L = int(section["L"])
    reference = _fixed_state(Chain(L, float(section["U"]), filling=float(config["common"]["filling"])), numeric, "hermitian")
    _save(config, "green", reference, "hermitian_reference", {"q": 0.0})
    if not reference.converged:
        print("  [green] Hermitian reference failed; Green-function scan skipped", flush=True)
        return
    H0 = bdg_matrix(reference.chain, reference.mu, reference.n_up, reference.n_down, reference.delta_plus, reference.delta_minus)
    omega = np.linspace(float(section["omega_min"]), float(section["omega_max"]), int(section["n_omega"]))
    for q in section["q"]:
        mapped = _map_obc(reference, float(q) / (L - 1))
        Hg = bdg_matrix(mapped.chain, mapped.mu, mapped.n_up, mapped.n_down, mapped.delta_plus, mapped.delta_minus)
        rows, errors, center = [], [], L // 2
        for value in omega:
            G0, Gg = green(H0, float(value), float(section["eta"])), green(Hg, float(value), float(section["eta"]))
            pp0, ppg = particle_block(G0), particle_block(Gg)
            errors.append(green_covariance_error(Gg, G0, mapped.chain))
            rows.append({
                "omega": float(value),
                "center_ldos": float(-np.imag(ppg[center, center]) / np.pi),
                "edge_ldos": float(-np.imag(ppg[0, 0]) / np.pi),
                "local_difference": float(abs(ppg[center, center] - pp0[center, center])),
                "G_1L_abs": float(abs(ppg[0, -1])), "G_L1_abs": float(abs(ppg[-1, 0])),
                "stripped_1L_abs": float(abs(stripped_propagator(ppg[0, -1], 0, L - 1, mapped.chain))),
                "covariance_error": errors[-1],
            })
        directory = _save(config, "green", mapped, "green", {"q": float(q), "green_covariance_max": float(max(errors))})
        with (directory / "green.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def run_fig3(config: dict[str, Any], filters: dict[str, set[float]]) -> None:
    """Study E1: only the configured crossover branches, without PBC continuation."""

    numeric, section = _numerics(config), config["studies"]["fig3"]
    for item in _fig3_branches(section):
        L, g = int(item["L"]), float(item["g"])
        if not _selected(filters, "L", L) or not _selected(filters, "g", g):
            continue
        if _has_metric_crossings(config, "fig3", L, float(section["U"]), g):
            print(f"  [fig3] L={L} g={g:g}: reusing validated crossover branch", flush=True)
            continue
        # Figure 2 already supplies the converged Hermitian saddle at every
        # requested size.  Mapping that state is the defined OBC seed and
        # avoids re-solving an identical fixed-filling problem here.
        reference = _u2_hermitian_reference(config, L, numeric)
        if reference is not None:
            obc = _map_obc(reference, g)
            seed_method = "fig2_hermitian_map"
        else:
            obc = _fixed_state(Chain(L, float(section["U"]), g=g, filling=float(config["common"]["filling"])), numeric, "rescaled")
            seed_method = "direct_obc"
        branch = _threshold_branch(
            obc,
            numeric,
            section,
            full_pbc=False,
            lambda_min=float(item.get("lambda_min", 1.0e-12)),
        )
        _save_branch(config, "fig3", branch, seed_method=seed_method)


def run_fig4(config: dict[str, Any], filters: dict[str, set[float]]) -> None:
    """Study E2--E5: selected L=24,40 branches through the PBC endpoint."""

    numeric, section = _numerics(config), config["studies"]["fig4"]
    for L in section["L"]:
        if not _selected(filters, "L", int(L)):
            continue
        if _fig4_checks_exist(config, int(L), float(section["U"]), float(section["g"])):
            print(f"  [fig4] L={L}: reusing branch, PBC seed checks, and momentum benchmark", flush=True)
            continue
        endpoint = _load_endpoint(config, "fig4", int(L), float(section["U"]), float(section["g"]), numeric)
        if endpoint is not None:
            print(f"  [fig4] L={L}: reusing branch and adding missing endpoint checks", flush=True)
            _endpoint_checks(config, "fig4", endpoint, numeric)
            continue
        obc = _fixed_state(Chain(int(L), float(section["U"]), g=float(section["g"]), filling=float(config["common"]["filling"])), numeric, "rescaled")
        branch = _threshold_branch(obc, numeric, section, full_pbc=True)
        _save_branch(config, "fig4", branch, seed_method="direct_obc")
        if branch.accepted[-1].chain.lambda_ == 1.0:
            _endpoint_checks(config, "fig4", branch.accepted[-1], numeric)


def run_branch_audit(config: dict[str, Any], filters: dict[str, set[float]]) -> None:
    """Audit full forward/reverse branches and local fixed-filling curves."""

    numeric, section = _numerics(config), config["studies"]["branch_audit"]
    for item in section["branches"]:
        L, g, U = int(item["L"]), float(item["g"]), float(section["U"])
        if not _selected(filters, "L", L) or not _selected(filters, "g", g):
            continue
        obc = _fixed_state(Chain(L, U, g=g, filling=float(config["common"]["filling"])), numeric, "rescaled")
        forward_targets = [
            *np.geomspace(1.0e-12, 1.0, int(section["continuation_points"])).tolist(),
        ]
        forward = continue_branch(obc, forward_targets, numeric)
        _save_branch(config, "branch_audit", forward, seed_method="direct_obc")
        if forward.terminated or not np.isclose(forward.accepted[-1].chain.lambda_, 1.0):
            continue

        endpoint = forward.accepted[-1]
        reverse_targets = [
            state.chain.lambda_ for state in reversed(forward.accepted[:-1])
        ]
        reverse = continue_branch(endpoint, reverse_targets, numeric)
        _save_reverse_branch(config, reverse, obc)

        middle = min(forward.accepted, key=lambda state: abs(state.chain.lambda_ - 0.1))
        for location, reference in (("obc", obc), ("middle", middle), ("pbc", endpoint)):
            solver = MeanFieldSolver(reference.chain, numeric, "rescaled")
            audit = audit_filling_curve(
                solver,
                reference,
                half_width=float(section["filling_audit_half_width"]),
                points=int(section["filling_audit_points"]),
            )
            for index, state in enumerate(audit.states):
                _save(
                    config,
                    "branch_audit",
                    state,
                    "filling_audit",
                    {
                        "audit_location": location,
                        "audit_index": index,
                        "audit_monotone": audit.monotone,
                        "audit_target_crossings": audit.target_crossings,
                        "audit_minimum_branch_overlap": audit.minimum_branch_overlap,
                        "audit_maximum_density_imaginary": audit.maximum_density_imaginary,
                    },
                )


def run_generalization(config: dict[str, Any], filters: dict[str, set[float]]) -> None:
    """Scan interaction and filling dependence of the weak-link crossover."""

    numeric, section = _numerics(config), config["studies"]["generalization"]
    for case in section["cases"]:
        U, filling = float(case["U"]), float(case["filling"])
        if not _selected(filters, "U", U) or not _selected(filters, "filling", filling):
            continue
        references: dict[int, HFBState] = {}
        for item in section["branches"]:
            L, g = int(item["L"]), float(item["g"])
            if not _selected(filters, "L", L) or not _selected(filters, "g", g):
                continue
            if L not in references:
                references[L] = _fixed_state(Chain(L, U, filling=filling), numeric, "hermitian")
            hermitian = references[L]
            if not hermitian.converged:
                _save(
                    config,
                    "generalization",
                    hermitian,
                    "hermitian_seed_failed",
                    {"target_g": g},
                )
                continue
            obc = _map_obc(hermitian, g)
            branch = _threshold_branch(
                obc,
                numeric,
                section,
                full_pbc=False,
                lambda_min=float(item.get("lambda_min", 1.0e-12)),
            )
            _save_branch(config, "generalization", branch, seed_method="direct_obc")



def _threshold_branch(
    obc: HFBState,
    numeric: Numerics,
    section: dict[str, Any],
    full_pbc: bool,
    lambda_min: float = 1.0e-12,
) -> Branch:
    """Track a coarse branch, refining only intervals containing a threshold.

    ``lambda_min`` is a study-configured lower continuation bound.  The OBC
    point is always retained separately; a larger lower bound is appropriate
    when prior finite-size data already establish that all requested
    threshold crossings occur above it.
    """

    cap = 1.0 if full_pbc else float(section["crossover_lambda_cap"])
    if not 0.0 < lambda_min < cap:
        raise ValueError("lambda_min must lie strictly between zero and the branch cap.")
    coarse = continue_branch(
        obc,
        np.geomspace(lambda_min, cap, int(section["coarse_lambda_points"])).tolist(),
        numeric,
        max_consecutive_scf_failures=section.get("coarse_scf_failure_limit"),
    )
    if coarse.terminated:
        # A failed coarse segment cannot establish a threshold bracket, so
        # local refinement would add cost without producing a valid crossing.
        return coarse
    accepted = {state.chain.lambda_: state for state in coarse.accepted}
    trials = list(coarse.trials)
    messages = [coarse.message]
    for lower, upper in _crossing_intervals(coarse.accepted, obc):
        interior = np.geomspace(max(lower.chain.lambda_, 1.0e-14), upper.chain.lambda_, int(section["refinement_points"]) + 2)[1:-1]
        refined = continue_branch(
            lower,
            [*interior, upper.chain.lambda_],
            numeric,
            max_consecutive_scf_failures=section.get("refinement_scf_failure_limit"),
        )
        trials.extend(refined.trials)
        accepted.update({state.chain.lambda_: state for state in refined.accepted})
        messages.append(refined.message)
    return Branch(
        [accepted[key] for key in sorted(accepted)],
        trials,
        coarse.terminated or any(message.startswith("REFINEMENT_STOP") for message in messages),
        "; ".join(dict.fromkeys(messages)),
    )


def _crossing_intervals(states: list[HFBState], reference: HFBState) -> list[tuple[HFBState, HFBState]]:
    intervals: dict[tuple[float, float], tuple[HFBState, HFBState]] = {}
    for lower, upper in pairwise(states):
        if upper.chain.lambda_ <= 0.0:
            continue
        for low, high in zip(_threshold_values(lower, reference), _threshold_values(upper, reference)):
            if np.isfinite(low) and np.isfinite(high) and low <= 0.0 <= high:
                intervals[(lower.chain.lambda_, upper.chain.lambda_)] = (lower, upper)
    return list(intervals.values())


def _threshold_values(state: HFBState, reference: HFBState) -> list[float]:
    values = [gamma_max(state) - 1.0e-4]
    if state.chain.U == 0.0:
        return values
    values.extend([metric_violation(state)[0] - value for value in (1.0e-4, 1.0e-3, 1.0e-2)])
    values.append(pair_deformation(state, reference, bulk=True) - 1.0e-2)
    return values


def _fig3_branches(section: dict[str, Any]) -> list[dict[str, float]]:
    """Read explicit Fig. 3 branch pairs, with a compact legacy fallback."""

    if "branches" in section:
        branches: list[dict[str, float]] = []
        for item in section["branches"]:
            branch = {"L": int(item["L"]), "g": float(item["g"])}
            if "lambda_min" in item:
                branch["lambda_min"] = float(item["lambda_min"])
            branches.append(branch)
        return branches
    return [{"L": int(L), "g": float(g)} for L in section["L"] for g in section["g"]]


def _selected(filters: dict[str, set[float]], key: str, value: float) -> bool:
    """Return true when an optional CLI filter permits one physical value."""

    selected = filters.get(key, set())
    return not selected or any(np.isclose(value, candidate, rtol=0.0, atol=1.0e-12) for candidate in selected)


def _source_hashes(config: dict[str, Any]) -> set[str]:
    """Return the current and explicitly declared compatible raw-data hashes."""

    return {_hash(config), *map(str, config.get("reuse_config_hashes", []))}


def _raw_records(config: dict[str, Any], study: str) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Load raw metadata/status rows from the current or approved prior runs."""

    records: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for metadata_path in sorted((ROOT / "data" / "raw" / study).glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_hash") not in _source_hashes(config):
            continue
        status_path = metadata_path.parent / "status.json"
        if status_path.exists():
            records.append((metadata_path.parent, metadata, json.loads(status_path.read_text(encoding="utf-8"))))
    return records


def _physical_match(metadata: dict[str, Any], L: int, U: float, g: float, lambda_: float | None = None) -> bool:
    """Match one raw run to specified physical parameters without run-id assumptions."""

    physical = metadata["physical"]
    matched = int(physical["L"]) == L and np.isclose(float(physical["U"]), U) and np.isclose(float(physical["g"]), g)
    return matched and (lambda_ is None or np.isclose(float(physical["lambda"]), lambda_))


def _has_metric_crossings(config: dict[str, Any], study: str, L: int, U: float, g: float) -> bool:
    """Test whether accepted raw trials already bracket all three Mmc thresholds."""

    values: list[float] = []
    obc_ok = False
    for directory, metadata, status in _raw_records(config, study):
        if not _physical_match(metadata, L, U, g):
            continue
        if metadata["kind"] == "obc_seed" and status["status"] == "SUCCESS":
            obc_ok = True
        if metadata["kind"] != "branch_trial" or status["status"] != "SUCCESS" or not metadata.get("accepted", False):
            continue
        observables = json.loads((directory / "observables.json").read_text(encoding="utf-8"))
        value = float(observables.get("metric_violation", float("nan")))
        if np.isfinite(value):
            values.append(value)
    return obc_ok and bool(values) and min(values) <= 1.0e-4 and max(values) >= 1.0e-2


def _has_endpoint(config: dict[str, Any], study: str, L: int, U: float, g: float) -> bool:
    """Return true if a branch-valid periodic endpoint is already stored."""

    for _, metadata, status in _raw_records(config, study):
        if metadata["kind"] != "branch_trial" or status["status"] != "SUCCESS" or not metadata.get("accepted", False):
            continue
        if _physical_match(metadata, L, U, g, 1.0) and float(metadata.get("s_min", status.get("branch_overlap", 0.0))) >= 0.7:
            return True
    return False


def _fig4_checks_exist(config: dict[str, Any], L: int, U: float, g: float) -> bool:
    """Check that a reused Fig. 4 endpoint also has seed and benchmark outputs."""

    if not _has_endpoint(config, "fig4", L, U, g):
        return False
    needed = {"pbc_seed_endpoint", "pbc_seed_uniform", "pbc_seed_random", "momentum_benchmark", "bandwidth_control"}
    found: set[str] = set()
    for _, metadata, status in _raw_records(config, "fig4"):
        if status["status"] != "SUCCESS" or int(metadata["physical"]["L"]) != L or not np.isclose(float(metadata["physical"]["U"]), U):
            continue
        if metadata["kind"] == "bandwidth_control" or _physical_match(metadata, L, U, g, 1.0):
            found.add(metadata["kind"])
    return needed <= found


def _load_state(directory: Path, metadata: dict[str, Any], status: dict[str, Any], numeric: Numerics) -> HFBState:
    """Reconstruct an HFB state, including its stored occupied sheet, from raw data."""

    physical = metadata["physical"]
    chain = Chain(int(physical["L"]), float(physical["U"]), g=float(physical["g"]), lambda_=float(physical["lambda"]), filling=float(physical["filling"]), t=float(physical["t"]))
    stored = np.load(directory / "state.npz")
    mu = float(stored["mu"])
    plus, minus = stored["delta_plus"], stored["delta_minus"]
    n_up, n_down = stored["n_up"], stored["n_down"]
    matrix = bdg_matrix(chain, mu, n_up, n_down, plus, minus)
    eigensystem, _ = MeanFieldSolver._eigensystem(matrix, None)
    saved_values, saved_occupied = stored["eigenvalues"], stored["occupied"]
    if saved_values.size == eigensystem.values.size and saved_occupied.size == eigensystem.values.size:
        old, new = linear_sum_assignment(abs(saved_values[:, None] - eigensystem.values[None, :]))
        occupied = np.zeros(eigensystem.values.size, dtype=bool)
        occupied[new] = saved_occupied[old]
        eigensystem = replace(eigensystem, occupied=occupied)
    return HFBState(
        chain, mu, plus, minus, n_up, n_down, eigensystem,
        float(status["field_residual"]), float(status["density_residual"]), float(status["number_residual"]),
        0, status["status"] == "SUCCESS", str(status["message"]), float(status.get("branch_overlap", float("nan"))),
        int(status.get("mu_evaluations", 1)), int(status.get("total_scf_iterations", 0)), float(status.get("wall_seconds", 0.0)),
        int(status.get("cached_mu_warm_starts", 0)),
        float(status.get("max_density_imaginary", 0.0)),
        float(status.get("total_density_imaginary", 0.0)),
    )


def _load_endpoint(config: dict[str, Any], study: str, L: int, U: float, g: float, numeric: Numerics) -> HFBState | None:
    """Load a branch-valid lambda=1 endpoint from compatible raw data."""

    for directory, metadata, status in _raw_records(config, study):
        if metadata["kind"] == "branch_trial" and metadata.get("accepted", False) and status["status"] == "SUCCESS" and _physical_match(metadata, L, U, g, 1.0):
            return _load_state(directory, metadata, status, numeric)
    return None


def _u2_hermitian_reference(config: dict[str, Any], L: int, numeric: Numerics) -> HFBState | None:
    """Load the validated U=2 Hermitian saddle for deterministic cross-U seeding."""

    for directory, metadata, status in _raw_records(config, "fig2"):
        if metadata["kind"] == "hermitian" and status["status"] == "SUCCESS" and _physical_match(metadata, L, 2.0, 0.0, 0.0):
            return _load_state(directory, metadata, status, numeric)
    return None



def _endpoint_checks(config: dict[str, Any], study: str, endpoint: HFBState, numeric: Numerics, seeds: bool = True) -> None:
    """Store PBC seed, momentum, and bandwidth-control checks for one endpoint."""

    if seeds:
        for name, seed in _endpoint_seeds(endpoint, numeric.random_seed).items():
            state = solve_fixed_filling(MeanFieldSolver(endpoint.chain, numeric, "rescaled"), seed, endpoint.eigensystem)
            _save(config, study, state, f"pbc_seed_{name}", _endpoint_distance(state, endpoint))
    pair_uniformity = relative_error(endpoint.pair_product, np.full(endpoint.chain.L, np.mean(endpoint.pair_product)))
    density_uniformity = relative_error(endpoint.density.astype(complex), np.full(endpoint.chain.L, np.mean(endpoint.density)))
    if pair_uniformity < 1.0e-8 and density_uniformity < 1.0e-8:
        _, blocks = pbc_blocks(endpoint.chain, endpoint.mu, float(np.mean(endpoint.n_up)), complex(np.mean(endpoint.delta_plus)), complex(np.mean(endpoint.delta_minus)))
        spectrum = np.concatenate([np.linalg.eigvals(block) for block in blocks])
        distance = spectrum_distance(endpoint.eigensystem.values, spectrum)
    else:
        distance = float("nan")
    _save(config, study, endpoint, "momentum_benchmark", {"pair_uniformity": pair_uniformity, "density_uniformity": density_uniformity, "momentum_spectrum_error": distance})
    control_chain = Chain(endpoint.chain.L, endpoint.chain.U, lambda_=1.0, filling=endpoint.chain.filling, t=endpoint.chain.t * np.cosh(endpoint.chain.g))
    control = _fixed_state(control_chain, numeric, "hermitian")
    _save(config, study, control, "bandwidth_control", {"pair_product_error_to_nh": relative_error(control.pair_product, endpoint.pair_product)})


def _endpoint_seeds(endpoint: HFBState, seed: int) -> dict[str, HFBState]:
    uniform_plus = np.full(endpoint.chain.L, np.mean(endpoint.delta_plus), dtype=complex)
    uniform_minus = np.full(endpoint.chain.L, np.mean(endpoint.delta_minus), dtype=complex)
    uniform = replace(endpoint, delta_plus=uniform_plus, delta_minus=uniform_minus, n_up=np.full(endpoint.chain.L, np.mean(endpoint.n_up)), n_down=np.full(endpoint.chain.L, np.mean(endpoint.n_down)))
    rng = np.random.default_rng(seed)
    random = replace(endpoint, delta_plus=uniform_plus * (1.0 + 0.05 * (rng.normal(size=endpoint.chain.L) + 1j * rng.normal(size=endpoint.chain.L))), delta_minus=uniform_minus * (1.0 + 0.05 * (rng.normal(size=endpoint.chain.L) + 1j * rng.normal(size=endpoint.chain.L))))
    return {"endpoint": endpoint, "uniform": uniform, "random": random}


def _endpoint_distance(state: HFBState, endpoint: HFBState) -> dict[str, float]:
    return {"pair_product_error_to_endpoint": relative_error(state.pair_product, endpoint.pair_product), "density_error_to_endpoint": relative_error(state.density.astype(complex), endpoint.density.astype(complex)), "spectrum_error_to_endpoint": _spectrum_error(state, endpoint)}


def _fixed_state(chain: Chain, numeric: Numerics, representation: str) -> HFBState:
    return solve_fixed_filling(MeanFieldSolver(chain, numeric, representation))


def _map_obc(reference: HFBState, g: float) -> HFBState:
    chain = replace(reference.chain, g=g, lambda_=0.0)
    plus, minus = map_fields_from_hermitian(reference.delta_plus, g)
    matrix = bdg_matrix(chain, reference.mu, reference.n_up, reference.n_down, plus, minus)
    eigensystem, _ = MeanFieldSolver._eigensystem(matrix, None)
    return HFBState(chain, reference.mu, plus, minus, reference.n_up.copy(), reference.n_down.copy(), eigensystem, 0.0, 0.0, reference.number_residual, 0, True, "SUCCESS")


def _comparable(first: HFBState, second: HFBState) -> bool:
    """Return whether two converged states carry equal-size eigensystems."""

    return first.converged and second.converged and first.eigensystem.values.size == second.eigensystem.values.size and first.eigensystem.values.size > 0


def _spectrum_error(first: HFBState, second: HFBState) -> float:
    """Return an assigned spectrum distance, or NaN for an invalid comparison."""

    return spectrum_distance(first.eigensystem.values, second.eigensystem.values) if _comparable(first, second) else float("nan")


def _fig2_comparison(mapped: HFBState, state: HFBState) -> dict[str, float]:
    """Validation quantities for a direct OBC solve, guarded after a failed run."""

    if not _comparable(mapped, state):
        return {"component_error": float("nan"), "scale_real": float("nan"), "scale_imag": float("nan"), "pair_product_error": float("nan"), "density_error": float("nan"), "spectrum_error": float("nan")}
    scale, component_error = align_nambu_scale(mapped, state)
    return {
        "component_error": component_error,
        "scale_real": float(np.real(scale)),
        "scale_imag": float(np.imag(scale)),
        "pair_product_error": relative_error(state.pair_product, mapped.pair_product),
        "density_error": relative_error(state.density.astype(complex), mapped.density.astype(complex)),
        "spectrum_error": _spectrum_error(state, mapped),
    }


def _numerics(config: dict[str, Any]) -> Numerics:
    """Construct the one shared numerical-parameter object from YAML."""

    return Numerics(**config["numerics"])


def _save(config: dict[str, Any], study: str, state: HFBState, kind: str, extra: dict[str, Any] | None = None) -> Path:
    extra = extra or {}
    config_hash = _hash(config)
    identity = {"study": study, "kind": kind, "L": state.chain.L, "U": state.chain.U, "g": state.chain.g, "lambda": state.chain.lambda_, "filling": state.chain.filling, "extra": extra}
    run_id = f"{study}_{kind}_{hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
    directory = ROOT / "data" / "raw" / study / run_id
    directory.mkdir(parents=True, exist_ok=True)
    summary = _summary(state)
    metadata = {"run_id": run_id, "study": study, "kind": kind, "config_hash": config_hash, **_git_provenance(), "physical": {"L": state.chain.L, "U": state.chain.U, "g": state.chain.g, "lambda": state.chain.lambda_, "t": state.chain.t, "filling": state.chain.filling}, **extra}
    status = {"status": "SUCCESS" if state.converged else "FAILED", "message": state.message, "field_residual": state.field_residual, "density_residual": state.density_residual, "number_residual": state.number_residual, "branch_overlap": state.branch_overlap, "mu_evaluations": state.mu_evaluations, "total_scf_iterations": state.total_scf_iterations, "wall_seconds": state.wall_seconds, "cached_mu_warm_starts": state.cached_mu_warm_starts, "max_density_imaginary": state.max_density_imaginary, "total_density_imaginary": state.total_density_imaginary}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    (directory / "observables.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(directory / "state.npz", delta_plus=state.delta_plus, delta_minus=state.delta_minus, n_up=state.n_up, n_down=state.n_down, mu=state.mu, eigenvalues=state.eigensystem.values, occupied=state.eigensystem.occupied)
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status))
        writer.writeheader(); writer.writerow(status)
    print(f"  [{study}] L={state.chain.L} U={state.chain.U:g} g={state.chain.g:g} lambda={state.chain.lambda_:.3e} {kind}: {status['status']} μ={state.mu:.8g} Nμ={state.mu_evaluations} SCF={state.total_scf_iterations} time={state.wall_seconds:.1f}s", flush=True)
    return directory


def _save_branch(config: dict[str, Any], study: str, branch: Branch, seed_method: str) -> None:
    """Persist a branch and the automatic seed route that produced it."""

    reference = branch.accepted[0]
    _save(config, study, reference, "obc_seed", {"accepted": True, "requested": True, "seed_method": seed_method, "branch_terminated": branch.terminated, "branch_message": branch.message})
    for index, trial in enumerate(branch.trials):
        extra = {"accepted": trial.accepted, "requested": trial.requested, "reason": trial.reason, "s_min": trial.s_min, "trial_index": index, "chi": trial.state.chain.lambda_ * np.exp(trial.state.chain.g * trial.state.chain.L), "seed_method": seed_method, **_state_diagnostics(trial.state, reference)}
        _save(config, study, trial.state, "branch_trial", extra)


def _save_reverse_branch(config: dict[str, Any], branch: Branch, obc_reference: HFBState) -> None:
    """Persist a PBC-to-OBC audit and compare its endpoint with the OBC seed."""

    for index, trial in enumerate(branch.trials):
        _save(
            config,
            "branch_audit",
            trial.state,
            "reverse_branch_trial",
            {
                "accepted": trial.accepted,
                "requested": trial.requested,
                "reason": trial.reason,
                "s_min": trial.s_min,
                "trial_index": index,
                "branch_terminated": branch.terminated,
                "branch_message": branch.message,
            },
        )
    endpoint = branch.accepted[-1]
    distance = _endpoint_distance(endpoint, obc_reference)
    if endpoint.eigensystem.values.size and obc_reference.eigensystem.values.size:
        distance["projector_error_to_endpoint"] = relative_error(
            occupied_projector(endpoint.eigensystem),
            occupied_projector(obc_reference.eigensystem),
        )
    _save(
        config,
        "branch_audit",
        endpoint,
        "reverse_obc_endpoint",
        {
            **distance,
            "branch_terminated": branch.terminated,
            "branch_message": branch.message,
        },
    )


def _summary(state: HFBState) -> dict[str, Any]:
    pair = state.pair_product
    bulk = slice(state.chain.L // 4, 3 * state.chain.L // 4)
    metric, coefficient, stable = metric_violation(state) if state.chain.U > 0.0 else (float("nan"), complex("nan"), False)
    global_metric, global_coefficient = global_conjugacy_violation(state) if state.chain.U > 0.0 else (float("nan"), float("nan"))
    return {
        "mu": state.mu,
        "bulk_pair_product_real": float(np.real(np.mean(pair[bulk]))),
        "bulk_pair_product_imag": float(np.imag(np.mean(pair[bulk]))),
        "max_pair_product_imaginary": float(np.max(np.abs(np.imag(pair)))),
        "metric_violation": metric, "metric_K_real": float(np.real(coefficient)), "metric_phase_stable": stable,
        "metric_epsilon": METRIC_EPSILON,
        "global_conjugacy_violation": global_metric, "global_conjugacy_K": global_coefficient,
        "gamma_max_over_t": gamma_max(state) if state.eigensystem.values.size else float("nan"),
        "complex_fraction": complex_fraction(state) if state.eigensystem.values.size else float("nan"),
        "minimum_eigenvalue_separation": min_spectrum_separation(state),
        "right_condition": float(np.linalg.cond(state.eigensystem.right)) if state.eigensystem.right.size else float("inf"),
        **projector_diagnostics(state.eigensystem),
    }


def _state_diagnostics(state: HFBState, reference: HFBState) -> dict[str, float]:
    if state.chain.U == 0.0:
        return {"delta_P_full": float("nan"), "delta_P_bulk": float("nan")}
    return {
        "delta_P_full": pair_deformation(state, reference, False),
        "delta_P_bulk": pair_deformation(state, reference, True),
        "delta_P_bulk_one_third": pair_deformation_window(state, reference, 1.0 / 3.0),
        "delta_P_bulk_two_thirds": pair_deformation_window(state, reference, 2.0 / 3.0),
    }


def _hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _git_provenance() -> dict[str, Any]:
    """Record the exact source revision, excluding generated output changes."""

    try:
        source_pathspec = [
            ".",
            ":(exclude)data/raw/**",
            ":(exclude)data/processed/**",
            ":(exclude)figures/**",
            ":(exclude)logs/**",
            ":(exclude)artifacts/**",
        ]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *source_pathspec],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *source_pathspec],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "UNKNOWN", "git_dirty": True, "git_diff_sha256": "UNKNOWN"}


def _mp_similarity(matrix: np.ndarray, expected: np.ndarray, V: np.ndarray, digits: int) -> float:
    mpmath.mp.dps = digits
    dimension = matrix.shape[0]
    raw = mpmath.matrix([[mpmath.mpc(matrix[i, j]) for j in range(dimension)] for i in range(dimension)])
    transform = mpmath.matrix([[mpmath.mpc(V[i, j]) for j in range(dimension)] for i in range(dimension)])
    target = mpmath.matrix([[mpmath.mpc(expected[i, j]) for j in range(dimension)] for i in range(dimension)])
    value = transform * raw * (transform ** -1)
    numerator = mpmath.sqrt(sum(abs(value[i, j] - target[i, j]) ** 2 for i in range(dimension) for j in range(dimension)))
    denominator = mpmath.sqrt(sum(abs(target[i, j]) ** 2 for i in range(dimension) for j in range(dimension)))
    return float(numerator / denominator)
