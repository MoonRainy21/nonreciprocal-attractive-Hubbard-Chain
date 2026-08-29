"""OBC-connected weak-link continuation on one occupied BdG sheet."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .fixed_filling import solve_fixed_filling
from .model import Numerics
from .observables import occupied_overlap, projector_diagnostics
from .solver import HFBState, MeanFieldSolver


@dataclass
class BranchTrial:
    """One accepted or rejected attempt retained for branch diagnostics."""

    state: HFBState
    requested: bool
    accepted: bool
    s_min: float
    reason: str


@dataclass
class Branch:
    """Accepted OBC-connected states and all attempted continuation points."""

    accepted: list[HFBState]
    trials: list[BranchTrial]
    terminated: bool
    message: str


def continue_branch(
    obc_state: HFBState,
    lambda_targets: list[float],
    numerics: Numerics,
    max_consecutive_scf_failures: int | None = None,
) -> Branch:
    """Track an occupied branch in either direction with midpoint subdivision.

    A candidate is accepted only after its fixed-filling HFB solve converges,
    field and number residuals pass the published gate, and the occupied
    right-subspace overlap satisfies ``s_min >= 0.7``.  Failed targets are not
    extrapolated; a midpoint is attempted until the configured minimum step.
    """

    if not obc_state.converged:
        return Branch([obc_state], [], True, "OBC seed did not converge")
    accepted, trials, current = [obc_state], [], obc_state
    goals = list(dict.fromkeys(float(value) for value in lambda_targets))
    if not goals:
        return Branch(accepted, trials, False, "SUCCESS")
    direction = np.sign(goals[0] - current.chain.lambda_)
    if direction == 0.0:
        goals = goals[1:]
        if not goals:
            return Branch(accepted, trials, False, "SUCCESS")
        direction = np.sign(goals[0] - current.chain.lambda_)
    if direction == 0.0 or any(
        direction * (later - earlier) <= 0.0 for earlier, later in pairwise(goals)
    ):
        raise ValueError("lambda_targets must be strictly monotone from the initial state")
    if any(not 0.0 <= goal <= 1.0 for goal in goals):
        raise ValueError("lambda targets must lie in [0, 1]")

    for goal in goals:
        if direction * (goal - current.chain.lambda_) <= 0.0:
            continue
        target, requested, goal_scf_failures = goal, True, 0
        while True:
            candidate_chain = current.chain.with_link(target)
            solver = MeanFieldSolver(candidate_chain, numerics, representation="rescaled")
            candidate = solve_fixed_filling(solver, initial_state=current, occupied_reference=current.eigensystem)
            s_min = occupied_overlap(current.eigensystem, candidate.eigensystem)
            reason = _reason(candidate, s_min, numerics)
            candidate.branch_overlap = s_min
            trial = BranchTrial(candidate, requested, reason == "SUCCESS", s_min, reason)
            trials.append(trial)
            print(
                f"    [branch] lambda={target:.3e} {'accepted' if trial.accepted else 'rejected'} "
                f"({reason}); Nmu={candidate.mu_evaluations} SCF={candidate.total_scf_iterations} "
                f"cache={candidate.cached_mu_warm_starts}",
                flush=True,
            )
            if trial.accepted:
                accepted.append(candidate)
                current = candidate
                # ``np.isclose`` defaults to an absolute tolerance of 1e-8,
                # which incorrectly identifies distinct logarithmic weak
                # links such as 1e-12 and 1e-10 as the same target.
                if np.isclose(target, goal, rtol=1.0e-12, atol=0.0):
                    break
                target, requested = goal, True
                continue
            if reason == "NO_SCF_CONVERGENCE" and np.isclose(target, goal, rtol=1.0e-12, atol=0.0):
                goal_scf_failures += 1
            if max_consecutive_scf_failures is not None and goal_scf_failures >= max_consecutive_scf_failures:
                return Branch(
                    accepted,
                    trials,
                    True,
                    f"REFINEMENT_STOP: {goal_scf_failures} NO_SCF_CONVERGENCE trials at lambda={goal:.12g}",
                )
            midpoint = _midpoint(current.chain.lambda_, target)
            if (
                direction * (midpoint - current.chain.lambda_) <= 0.0
                or abs(target - midpoint) < numerics.minimum_lambda_step
            ):
                return Branch(accepted, trials, True, f"MIN_STEP_REACHED: {reason}")
            target, requested = midpoint, False
    return Branch(accepted, trials, False, "SUCCESS")


def _midpoint(current: float, target: float) -> float:
    """Use a geometric midpoint away from zero and an arithmetic zero step."""

    if current == 0.0 or target == 0.0:
        return 0.5 * (current + target)
    return float(np.sqrt(current * target))


def _reason(state: HFBState, s_min: float, numerics: Numerics) -> str:
    if not state.converged:
        return state.message
    if state.field_residual >= 1.0e-8:
        return "NO_SCF_CONVERGENCE: field residual exceeds branch gate"
    if state.number_residual >= 1.0e-8:
        return "NUMBER_SOLVE_FAILED: number residual exceeds branch gate"
    if s_min < numerics.minimum_branch_overlap:
        return "BRANCH_OVERLAP_FAILED"
    diagnostics = projector_diagnostics(state.eigensystem)
    if diagnostics["projector_idempotency"] >= numerics.projector_idempotency_tolerance:
        return "PROJECTOR_FAILED: idempotency residual exceeds branch gate"
    if diagnostics["biorthogonality_error"] >= numerics.biorthogonality_tolerance:
        return "PROJECTOR_FAILED: biorthogonality residual exceeds branch gate"
    if diagnostics["occupied_unoccupied_separation"] <= numerics.minimum_occupied_unoccupied_separation:
        return "SPECTRAL_CLUSTER_FAILED: occupied and unoccupied clusters are unresolved"
    return "SUCCESS"
