"""Safeguarded chemical-potential bisection built on the inner HFB solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .solver import Eigensystem, HFBState, MeanFieldSolver


@dataclass(frozen=True)
class FillingCurveAudit:
    """Local fixed-sheet filling scan around one converged solution."""

    states: tuple[HFBState, ...]
    monotone: bool
    target_crossings: int
    minimum_branch_overlap: float
    maximum_density_imaginary: float


def audit_filling_curve(
    solver: MeanFieldSolver,
    reference: HFBState,
    half_width: float = 0.05,
    points: int = 5,
) -> FillingCurveAudit:
    """Sample ``N(mu)`` on the reference occupied sheet near a fixed point."""

    if points < 3 or points % 2 == 0:
        raise ValueError("filling audit requires an odd number of at least three points")
    if half_width <= 0.0:
        raise ValueError("filling audit half width must be positive")
    mus = np.linspace(reference.mu - half_width, reference.mu + half_width, points)
    states = tuple(
        solver.solve_at_mu(float(mu), initial_state=reference, occupied_reference=reference.eigensystem)
        for mu in mus
    )
    if not all(state.converged for state in states):
        return FillingCurveAudit(
            states,
            False,
            0,
            min((state.branch_overlap for state in states), default=float("nan")),
            max((state.max_density_imaginary for state in states), default=float("nan")),
        )
    fillings = np.array([np.mean(state.density) for state in states], dtype=float)
    residuals = fillings - solver.chain.filling
    monotone = bool(np.all(np.diff(fillings) >= -solver.numerics.number_tolerance))
    tolerance = solver.numerics.number_tolerance
    nonzero_signs = np.sign(residuals[np.abs(residuals) > tolerance])
    crossings = int(np.count_nonzero(nonzero_signs[:-1] != nonzero_signs[1:]))
    if not crossings and np.any(np.abs(residuals) <= tolerance):
        crossings = 1
    return FillingCurveAudit(
        states,
        monotone,
        crossings,
        float(min(state.branch_overlap for state in states)),
        float(max(state.max_density_imaginary for state in states)),
    )


def solve_fixed_filling(
    solver: MeanFieldSolver,
    initial_state: HFBState | None = None,
    occupied_reference: Eigensystem | None = None,
) -> HFBState:
    """Find the target total filling with fully converged inner HFB solves.

    The function is deliberately a conventional safeguarded bisection.  A
    number residual is never used to choose a bracket side unless the inner
    HFB calculation at that chemical potential has converged.  ``initial`` is
    only the natural continuation warm start.  Every fully converged trial is
    retained in a small nearest-``mu`` cache for later bisection evaluations;
    this changes only the initial fields, never the safeguarded bisection
    decision or convergence gates.  There is no prediction, secant, Newton,
    or DIIS path.
    """

    started, calls, total_iterations, cached_warm_starts = perf_counter(), 0, 0, 0
    chain, numeric = solver.chain, solver.numerics
    cache: list[HFBState] = []

    def finish(state: HFBState, converged: bool, message: str) -> HFBState:
        return replace(
            state,
            converged=converged,
            message=message,
            number_residual=abs(float(np.mean(state.density)) - chain.filling),
            mu_evaluations=calls,
            total_scf_iterations=total_iterations,
            wall_seconds=perf_counter() - started,
            cached_mu_warm_starts=cached_warm_starts,
        )

    def evaluate(mu: float, seed: HFBState | None) -> HFBState:
        nonlocal calls, total_iterations, cached_warm_starts
        if cache:
            seed = min(cache, key=lambda state: abs(state.mu - mu))
            cached_warm_starts += 1
        state = solver.solve_at_mu(mu, initial_state=seed, occupied_reference=occupied_reference)
        calls += 1
        total_iterations += state.iterations
        if state.converged:
            cache.append(state)
        return state

    kf = 0.5 * np.pi * chain.filling
    estimate = -2.0 * chain.t * np.cosh(chain.g) * np.cos(kf) - 0.5 * chain.U * chain.filling
    center_mu = initial_state.mu if initial_state is not None else estimate
    center = evaluate(float(np.clip(center_mu, *numeric.mu_bounds)), initial_state)
    if not center.converged:
        return finish(center, False, center.message)
    center_value = float(np.mean(center.density) - chain.filling)
    if abs(center_value) < numeric.number_tolerance:
        return finish(center, True, "SUCCESS")

    half_width = (
        max(1.0e-5 * chain.t, 8.0 * chain.t * abs(center_value))
        if initial_state is not None
        else 4.0 * chain.t * np.cosh(chain.g)
    )
    lower_bound, upper_bound = numeric.mu_bounds
    if center_value < 0.0:
        low, low_value = center, center_value
        high_mu = min(upper_bound, center.mu + half_width)
        high = evaluate(high_mu, center)
        if not high.converged:
            return finish(high, False, high.message)
        high_value = float(np.mean(high.density) - chain.filling)
    else:
        high, high_value = center, center_value
        low_mu = max(lower_bound, center.mu - half_width)
        low = evaluate(low_mu, center)
        if not low.converged:
            return finish(low, False, low.message)
        low_value = float(np.mean(low.density) - chain.filling)

    while low_value > 0.0 or high_value < 0.0:
        half_width *= 2.0
        if low_value > 0.0:
            next_mu = max(lower_bound, center.mu - half_width)
            if np.isclose(next_mu, low.mu):
                return finish(center, False, "NUMBER_SOLVE_FAILED: lower μ bound did not bracket filling")
            low = evaluate(next_mu, low)
            if not low.converged:
                return finish(low, False, low.message)
            low_value = float(np.mean(low.density) - chain.filling)
        if high_value < 0.0:
            next_mu = min(upper_bound, center.mu + half_width)
            if np.isclose(next_mu, high.mu):
                return finish(center, False, "NUMBER_SOLVE_FAILED: upper μ bound did not bracket filling")
            high = evaluate(next_mu, high)
            if not high.converged:
                return finish(high, False, high.message)
            high_value = float(np.mean(high.density) - chain.filling)

    best = center
    for _ in range(numeric.max_mu_iterations):
        midpoint = 0.5 * (low.mu + high.mu)
        seed = low if abs(midpoint - low.mu) <= abs(high.mu - midpoint) else high
        mid = evaluate(midpoint, seed)
        if not mid.converged:
            return finish(mid, False, mid.message)
        value = float(np.mean(mid.density) - chain.filling)
        best = mid
        if abs(value) < numeric.number_tolerance:
            result = finish(mid, True, "SUCCESS")
            return replace(result, mu_evaluations=calls, total_scf_iterations=total_iterations, wall_seconds=perf_counter() - started)
        if value < 0.0:
            low, low_value = mid, value
        else:
            high, high_value = mid, value
    return finish(best, False, "NUMBER_SOLVE_FAILED: maximum bisection iterations")
