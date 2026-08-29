"""Gauge-invariant diagnostics, spectra, and direct Green functions."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from .model import Chain, ComplexArray, coordinates, nambu_similarity
from .solver import Eigensystem, HFBState, occupied_projector

METRIC_EPSILON = 1.0e-30


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """L2 relative difference with a safe zero-norm denominator."""

    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-30))


def align_nambu_scale(reference: HFBState, candidate: HFBState) -> tuple[complex, float]:
    """Align the global two-field normalization before component comparison."""

    denominator = np.vdot(reference.delta_plus, reference.delta_plus)
    if abs(denominator) < METRIC_EPSILON:
        return complex("nan"), float("nan")
    scale = np.vdot(reference.delta_plus, candidate.delta_plus) / denominator
    if abs(scale) < 1.0e-30:
        return scale, float("inf")
    aligned = candidate.delta_plus / scale
    return scale, relative_error(aligned, reference.delta_plus)


def metric_violation(state: HFBState) -> tuple[float, complex, bool]:
    """Return the manuscript-defined OBC metric-conjugacy mismatch."""

    x = coordinates(state.chain.L)
    A = np.exp(-4.0 * state.chain.g * x) * np.conjugate(state.delta_plus)
    denominator = np.vdot(A, A)
    if abs(denominator) < 1.0e-30:
        return float("nan"), complex("nan"), False
    unconstrained = np.vdot(A, state.delta_minus) / denominator
    coefficient = max(float(np.real(unconstrained)), 0.0)
    fitted = coefficient * A
    residual = float(
        np.linalg.norm(state.delta_minus - fitted)
        / (np.linalg.norm(state.delta_minus) + np.linalg.norm(fitted) + METRIC_EPSILON)
    )
    phase_stable = coefficient > 0.0 and abs(np.imag(unconstrained)) <= 1.0e-8 * max(coefficient, METRIC_EPSILON)
    return residual, complex(coefficient), bool(phase_stable)


def global_conjugacy_violation(state: HFBState) -> tuple[float, float]:
    """Fit ``Delta_minus = K conj(Delta_plus)`` with real ``K >= 0``."""

    reference = np.conjugate(state.delta_plus)
    denominator = np.vdot(reference, reference)
    if abs(denominator) < METRIC_EPSILON:
        return float("nan"), float("nan")
    coefficient = max(float(np.real(np.vdot(reference, state.delta_minus) / denominator)), 0.0)
    fitted = coefficient * reference
    residual = np.linalg.norm(state.delta_minus - fitted) / (
        np.linalg.norm(state.delta_minus) + np.linalg.norm(fitted) + METRIC_EPSILON
    )
    return float(residual), coefficient


def projector_diagnostics(eigensystem: Eigensystem) -> dict[str, float]:
    """Return algebraic and spectral diagnostics for one occupied sheet."""

    if eigensystem.values.size == 0:
        return {
            "projector_idempotency": float("nan"),
            "projector_trace_error": float("nan"),
            "biorthogonality_error": float("nan"),
            "occupied_unoccupied_separation": float("nan"),
            "real_line_gap": float("nan"),
            "projector_norm": float("nan"),
            "real_part_projector_difference": float("nan"),
            "real_part_occupied_rank": float("nan"),
        }
    projector = occupied_projector(eigensystem)
    rank = int(np.count_nonzero(eigensystem.occupied))
    occupied = eigensystem.values[eigensystem.occupied]
    unoccupied = eigensystem.values[~eigensystem.occupied]
    separation = (
        float(np.min(np.abs(occupied[:, None] - unoccupied[None, :])))
        if occupied.size and unoccupied.size
        else float("nan")
    )
    normalization = eigensystem.left.conj().T @ eigensystem.right
    real_occupied = np.real(eigensystem.values) < 0.0
    if np.count_nonzero(real_occupied) == rank:
        real_projector = (
            eigensystem.right[:, real_occupied]
            @ eigensystem.left[:, real_occupied].conj().T
        )
        real_projector_difference = relative_error(real_projector, projector)
    else:
        real_projector_difference = float("inf")
    return {
        "projector_idempotency": relative_error(projector @ projector, projector),
        "projector_trace_error": float(abs(np.trace(projector) - rank)),
        "biorthogonality_error": relative_error(normalization, np.eye(normalization.shape[0])),
        "occupied_unoccupied_separation": separation,
        "real_line_gap": float(np.min(np.abs(np.real(eigensystem.values)))),
        "projector_norm": float(np.linalg.norm(projector, ord=2)),
        "real_part_projector_difference": real_projector_difference,
        "real_part_occupied_rank": float(np.count_nonzero(real_occupied)),
    }


def pair_deformation(state: HFBState, reference: HFBState, bulk: bool) -> float:
    """Relative deformation of the complex pair product, optionally in central half."""

    current, initial = state.pair_product, reference.pair_product
    if bulk:
        selection = slice(state.chain.L // 4, 3 * state.chain.L // 4)
        current, initial = current[selection], initial[selection]
    return relative_error(current, initial)


def pair_deformation_window(state: HFBState, reference: HFBState, fraction: float) -> float:
    """Return pair-product deformation in a centered fractional window."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("bulk fraction must lie in (0, 1]")
    width = max(1, round(state.chain.L * fraction))
    start = (state.chain.L - width) // 2
    stop = start + width
    return relative_error(state.pair_product[start:stop], reference.pair_product[start:stop])


def occupied_overlap(old: Eigensystem, new: Eigensystem) -> float:
    """Smallest principal singular value between occupied right subspaces."""

    if old.right.size == 0 or new.right.size == 0:
        return 0.0
    old_q, _ = np.linalg.qr(old.right[:, old.occupied])
    new_q, _ = np.linalg.qr(new.right[:, new.occupied])
    singular = np.linalg.svd(old_q.conj().T @ new_q, compute_uv=False)
    return float(np.min(singular)) if singular.size else 0.0


def spectrum_distance(first: ComplexArray, second: ComplexArray) -> float:
    """Hungarian-matched maximum distance between equal-size complex spectra."""

    if first.size != second.size:
        raise ValueError("Spectrum sizes differ.")
    rows, columns = linear_sum_assignment(np.abs(first[:, None] - second[None, :]))
    return float(np.max(np.abs(first[rows] - second[columns])))


def gamma_max(state: HFBState) -> float:
    """Dimensionless ``max |Im E| / t``."""

    return float(np.max(np.abs(np.imag(state.eigensystem.values))) / state.chain.t)


def complex_fraction(state: HFBState) -> float:
    """Fraction of eigenvalues with ``|Im E| > 1e-8``."""

    return float(np.mean(np.abs(np.imag(state.eigensystem.values)) > 1.0e-8))


def min_spectrum_separation(state: HFBState) -> float:
    """Minimum pairwise separation; this is not an exceptional-point claim."""

    values = state.eigensystem.values
    if values.size < 2:
        return float("nan")
    distances = np.abs(values[:, None] - values[None, :])
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def green(matrix: ComplexArray, omega: float, eta: float) -> ComplexArray:
    """Directly solve ``(zI-H)G=I``; no eigenvector expansion is used."""

    z = omega + 1j * eta
    return scipy.linalg.solve(z * np.eye(matrix.shape[0], dtype=complex) - matrix, np.eye(matrix.shape[0]), check_finite=False)


def green_covariance_error(nonhermitian: ComplexArray, hermitian: ComplexArray, chain: Chain) -> float:
    """Check the exact OBC relation ``G_g=V^-1 G_0 V``."""

    V = nambu_similarity(chain)
    inverse = np.diag(1.0 / np.diag(V))
    return relative_error(nonhermitian, inverse @ hermitian @ V)


def particle_block(resolvent: ComplexArray) -> ComplexArray:
    """Particle-particle block of a Nambu resolvent."""

    L = resolvent.shape[0] // 2
    return resolvent[:L, :L]


def stripped_propagator(value: complex, i: int, j: int, chain: Chain) -> complex:
    """Remove the known OBC factor ``exp[g(x_i-x_j)]`` from one propagator."""

    x = coordinates(chain.L)
    return value * np.exp(-chain.g * (x[i] - x[j]))
