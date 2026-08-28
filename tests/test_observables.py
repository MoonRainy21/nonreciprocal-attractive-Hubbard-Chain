from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nhbdg.model import Chain, bdg_matrix, map_fields_from_hermitian
from nhbdg.observables import (
    green,
    green_covariance_error,
    metric_violation,
    projector_diagnostics,
)
from nhbdg.solver import MeanFieldSolver


def test_direct_green_function_obeys_obc_similarity() -> None:
    reference = Chain(4, 2.0)
    mapped = Chain(4, 2.0, g=0.2)
    density = np.full(4, 0.4)
    gap = np.full(4, 0.3, complex)
    plus, minus = map_fields_from_hermitian(gap, mapped.g)
    H0 = bdg_matrix(reference, -0.5, density, density, gap, gap)
    Hg = bdg_matrix(mapped, -0.5, density, density, plus, minus)
    assert green_covariance_error(green(Hg, 0.3, 0.02), green(H0, 0.3, 0.02), mapped) < 1.0e-10


def test_metric_violation_uses_symmetric_manuscript_normalization() -> None:
    state = SimpleNamespace(
        chain=SimpleNamespace(L=2, g=0.0),
        delta_plus=np.array([1.0, 1.0], complex),
        delta_minus=np.array([1.0, 2.0], complex),
    )
    residual, coefficient, stable = metric_violation(state)
    fitted = coefficient.real * np.conjugate(state.delta_plus)
    expected = np.linalg.norm(state.delta_minus - fitted) / (
        np.linalg.norm(state.delta_minus) + np.linalg.norm(fitted) + 1.0e-30
    )
    assert np.isclose(residual, expected)
    assert coefficient.real >= 0.0
    assert stable


def test_projector_diagnostics_are_small_for_hermitian_problem() -> None:
    matrix = np.diag([-2.0, -1.0, 1.0, 3.0]).astype(complex)
    eigensystem, _ = MeanFieldSolver._eigensystem(matrix, None)
    diagnostics = projector_diagnostics(eigensystem)
    assert diagnostics["projector_idempotency"] < 1.0e-14
    assert diagnostics["projector_trace_error"] < 1.0e-14
    assert diagnostics["biorthogonality_error"] < 1.0e-14
    assert np.isclose(diagnostics["occupied_unoccupied_separation"], 2.0)
    assert np.isclose(diagnostics["real_line_gap"], 1.0)
