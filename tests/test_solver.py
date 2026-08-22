from __future__ import annotations

import numpy as np

from nhbdg.model import Chain, Numerics
from nhbdg.solver import MeanFieldSolver, occupied_projector
from tests.reference_fock import quadratic_fock_observables


def test_projector_convention_matches_independent_fock_reference() -> None:
    h_up, h_down = np.array([[-0.41]], complex), np.array([[-0.23]], complex)
    plus = np.array([0.37 + 0.11j])
    matrix = np.block([[h_up, np.diag(plus)], [np.diag(plus.conjugate()), -h_down.T]])
    eigensystem, _ = MeanFieldSolver._eigensystem(matrix, None)
    actual = MeanFieldSolver._observables(occupied_projector(eigensystem))
    expected = quadratic_fock_observables(h_up, h_down, plus, plus.conjugate())
    for got, want in zip(actual, expected):
        assert np.max(np.abs(got - want)) < 1.0e-10


def test_fixed_mu_solver_converges_with_linear_mixing() -> None:
    solver = MeanFieldSolver(Chain(4, 2.0), Numerics(field_tolerance=1.0e-8, density_tolerance=1.0e-8, max_scf_iterations=2000))
    state = solver.solve_at_mu(-1.0)
    assert state.converged
    assert state.field_residual < 1.0e-8
