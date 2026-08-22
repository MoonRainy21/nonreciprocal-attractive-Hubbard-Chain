from __future__ import annotations

from nhbdg.fixed_filling import solve_fixed_filling
from nhbdg.model import Chain, Numerics
from nhbdg.solver import MeanFieldSolver


def test_safeguarded_bisection_reaches_target_filling() -> None:
    numeric = Numerics(field_tolerance=1.0e-8, density_tolerance=1.0e-8, number_tolerance=1.0e-7, max_scf_iterations=2000, max_mu_iterations=32)
    state = solve_fixed_filling(MeanFieldSolver(Chain(4, 2.0, g=0.08), numeric, "rescaled"))
    assert state.converged
    assert state.number_residual < numeric.number_tolerance
    assert state.mu_evaluations > 1
