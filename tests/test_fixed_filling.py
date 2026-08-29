from __future__ import annotations

from nhbdg.fixed_filling import audit_filling_curve, solve_fixed_filling
from nhbdg.model import Chain, Numerics
from nhbdg.solver import MeanFieldSolver


def test_safeguarded_bisection_reaches_target_filling() -> None:
    numeric = Numerics(field_tolerance=1.0e-8, density_tolerance=1.0e-8, number_tolerance=1.0e-7, max_scf_iterations=2000, max_mu_iterations=32)
    state = solve_fixed_filling(MeanFieldSolver(Chain(4, 2.0, g=0.08), numeric, "rescaled"))
    assert state.converged
    assert state.number_residual < numeric.number_tolerance
    assert state.mu_evaluations > 1


def test_filling_curve_audit_is_monotone_with_one_target_crossing() -> None:
    numeric = Numerics(field_tolerance=1.0e-8, density_tolerance=1.0e-8, number_tolerance=1.0e-7, max_scf_iterations=2000, max_mu_iterations=32)
    solver = MeanFieldSolver(Chain(4, 2.0, g=0.05, lambda_=0.2), numeric, "rescaled")
    state = solve_fixed_filling(solver)
    audit = audit_filling_curve(solver, state, half_width=0.02, points=3)
    assert all(point.converged for point in audit.states)
    assert audit.monotone
    assert audit.target_crossings == 1
    assert audit.maximum_density_imaginary < numeric.density_imaginary_tolerance
