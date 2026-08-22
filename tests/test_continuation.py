from __future__ import annotations

from nhbdg.continuation import continue_branch
from nhbdg.fixed_filling import solve_fixed_filling
from nhbdg.model import Chain, Numerics
from nhbdg.solver import MeanFieldSolver


def test_continuation_retains_accepted_trials() -> None:
    numeric = Numerics(field_tolerance=1.0e-8, density_tolerance=1.0e-8, number_tolerance=1.0e-7, max_scf_iterations=2000, max_mu_iterations=32)
    obc = solve_fixed_filling(MeanFieldSolver(Chain(4, 2.0, g=0.08), numeric, "rescaled"))
    branch = continue_branch(obc, [1.0e-4, 1.0e-3], numeric)
    assert branch.trials
    assert branch.accepted[0].chain.lambda_ == 0.0
