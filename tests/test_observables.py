from __future__ import annotations

import numpy as np

from nhbdg.model import Chain, bdg_matrix, map_fields_from_hermitian
from nhbdg.observables import green, green_covariance_error


def test_direct_green_function_obeys_obc_similarity() -> None:
    reference = Chain(4, 2.0)
    mapped = Chain(4, 2.0, g=0.2)
    density = np.full(4, 0.4)
    gap = np.full(4, 0.3, complex)
    plus, minus = map_fields_from_hermitian(gap, mapped.g)
    H0 = bdg_matrix(reference, -0.5, density, density, gap, gap)
    Hg = bdg_matrix(mapped, -0.5, density, density, plus, minus)
    assert green_covariance_error(green(Hg, 0.3, 0.02), green(H0, 0.3, 0.02), mapped) < 1.0e-10
