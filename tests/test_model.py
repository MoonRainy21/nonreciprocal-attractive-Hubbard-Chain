from __future__ import annotations

import numpy as np
import pytest

from nhbdg.model import Chain, bdg_matrix, hopping


@pytest.mark.parametrize("L", [2, 3, 4])
def test_hopping_direction_and_transpose_identity(L: int) -> None:
    chain = Chain(L=L, U=2.0, t=1.7, g=0.31, lambda_=1.0)
    matrix = hopping(chain)
    if L == 2:
        assert matrix[1, 0] == pytest.approx(-chain.t * (np.exp(chain.g) + np.exp(-chain.g)))
    else:
        assert matrix[1, 0] == pytest.approx(-chain.t * np.exp(chain.g))
        assert matrix[0, 1] == pytest.approx(-chain.t * np.exp(-chain.g))
        assert matrix[0, -1] == pytest.approx(-chain.t * np.exp(chain.g))
    assert np.linalg.norm(matrix.T - hopping(Chain(L, 2.0, t=1.7, g=-0.31, lambda_=1.0))) < 1.0e-14


def test_bdg_lower_block_is_transpose() -> None:
    chain = Chain(3, 2.0, g=0.4)
    up, down = np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.2, 0.1])
    plus = np.array([0.2, 0.3, 0.4], complex)
    matrix = bdg_matrix(chain, -0.2, up, down, plus, plus.conjugate())
    expected_down = hopping(chain) + np.diag(0.2 - chain.U * up)
    assert np.linalg.norm(matrix[3:, 3:] + expected_down.T) < 1.0e-14
