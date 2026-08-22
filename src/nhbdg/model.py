"""Hamiltonian and exact OBC similarity conventions.

The site index is always ``j=0,...,L-1`` and the Nambu basis is
``(c_up, c_down^dagger)``.  Keeping all matrix conventions here makes the
sign of every hopping element and the lower-block transpose auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class Chain:
    """Physical parameters of one zero-temperature Hubbard-chain run."""

    L: int
    U: float
    g: float = 0.0
    lambda_: float = 0.0
    filling: float = 0.8
    t: float = 1.0

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError("L must be at least two.")
        if self.U < 0.0:
            raise ValueError("U is the non-negative attraction magnitude.")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda must be in [0, 1].")
        if not 0.0 <= self.filling <= 2.0:
            raise ValueError("The total filling must be in [0, 2].")

    @property
    def q(self) -> float:
        """Accumulated OBC nonreciprocity ``g(L-1)``."""

        return self.g * (self.L - 1)

    def with_link(self, lambda_: float) -> "Chain":
        """Return the same chain with a new boundary-link strength."""

        return replace(self, lambda_=lambda_)


@dataclass(frozen=True)
class Numerics:
    """Numerical, rather than physical, inputs shared by all studies."""

    field_tolerance: float = 1.0e-10
    density_tolerance: float = 1.0e-10
    number_tolerance: float = 1.0e-9
    max_scf_iterations: int = 5000
    max_mu_iterations: int = 42
    mixing: float = 0.30
    minimum_branch_overlap: float = 0.70
    minimum_lambda_step: float = 1.0e-14
    consecutive_converged_iterations: int = 3
    mu_bounds: tuple[float, float] = (-8.0, 8.0)
    random_seed: int = 1729


def coordinates(L: int) -> RealArray:
    """Centered coordinate ``x_j=j-(L-1)/2``."""

    return np.arange(L, dtype=float) - 0.5 * (L - 1)


def hopping(chain: Chain) -> ComplexArray:
    """Return the directed hopping matrix in the paper convention.

    ``h[j+1,j]=-t exp(g)``, ``h[j,j+1]=-t exp(-g)`` and the same directed
    entries close the boundary when ``lambda_`` is nonzero.
    """

    h = np.zeros((chain.L, chain.L), dtype=np.complex128)
    forward, backward = -chain.t * np.exp(chain.g), -chain.t * np.exp(-chain.g)
    for j in range(chain.L - 1):
        h[j + 1, j], h[j, j + 1] = forward, backward
    # += keeps the L=2 PBC convention explicit: the closing bond is a second
    # directed contribution on the only site pair.
    h[0, chain.L - 1] += chain.lambda_ * forward
    h[chain.L - 1, 0] += chain.lambda_ * backward
    return h


def bdg_matrix(
    chain: Chain,
    mu: float,
    n_up: RealArray,
    n_down: RealArray,
    delta_plus: ComplexArray,
    delta_minus: ComplexArray,
) -> ComplexArray:
    """Build the Hartree HFB matrix with independent anomalous fields.

    The lower normal block is exactly ``-h_down.T``.  It is not a Hermitian
    adjoint, because the non-Hermitian HFB convention is biorthogonal.
    """

    vectors = (n_up, n_down, delta_plus, delta_minus)
    if any(np.asarray(vector).shape != (chain.L,) for vector in vectors):
        raise ValueError("All HFB fields must have one value per site.")
    h0 = hopping(chain)
    h_up = h0 + np.diag(-mu - chain.U * n_down)
    h_down = h0 + np.diag(-mu - chain.U * n_up)
    return np.block([[h_up, np.diag(delta_plus)], [np.diag(delta_minus), -h_down.T]])


def nambu_similarity(chain: Chain) -> ComplexArray:
    """Return ``V=diag(D,D^-1)``, with ``D_jj=exp(-g x_j)``."""

    D = np.exp(-chain.g * coordinates(chain.L))
    return np.diag(np.concatenate((D, 1.0 / D))).astype(np.complex128)


def map_fields_from_hermitian(delta: ComplexArray, g: float) -> tuple[ComplexArray, ComplexArray]:
    """Map an OBC Hermitian gap to original-frame ``Delta_+`` and ``Delta_-``."""

    x = coordinates(delta.size)
    return np.exp(2.0 * g * x) * delta, np.exp(-2.0 * g * x) * np.conjugate(delta)


def strip_similarity(delta_plus: ComplexArray, delta_minus: ComplexArray, g: float) -> tuple[ComplexArray, ComplexArray]:
    """Remove reciprocal OBC component-field weights."""

    x = coordinates(delta_plus.size)
    return np.exp(-2.0 * g * x) * delta_plus, np.exp(2.0 * g * x) * delta_minus


def pbc_blocks(
    chain: Chain, mu: float, density_per_spin: float, delta_plus: complex, delta_minus: complex
) -> tuple[RealArray, ComplexArray]:
    """Return the real-space-convention PBC 2-by-2 momentum BdG blocks."""

    if chain.lambda_ != 1.0:
        raise ValueError("Momentum blocks require lambda=1.")
    k = 2.0 * np.pi * np.arange(chain.L, dtype=float) / chain.L
    h_k = -chain.t * (np.exp(chain.g - 1j * k) + np.exp(-chain.g + 1j * k))
    hartree = -mu - chain.U * density_per_spin
    blocks = np.empty((chain.L, 2, 2), dtype=np.complex128)
    for index, value in enumerate(h_k):
        blocks[index] = [[value + hartree, delta_plus], [delta_minus, -h_k[-index % chain.L] - hartree]]
    return k, blocks
