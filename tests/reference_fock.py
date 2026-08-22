"""Independent small-Fock-space reference for the correlation convention.

This test helper intentionally does not import projector extraction from the
production library.  It constructs creation/annihilation matrices directly in
the occupation basis for up to two lattice sites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _annihilation(mode: int, modes: int) -> NDArray[np.complex128]:
    dimension = 1 << modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for ket in range(dimension):
        if (ket >> mode) & 1:
            bra = ket ^ (1 << mode)
            sign = -1.0 if ((ket & ((1 << mode) - 1)).bit_count() % 2) else 1.0
            result[bra, ket] = sign
    return result


def quadratic_fock_observables(
    h_up: NDArray[np.complex128],
    h_down: NDArray[np.complex128],
    delta_plus: NDArray[np.complex128],
    delta_minus: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128], NDArray[np.complex128]]:
    """Diagonalize an independently constructed Hermitian quadratic Fock Hamiltonian."""

    L = h_up.shape[0]
    if L > 2:
        raise ValueError("Reference Fock test is intentionally limited to L <= 2.")
    modes = 2 * L
    c = [_annihilation(mode, modes) for mode in range(modes)]
    H = np.zeros_like(c[0])
    for i in range(L):
        for j in range(L):
            H += h_up[i, j] * c[i].conj().T @ c[j]
            H += h_down[i, j] * c[L + i].conj().T @ c[L + j]
        H += delta_plus[i] * c[i].conj().T @ c[L + i].conj().T
        H += delta_minus[i] * c[L + i] @ c[i]
    values, vectors = np.linalg.eigh(H)
    ground = vectors[:, np.argmin(values)]
    expectation = lambda operator: np.vdot(ground, operator @ ground)
    n_up = np.array([expectation(c[i].conj().T @ c[i]).real for i in range(L)])
    n_down = np.array([expectation(c[L + i].conj().T @ c[L + i]).real for i in range(L)])
    F = np.array([expectation(c[L + i] @ c[i]) for i in range(L)])
    F_bar = np.array([expectation(c[i].conj().T @ c[L + i].conj().T) for i in range(L)])
    return n_up, n_down, F, F_bar
