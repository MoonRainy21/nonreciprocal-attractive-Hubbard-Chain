"""The inner, fixed-chemical-potential self-consistent HFB solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from .model import (
    Chain,
    ComplexArray,
    Numerics,
    RealArray,
    bdg_matrix,
    coordinates,
    nambu_similarity,
)

Representation = Literal["raw", "rescaled", "hermitian"]


@dataclass
class Eigensystem:
    """Biorthogonal BdG eigensystem stored in the physical Nambu frame."""

    values: ComplexArray
    right: ComplexArray
    left: ComplexArray
    occupied: np.ndarray


@dataclass
class HFBState:
    """One fixed-μ or fixed-filling HFB state and its convergence metadata."""

    chain: Chain
    mu: float
    delta_plus: ComplexArray
    delta_minus: ComplexArray
    n_up: RealArray
    n_down: RealArray
    eigensystem: Eigensystem
    field_residual: float
    density_residual: float
    number_residual: float
    iterations: int
    converged: bool
    message: str
    branch_overlap: float = float("nan")
    mu_evaluations: int = 1
    total_scf_iterations: int = 0
    wall_seconds: float = 0.0
    cached_mu_warm_starts: int = 0
    max_density_imaginary: float = 0.0
    total_density_imaginary: float = 0.0

    @property
    def density(self) -> RealArray:
        """Spin-summed site density."""

        return self.n_up + self.n_down

    @property
    def pair_product(self) -> ComplexArray:
        """Similarity-invariant pairing product ``Delta_plus Delta_minus``."""

        return self.delta_plus * self.delta_minus


class MeanFieldSolver:
    """Solve the HFB fixed point at one prescribed chemical potential.

    This class deliberately does *not* search for ``mu``.  The outer
    fixed-filling bisection is implemented in :mod:`nhbdg.fixed_filling` and
    calls :meth:`solve_at_mu` only after selecting a trial chemical potential.
    """

    def __init__(self, chain: Chain, numerics: Numerics, representation: Representation = "raw") -> None:
        if representation not in {"raw", "rescaled", "hermitian"}:
            raise ValueError("representation must be raw, rescaled, or hermitian.")
        self.chain = chain
        self.numerics = numerics
        self.representation = representation

    def solve_at_mu(
        self,
        mu: float,
        initial_state: HFBState | None = None,
        occupied_reference: Eigensystem | None = None,
    ) -> HFBState:
        """Converge fields and Hartree densities at a fixed ``mu``.

        ``initial_state`` provides only a deterministic initial field/density.
        ``occupied_reference`` transports a previously accepted occupied sheet;
        without one, the initial real-spectrum state occupies negative BdG
        energies.  Linear mixing is the only fixed-point accelerator.
        """

        L, U = self.chain.L, self.chain.U
        default_gap = np.full(L, 0.05 * U, dtype=np.complex128)
        if initial_state is None:
            plus, minus = default_gap, np.conjugate(default_gap)
            n_up = np.full(L, self.chain.filling / 2.0)
            n_down = n_up.copy()
        else:
            plus, minus = initial_state.delta_plus.copy(), initial_state.delta_minus.copy()
            n_up, n_down = initial_state.n_up.copy(), initial_state.n_down.copy()
        if U > 0.0 and max(np.linalg.norm(plus), np.linalg.norm(minus)) < 1.0e-8:
            plus, minus = default_gap.copy(), np.conjugate(default_gap)

        plus_i, minus_i = self._to_internal_fields(plus, minus)
        # A Hermitian/OBC fixed-point solve has an unambiguous negative-energy
        # occupied subspace at every SCF iterate.  It therefore must not run
        # the continuation overlap/SVD machinery just because the fields were
        # updated once.  A tracker is used only when the caller explicitly
        # transports an occupied sheet from a previously accepted weak-link
        # state.
        tracker = self._to_solver_frame(occupied_reference)
        transport_occupied_sheet = tracker is not None
        mixing, last_residual, stable = self.numerics.mixing, float("inf"), 0
        last_eigensystem: Eigensystem | None = None
        last_overlap, field_residual, density_residual = float("nan"), float("inf"), float("inf")

        for iteration in range(1, self.numerics.max_scf_iterations + 1):
            physical_plus, physical_minus = self._from_internal_fields(plus_i, minus_i)
            matrix, inverse = self._solver_matrix(mu, n_up, n_down, plus_i, minus_i)
            if not np.all(np.isfinite(matrix)):
                return self._failed_state(
                    mu, physical_plus, physical_minus, n_up, n_down, iteration,
                    "ILL_CONDITIONED: non-finite SCF matrix",
                )
            try:
                eigensystem_i, overlap = self._eigensystem(matrix, tracker)
            except (np.linalg.LinAlgError, ValueError) as error:
                return self._failed_state(mu, physical_plus, physical_minus, n_up, n_down, iteration, str(error))
            correlation = self._physical_correlation(eigensystem_i, inverse)
            density_imaginary = self._density_imaginary(correlation)
            total_density_imaginary = self._total_density_imaginary(correlation)
            if density_imaginary > self.numerics.density_imaginary_tolerance:
                return self._failed_state(
                    mu,
                    physical_plus,
                    physical_minus,
                    n_up,
                    n_down,
                    iteration,
                    (
                        "COMPLEX_DENSITY: unprojected density imaginary part "
                        f"{density_imaginary:.6e} exceeds tolerance "
                        f"{self.numerics.density_imaginary_tolerance:.6e}"
                    ),
                    max_density_imaginary=density_imaginary,
                    total_density_imaginary=total_density_imaginary,
                )
            observed_up, observed_down, F, F_bar = self._observables(correlation)
            physical_eigensystem = self._from_solver_frame(eigensystem_i)
            if not all(np.all(np.isfinite(value)) for value in (observed_up, observed_down, F, F_bar)):
                return self._failed_state(
                    mu, physical_plus, physical_minus, n_up, n_down, iteration,
                    "ILL_CONDITIONED: non-finite correlation update",
                )

            if U == 0.0:
                zeros = np.zeros(L, dtype=np.complex128)
                return HFBState(
                    self.chain, mu, zeros, zeros, observed_up, observed_down, physical_eigensystem,
                    0.0, 0.0, abs(float(np.mean(observed_up + observed_down)) - self.chain.filling),
                    iteration, True, "SUCCESS", overlap,
                    max_density_imaginary=density_imaginary,
                    total_density_imaginary=total_density_imaginary,
                )

            target_plus, target_minus = -U * F, -U * F_bar
            target_plus_i, target_minus_i = self._to_internal_fields(target_plus, target_minus)
            field_residual = max(
                float(np.linalg.norm(target_plus - physical_plus) / max(np.linalg.norm(target_plus), np.linalg.norm(physical_plus), 1.0)),
                float(np.linalg.norm(target_minus - physical_minus) / max(np.linalg.norm(target_minus), np.linalg.norm(physical_minus), 1.0)),
            )
            density_residual = max(
                float(np.linalg.norm(observed_up - n_up) / max(np.linalg.norm(observed_up), np.linalg.norm(n_up), 1.0)),
                float(np.linalg.norm(observed_down - n_down) / max(np.linalg.norm(observed_down), np.linalg.norm(n_down), 1.0)),
            )
            residual = max(field_residual, density_residual)
            if residual > 2.0 * last_residual and mixing > 0.05:
                mixing *= 0.5
            last_residual = residual
            stable = (
                stable + 1
                if field_residual < self.numerics.field_tolerance
                and density_residual < self.numerics.density_tolerance
                else 0
            )
            last_eigensystem, last_overlap = physical_eigensystem, overlap
            if stable >= self.numerics.consecutive_converged_iterations:
                return HFBState(
                    self.chain, mu, physical_plus, physical_minus, n_up, n_down, physical_eigensystem,
                    field_residual, density_residual,
                    abs(float(np.mean(observed_up + observed_down)) - self.chain.filling),
                    iteration, True, "SUCCESS", overlap,
                    max_density_imaginary=density_imaginary,
                    total_density_imaginary=total_density_imaginary,
                )
            plus_i = (1.0 - mixing) * plus_i + mixing * target_plus_i
            minus_i = (1.0 - mixing) * minus_i + mixing * target_minus_i
            n_up = (1.0 - mixing) * n_up + mixing * observed_up
            n_down = (1.0 - mixing) * n_down + mixing * observed_down
            tracker = eigensystem_i if transport_occupied_sheet else None

        physical_plus, physical_minus = self._from_internal_fields(plus_i, minus_i)
        empty = Eigensystem(np.empty(0, complex), np.empty((0, 0), complex), np.empty((0, 0), complex), np.empty(0, bool))
        return HFBState(
            self.chain, mu, physical_plus, physical_minus, n_up, n_down,
            last_eigensystem or empty, field_residual, density_residual,
            abs(float(np.mean(n_up + n_down)) - self.chain.filling),
            self.numerics.max_scf_iterations, False, "NO_SCF_CONVERGENCE", last_overlap,
        )

    def gap_map_once(
        self, mu: float, n_up: RealArray, n_down: RealArray, delta_plus: ComplexArray, delta_minus: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]:
        """Apply one unmixed physical-frame gap map for the OBC validation gate."""

        plus_i, minus_i = self._to_internal_fields(delta_plus, delta_minus)
        matrix, inverse = self._solver_matrix(mu, n_up, n_down, plus_i, minus_i)
        eig, _ = self._eigensystem(matrix, None)
        _, _, F, F_bar = self._observables(self._physical_correlation(eig, inverse))
        return -self.chain.U * F, -self.chain.U * F_bar

    def _solver_matrix(
        self, mu: float, n_up: RealArray, n_down: RealArray, plus_i: ComplexArray, minus_i: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray | None]:
        if self.representation != "rescaled":
            return bdg_matrix(self.chain, mu, n_up, n_down, plus_i, minus_i), None
        V = nambu_similarity(self.chain)
        inverse = np.diag(1.0 / np.diag(V))
        if self.chain.lambda_ == 0.0:
            balanced = Chain(self.chain.L, self.chain.U, filling=self.chain.filling, t=self.chain.t)
            return bdg_matrix(balanced, mu, n_up, n_down, plus_i, minus_i), inverse
        raw_plus, raw_minus = self._from_internal_fields(plus_i, minus_i)
        return V @ bdg_matrix(self.chain, mu, n_up, n_down, raw_plus, raw_minus) @ inverse, inverse

    def _to_internal_fields(self, plus: ComplexArray, minus: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
        if self.representation != "rescaled":
            return plus.copy(), minus.copy()
        x = coordinates(self.chain.L)
        return np.exp(-2.0 * self.chain.g * x) * plus, np.exp(2.0 * self.chain.g * x) * minus

    def _from_internal_fields(self, plus: ComplexArray, minus: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
        if self.representation != "rescaled":
            return plus.copy(), minus.copy()
        x = coordinates(self.chain.L)
        return np.exp(2.0 * self.chain.g * x) * plus, np.exp(-2.0 * self.chain.g * x) * minus

    def _to_solver_frame(self, eig: Eigensystem | None) -> Eigensystem | None:
        if eig is None or self.representation != "rescaled" or eig.right.size == 0:
            return eig
        V = nambu_similarity(self.chain)
        inverse = np.diag(1.0 / np.diag(V))
        return Eigensystem(eig.values, V @ eig.right, inverse @ eig.left, eig.occupied.copy())

    def _from_solver_frame(self, eig: Eigensystem) -> Eigensystem:
        if self.representation != "rescaled" or eig.right.size == 0:
            return eig
        V = nambu_similarity(self.chain)
        inverse = np.diag(1.0 / np.diag(V))
        return Eigensystem(eig.values, inverse @ eig.right, V @ eig.left, eig.occupied.copy())

    @staticmethod
    def _eigensystem(matrix: ComplexArray, previous: Eigensystem | None) -> tuple[Eigensystem, float]:
        hermitian = np.allclose(matrix, matrix.conj().T, rtol=1.0e-12, atol=1.0e-12)
        if hermitian:
            values, right = scipy.linalg.eigh(matrix, check_finite=False)
            left = right.astype(np.complex128)
        else:
            values, left_raw, right = scipy.linalg.eig(matrix, left=True, right=True, check_finite=False)
            overlap = left_raw.conj().T @ right
            correction = scipy.linalg.solve(
                overlap.conj().T,
                np.eye(overlap.shape[0], dtype=np.complex128),
                assume_a="gen",
                check_finite=False,
            )
            left = left_raw @ correction
        if previous is None:
            if np.any(np.abs(values) < 1.0e-10):
                raise ValueError("DEFECTIVE_EIGENSYSTEM: zero-energy occupation ambiguity")
            occupied = np.real(values) < 0.0
            if not np.any(occupied) or np.all(occupied):
                raise ValueError("DEFECTIVE_EIGENSYSTEM: empty or full BdG occupied sheet")
            return Eigensystem(values, right, left, occupied), float("nan")
        assignment = np.abs(previous.left.conj().T @ right)
        old, new = linear_sum_assignment(-assignment)
        occupied = np.zeros(values.size, dtype=bool)
        occupied[new] = previous.occupied[old]
        old_q = scipy.linalg.orth(previous.right[:, previous.occupied])
        new_q = scipy.linalg.orth(right[:, occupied])
        singular = scipy.linalg.svdvals(old_q.conj().T @ new_q)
        return Eigensystem(values, right, left, occupied), float(np.min(singular))

    @staticmethod
    def _observables(correlation: ComplexArray) -> tuple[RealArray, RealArray, ComplexArray, ComplexArray]:
        L = correlation.shape[0] // 2
        n_up = np.real(np.diag(correlation[:L, :L])).astype(float)
        n_down = np.real(1.0 - np.diag(correlation[L:, L:])).astype(float)
        return n_up, n_down, np.diag(correlation[:L, L:]).copy(), np.diag(correlation[L:, :L]).copy()

    @staticmethod
    def _density_imaginary(correlation: ComplexArray) -> float:
        """Return the largest imaginary density component before projection."""

        L = correlation.shape[0] // 2
        up = np.diag(correlation[:L, :L])
        down = 1.0 - np.diag(correlation[L:, L:])
        return float(max(np.max(np.abs(np.imag(up))), np.max(np.abs(np.imag(down)))))

    @staticmethod
    def _total_density_imaginary(correlation: ComplexArray) -> float:
        """Return the imaginary part of the mean spin-summed density."""

        L = correlation.shape[0] // 2
        up = np.diag(correlation[:L, :L])
        down = 1.0 - np.diag(correlation[L:, L:])
        return float(abs(np.imag(np.mean(up + down))))

    @staticmethod
    def _physical_correlation(eig: Eigensystem, inverse: ComplexArray | None) -> ComplexArray:
        right, left = eig.right[:, eig.occupied], eig.left[:, eig.occupied]
        correlation = right @ left.conj().T
        return correlation if inverse is None else inverse @ correlation @ np.linalg.inv(inverse)

    def _failed_state(
        self, mu: float, plus: ComplexArray, minus: ComplexArray, n_up: RealArray,
        n_down: RealArray, iteration: int, error: str, max_density_imaginary: float = 0.0,
        total_density_imaginary: float = 0.0,
    ) -> HFBState:
        empty = Eigensystem(np.empty(0, complex), np.empty((0, 0), complex), np.empty((0, 0), complex), np.empty(0, bool))
        return HFBState(
            self.chain, mu, plus, minus, n_up, n_down, empty, float("inf"), float("inf"),
            abs(float(np.mean(n_up + n_down)) - self.chain.filling), iteration, False,
            error if error.startswith("COMPLEX_DENSITY:") else f"DEFECTIVE_EIGENSYSTEM: {error}",
            max_density_imaginary=max_density_imaginary,
            total_density_imaginary=total_density_imaginary,
        )


def occupied_projector(eigensystem: Eigensystem) -> ComplexArray:
    """Return ``R_occ L_occ^dagger`` in the frame of ``eigensystem``."""

    return (
        eigensystem.right[:, eigensystem.occupied]
        @ eigensystem.left[:, eigensystem.occupied].conj().T
    )
