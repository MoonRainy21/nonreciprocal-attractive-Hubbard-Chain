# Numerical review audit

This revision turns the complex-spectrum claims into gated, reproducible
numerical outputs. A production campaign is complete only when all checks in
`production_status.json` pass.

## Occupied branch

- Forward and reverse continuation use the same biorthogonal eigenvector
  assignment and adaptive midpoint refinement.
- Every accepted point passes right-subspace overlap, projector idempotency,
  full left/right biorthogonality, and occupied--unoccupied cluster-separation
  gates.
- Stored diagnostics include the real-line gap, projector norm, and the
  difference between the transported projector and the instantaneous
  `Re(E)<0` projector.
- Reverse endpoints are compared with the original OBC pair product, density,
  spectrum, and occupied projector.

This is an eigen-subspace continuation, not a claim of physical adiabatic time
evolution. The code does not relabel the branch independently at every link.

## Fixed filling

- The chemical potential remains real.
- Density imaginary components are measured before a real Hartree field is
  formed. A trial is rejected when they exceed the configured tolerance.
- Local fixed-sheet `N(mu)` scans are performed at OBC, an intermediate link,
  and PBC. Production requires a monotone curve and exactly one target-filling
  crossing in each scan.

## PBC control and crossover

- PBC outputs report the global Nambu-conjugacy residual, fitted positive
  scale, and maximum imaginary pair product.
- Both `g=0.05` and `g=0.10` branches are included in full endpoint and reverse
  audits. The matched `gL=2` pair `(L,g)=(20,0.10),(40,0.05)` is included.
- Processing quantifies curve scatter on common logarithmic `lambda` and `chi`
  intervals and requires the `chi` scatter to be smaller.
- Pair-deformation thresholds are recomputed for centered one-third, one-half,
  two-thirds, and full-chain windows.

## Reproducibility

Production rows record the Git commit, dirty-worktree flag, diff hash, config
hash, accepted and rejected adaptive trials, seeds, residuals, and all branch
diagnostics. Dirty or unknown revisions cannot produce a `COMPLETE` status.
`make archive` creates a checksummed archive containing raw states, processed
tables, source, tests, configs, and figures for persistent deposit.
