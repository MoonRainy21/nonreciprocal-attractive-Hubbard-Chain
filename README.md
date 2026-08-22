# Public numerical code: non-reciprocal Hubbard HFB

This is a standalone zero-temperature numerical package for the paper's
attractive non-reciprocal Hubbard-chain HFB calculations. It does not import
older project code. Interrupted campaigns may reuse only raw states produced
by this package under config hashes explicitly listed in `paper.yaml`; it
never takes legacy processed tables or figures as inputs.

The only production inputs are YAML files. `paper.yaml` defines every physical
scan, numerical tolerance, threshold, and figure range; no paper parameter is
hidden in Python source.

## Reading the code

```text
model.py → solver.py → fixed_filling.py → continuation.py → studies.py
             inner SCF       outer μ search       branch tracking
```

`MeanFieldSolver.solve_at_mu()` solves one self-consistent HFB fixed point at
a prescribed chemical potential. `solve_fixed_filling()` is a separate,
safeguarded bisection that calls that inner solver only after choosing a trial
μ. Weak-link continuation calls the fixed-filling routine and rejects a point
when its SCF, filling, or occupied-subspace gate fails.

## Conventions

Site indices are `j=0,...,L-1`, the Nambu basis is `(c_up, c_down†)`, and

```text
h[j+1,j] = -t exp(g)
h[j,j+1] = -t exp(-g)
h[0,L-1] = -lambda t exp(g)
h[L-1,0] = -lambda t exp(-g)
```

The lower BdG normal block is `-h_down.T`. The biorthogonal projector gives
`n_up=C_pp`, `n_down=1-C_hh`, `Delta_plus=-U C_ph`, and
`Delta_minus=-U C_hp`.

## Commands

```bash
make test        # convention tests
make lint        # Python syntax compilation
make smoke       # small end-to-end calculation of Figs. 2--4 and S1--S3
make clean-output # remove only generated raw/processed/figure/log outputs
make finish       # reuse compatible raw data and calculate only missing branches
make fig2        # Fig. 2 only, from scratch
make fig3        # Fig. 3 only, from scratch
make fig4        # Fig. 4 only, from scratch
make supplement  # Figs. S1--S3 only, from scratch
make reproduce   # tests, final-scope calculations, processing, and figures
```

`make reproduce` writes an ignored live log to `logs/reproduce.log`; monitor
it with `tail -f logs/reproduce.log`. It first removes prior generated output,
so its processed tables cannot mix runs.
Figure 1 is analytical; Figures 2--4 and S1--S3 are regenerated here.

`make finish` is intended for an interrupted production campaign. It leaves
generated files in place, reuses only the config hashes explicitly listed in
`paper.yaml`, and runs any branch whose required validation output is absent.

Processing writes `figure_data/production_status.json`. A main figure is
written only when every branch required by its configured scientific claim is
valid: Fig. 3 needs all three metric-threshold crossings for its four requested
branches, and Fig. 4 needs both requested PBC endpoints. The final status does
not depend on exploratory interaction scans or on $L\geq60$ weak-link runs.
