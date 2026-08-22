# Reproducibility

Use Python 3.11 or later and install dependencies:

```bash
python -m pip install -e '.[test]'
```

Run the complete numerical workflow from `public_numerics/`:

```bash
nohup make reproduce >/dev/null 2>&1 &
```

Live output is written by the Makefile to `logs/reproduce.log` and can be
viewed with `tail -f logs/reproduce.log`.

Every raw run stores its config hash, Git commit, physical parameters,
residuals, branch status, μ-evaluation count, accumulated SCF iterations,
wall time, and state arrays. `data/processed/run_manifest.csv` inventories the
current config's results, and `figure_manifest.json` maps every figure to its
processed input tables. Generated data, figures, logs, and manually authored
reports are ignored by Git;
a clean clone plus `conda run -n cml make reproduce` regenerates the
final-scope numerical outputs. The production matrix is
limited to Fig. 2/S1/S2 validation, the four $U=2$ Fig. 3 branches at
$L=24,40$ and $g=0.05,0.10$, and the $g=0.05$, $L=24,40$ Fig. 4 PBC branches.
