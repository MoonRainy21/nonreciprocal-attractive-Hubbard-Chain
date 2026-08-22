# Fig. 3 / Fig. 4 continuation-consistency audit — interim

## Status: INCONCLUSIVE

The existing accepted Fig. 3 and Fig. 4 trajectories share only one nonzero lambda, `1e-12`, for each of `L=24,40`; this is fewer than the required three useful common points. No final canonical branch, threshold processing, or final replot has been authorized.

At the shared point, route agreement is excellent: maximum distances are `d_P=2.52e-9`, `d_n=2.14e-9`, `d_E=4.18e-9`, `d_C=3.51e-9`, and `|Delta mu|=2.72e-9`. Both routes are real-spectrum there. This verifies only the OBC-near limit, not crossover route independence.

The stored gamma brackets differ because the grids differ. For Fig. 3 versus Fig. 4 they are respectively `[0.04948,0.1]` versus `[0.13895,0.22758]` at `L=24`, and `[0.01212,0.02448]` versus `[0.01931,0.03162]` at `L=40`. These brackets do not determine whether the discrepancy is sampling or distinct saddles.

Required next step: run only the prescribed paired route audit targets, using each route's preceding accepted state as seed. Until those comparisons pass, gamma onset must not be used as a precise scalar and final manuscript figures must remain blocked.
