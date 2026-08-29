from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.process import _collapse_quality, _matched_gl_quality


def _audit_rows() -> pd.DataFrame:
    rows = []
    for L, g, factor in ((20, 0.1, np.exp(2.0)), (40, 0.05, np.exp(2.0)), (24, 0.05, np.exp(1.2))):
        for chi in np.logspace(-3, 0, 7):
            rows.append({
                "kind": "branch_trial",
                "status": "SUCCESS",
                "accepted": True,
                "L": L,
                "g": g,
                "lambda": chi / factor,
                "chi": chi,
                "metric_violation": chi,
            })
    return pd.DataFrame(rows)


def test_chi_collapse_scatter_is_quantified() -> None:
    quality = _collapse_quality(_audit_rows()).set_index("coordinate")
    assert quality.loc["chi", "mean_log10_std"] < quality.loc["lambda", "mean_log10_std"]


def test_equal_gl_pair_is_reported() -> None:
    matched = _matched_gl_quality(_audit_rows())
    pair = matched[
        (matched["L_first"] == 20)
        & np.isclose(matched["g_first"], 0.1)
        & (matched["L_second"] == 40)
        & np.isclose(matched["g_second"], 0.05)
    ]
    assert len(pair) == 1
    assert pair["mean_abs_log10_difference"].iloc[0] < 1.0e-12
