"""Regression checks for the processed publication-figure data contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhbdg.figures import figure03, figure_s3


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "figure_data"


def test_canonical_g005_branch_is_not_duplicated_in_fig3() -> None:
    """Canonical Fig. 3 g=.05 rows must originate from the full Fig. 4 path."""

    fig3 = pd.read_csv(DATA / "fig3.csv")
    canonical = fig3[np.isclose(fig3["g"], 0.05)]
    assert not canonical.empty
    assert canonical["run_id"].str.startswith("fig4_").all()
    assert canonical["branch_id"].isin({"U2_L24_g0p05", "U2_L40_g0p05"}).all()


def test_manuscript_gamma_output_is_a_bracket_not_a_point_claim() -> None:
    """Fig. 3 does not consume a scalar gamma threshold; brackets remain diagnostic data."""

    brackets = pd.read_csv(DATA / "gamma_brackets.csv")
    assert {"chi_below", "chi_above"}.issubset(brackets.columns)
    assert (brackets["status"] == "BRACKETED").all()
    assert "gamma" not in inspect.getsource(figure03).lower()


def test_not_reached_pair_threshold_is_explicit() -> None:
    thresholds = pd.read_csv(DATA / "thresholds.csv")
    row = thresholds[
        (thresholds["study"] == "fig3")
        & (thresholds["L"] == 40)
        & np.isclose(thresholds["g"], 0.05)
        & (thresholds["quantity"] == "pair")
    ].iloc[0]
    assert row["status"] == "NOT_REACHED"
    assert np.isnan(row["chi_c"])
    assert np.isfinite(row["max_chi"])


def test_s3_uses_canonical_fig3_data_only() -> None:
    source = inspect.getsource(figure_s3)
    assert '"fig4.csv"' not in source


def test_complete_status_requires_saved_paired_route_pass() -> None:
    status = json.loads((DATA / "production_status.json").read_text(encoding="utf-8"))
    assert status["paired_route_audit_required"] is True
    assert status["paired_route_audit_verdict"] in {"PASS", "PASS_WITH_NOTE"}
    assert status["final_status"] == "COMPLETE"
