"""Publication figures built exclusively from processed numerical data.

This module deliberately contains no solver imports.  It is the final visual
layer of the reproduction pipeline: figures are redrawn from CSV tables, with
one explicit legend/layout decision per logical panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Colour-blind-safe colours, with a marker and a line style in addition to
# colour whenever curves represent different physical branches.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
LINESTYLES = ("-", "--", "-.", ":")
BRANCH_KEYS = ((24, 0.05), (24, 0.10), (40, 0.05), (40, 0.10))


plt.rcParams.update(
    {
        "font.size": 8.7,
        "axes.labelsize": 8.7,
        "axes.titlesize": 8.3,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def _read(path: Path) -> pd.DataFrame:
    """Read a figure table, giving a useful empty frame for optional tables."""

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _accepted(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only accepted continuation states used in manuscript figures."""

    if frame.empty:
        return frame.copy()
    return frame.loc[
        (frame["kind"] == "branch_trial")
        & (frame["status"] == "SUCCESS")
        & frame["accepted"].astype(bool)
        & (frame["s_min"] >= 0.70)
    ].copy()


def _branch_style(length: int, g: float) -> tuple[str, str, str]:
    """Return one stable colour/marker/line-style encoding per branch."""

    for index, (candidate_length, candidate_g) in enumerate(BRANCH_KEYS):
        if length == candidate_length and np.isclose(g, candidate_g):
            return COLORS[index], MARKERS[index], LINESTYLES[index]
    return "#444444", "o", "-"


def _sparse_every(size: int, maximum: int = 11) -> int:
    return max(1, int(np.ceil(size / maximum)))


def _unique_lambda(frame: pd.DataFrame) -> pd.DataFrame:
    """Adaptive continuation may retain a duplicate accepted endpoint."""

    return frame.sort_values("lambda").drop_duplicates("lambda", keep="last")


def _panel_label(axis: plt.Axes, label: str, *, x: float = -0.16, y: float = 1.035) -> None:
    """Place labels only on logical panels, never on nested mini-axes by default."""

    axis.text(x, y, label, transform=axis.transAxes, fontweight="bold", va="bottom", ha="left", clip_on=False)


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.pdf")
    figure.savefig(output / f"{stem}.png", dpi=300)
    plt.close(figure)


def _run_profile(profiles: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return profiles.loc[profiles["run_id"] == run_id].sort_values("j")


def figure02(data: Path, output: Path) -> None:
    """Fig. 2: direct verification of self-consistent OBC covariance."""

    summary = _read(data / "fig2.csv")
    profiles = _read(data / "profiles.csv")
    q_value = 6.0
    g_value = q_value / 79.0
    profile = profiles.loc[
        (profiles["study"] == "fig2")
        & (profiles["L"] == 80)
        & np.isclose(profiles["g"], g_value, rtol=0.0, atol=1.0e-12)
    ].copy()
    hermitian = profiles.loc[
        (profiles["study"] == "fig2")
        & (profiles["L"] == 80)
        & (profiles["kind"] == "hermitian")
    ].sort_values("j")

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.62))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.785, wspace=0.46, hspace=0.43)
    field_axis, pair_axis = axes[0]
    density_axis, validation_axis = axes[1]

    mapped = profile.loc[profile["kind"] == "mapped"].sort_values("j")
    raw = profile.loc[profile["kind"] == "raw"].sort_values("j")
    rescaled = profile.loc[profile["kind"] == "rescaled"].sort_values("j")
    for colour, column in zip(COLORS[:2], ("delta_plus_abs", "delta_minus_abs")):
        field_axis.semilogy(mapped["j"], mapped[column], color=colour, lw=1.55, zorder=2)
        for marker, source in (("o", raw), ("s", rescaled)):
            field_axis.semilogy(
                source["j"],
                source[column],
                linestyle="None",
                marker=marker,
                markevery=_sparse_every(len(source), 10),
                markerfacecolor="white",
                markeredgewidth=0.9,
                color=colour,
                ms=4.0,
                zorder=3,
            )

    # A factorised, panel-specific key leaves the profile itself unobscured.
    field_key = field_axis.legend(
        handles=[
            Line2D([], [], color=COLORS[0], lw=1.6, label=r"$\Delta_+$"),
            Line2D([], [], color=COLORS[1], lw=1.6, label=r"$\Delta_-$"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.00, 1.365),
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    field_axis.add_artist(field_key)
    field_axis.legend(
        handles=[
            Line2D([], [], color="0.15", lw=1.45, label="mapped"),
            Line2D([], [], color="0.15", marker="o", linestyle="None", markerfacecolor="white", label="direct"),
            Line2D([], [], color="0.15", marker="s", linestyle="None", markerfacecolor="white", label="rescaled"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.00, 1.165),
        ncol=3,
        frameon=False,
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    field_axis.set(xlabel=r"site $j$", ylabel=r"$|\Delta_\pm|/t$")

    if not hermitian.empty:
        pair_axis.plot(hermitian["j"], hermitian["P_real"], color="0.40", linestyle="--", lw=1.25, label=r"Hermitian $|\widetilde\Delta_j|^2/t^2$")
    pair_axis.plot(mapped["j"], mapped["P_real"], color=COLORS[0], lw=1.5, label=r"mapped $\operatorname{Re}P_j/t^2$")
    for marker, source, label in (("o", raw, "direct"), ("s", rescaled, "rescaled")):
        pair_axis.plot(
            source["j"], source["P_real"], linestyle="None", marker=marker,
            markevery=_sparse_every(len(source), 10), markerfacecolor="white",
            markeredgewidth=0.9, color=COLORS[0], ms=3.8, label=label,
        )
    pair_axis.set(xlabel=r"site $j$", ylabel=r"$\operatorname{Re}P_j/t^2$")
    pair_axis.legend(
        handles=[
            Line2D([], [], color="0.40", linestyle="--", lw=1.25, label="Hermitian"),
            Line2D([], [], color=COLORS[0], lw=1.5, label="mapped"),
            Line2D([], [], color=COLORS[0], marker="o", linestyle="None", markerfacecolor="white", label="direct"),
            Line2D([], [], color=COLORS[0], marker="s", linestyle="None", markerfacecolor="white", label="rescaled"),
        ],
        loc="lower left", bbox_to_anchor=(0.00, 1.16), ncol=2, frameon=False,
        handlelength=1.45, columnspacing=0.9, labelspacing=0.28, borderaxespad=0.0,
    )

    if not hermitian.empty:
        density_axis.plot(hermitian["j"], hermitian["density"], color="0.40", linestyle="--", lw=1.25, label="Hermitian reference")
    density_axis.plot(mapped["j"], mapped["density"], color=COLORS[2], lw=1.5, label="mapped OBC")
    for marker, source, label in (("o", raw, "direct"), ("s", rescaled, "rescaled")):
        density_axis.plot(
            source["j"], source["density"], linestyle="None", marker=marker,
            markevery=_sparse_every(len(source), 10), markerfacecolor="white",
            markeredgewidth=0.9, color=COLORS[2], ms=3.8, label=label,
        )
    density_axis.set(xlabel=r"site $j$", ylabel=r"$n_j$")

    categories = (
        ("gap\nmap", "map_error"),
        ("pair\nproduct", "pair_product_error"),
        ("density", "density_error"),
        ("spectrum", "spectrum_error"),
    )
    validation = summary.loc[summary["kind"].isin(["raw", "rescaled"])]
    plot_styles = (("raw", COLORS[0], "o", "direct solve"), ("rescaled", COLORS[1], "s", "rescaled solve"))
    shown_labels: set[str] = set()
    for offset, (kind, colour, marker, label) in zip((-0.13, 0.13), plot_styles):
        values = validation.loc[validation["kind"] == kind]
        if values.empty:
            continue
        for index, (_, column) in enumerate(categories):
            value = float(values[column].max())
            if np.isfinite(value):
                validation_axis.scatter(index + offset, value, color=colour, marker=marker, s=27, zorder=3, label=label if label not in shown_labels else None)
                shown_labels.add(label)
    validation_axis.set_yscale("log")
    validation_axis.set_xticks(range(len(categories)), [label for label, _ in categories])
    validation_axis.set_ylabel("max. relative discrepancy")
    validation_axis.set_title("over validation set", loc="left", fontsize=7.6, pad=4.0)
    validation_axis.grid(axis="y", which="major", color="0.88", lw=0.55)
    validation_axis.legend(loc="lower left", frameon=False, handlelength=1.0, labelspacing=0.25)

    for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        _panel_label(axis, label)
    _save(figure, output, "fig02_obc_covariance")


def figure03(data: Path, output: Path) -> None:
    """Fig. 3: weak-link crossover using each physical branch only once."""

    branch = _accepted(_read(data / "fig3.csv"))
    thresholds = _read(data / "thresholds.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.55))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.835, wspace=0.33, hspace=0.61)
    lambda_axis, chi_axis = axes[0]
    threshold_axis, scale_axis = axes[1]

    branch_handles: list[Line2D] = []
    for length, g in BRANCH_KEYS:
        source = _unique_lambda(branch.loc[(branch["L"] == length) & np.isclose(branch["g"], g) & (branch["lambda"] > 0.0)])
        if source.empty:
            continue
        colour, marker, linestyle = _branch_style(length, g)
        for axis, x_column in ((lambda_axis, "lambda"), (chi_axis, "chi")):
            axis.loglog(
                source[x_column], source["metric_violation"], color=colour, linestyle=linestyle,
                marker=marker, markevery=_sparse_every(len(source), 10), markerfacecolor="white",
                markeredgewidth=0.85, ms=3.9,
            )
        branch_handles.append(Line2D([], [], color=colour, linestyle=linestyle, marker=marker, markerfacecolor="white", label=rf"$L={length},\ g={g:.2f}$"))
    lambda_axis.set(xlabel=r"$\lambda$", ylabel=r"$M_{\rm mc}$")
    chi_axis.set(xlabel=r"$\chi=\lambda e^{gL}$", ylabel=r"$M_{\rm mc}$")
    for axis in (lambda_axis, chi_axis):
        axis.set_ylim(1.0e-11, 1.25)
        axis.grid(which="major", color="0.90", lw=0.55)
    figure.legend(
        handles=branch_handles, loc="upper center", bbox_to_anchor=(0.51, 0.987), ncol=4,
        frameon=False, handlelength=2.0, columnspacing=1.15, borderaxespad=0.0,
    )

    metric_thresholds = thresholds.loc[
        (thresholds["study"] == "fig3")
        & thresholds["quantity"].isin(["metric_1e-4", "metric_1e-3", "metric_1e-2"])
        & np.isfinite(thresholds["chi_c"])
    ].copy()
    metric_styles = {
        "metric_1e-4": ("o", r"$M_{\rm mc}=10^{-4}$"),
        "metric_1e-3": ("s", r"$M_{\rm mc}=10^{-3}$"),
        "metric_1e-2": ("^", r"$M_{\rm mc}=10^{-2}$"),
    }
    threshold_handles = []
    for quantity, (marker, label) in metric_styles.items():
        source = metric_thresholds.loc[metric_thresholds["quantity"] == quantity]
        for _, row in source.iterrows():
            colour, _, _ = _branch_style(int(row["L"]), float(row["g"]))
            threshold_axis.scatter(row["g"] * row["L"], row["chi_c"], color=colour, marker=marker, s=31, zorder=3)
        threshold_handles.append(Line2D([], [], color="0.18", marker=marker, linestyle="None", markerfacecolor="white", label=label))
    threshold_axis.set(xlabel=r"$gL$", ylabel=r"$\chi_c$", yscale="log")
    threshold_axis.grid(axis="y", which="major", color="0.90", lw=0.55)
    threshold_axis.legend(
        handles=threshold_handles, loc="lower left", bbox_to_anchor=(0.00, 1.18), ncol=3,
        frameon=False, handletextpad=0.4, columnspacing=0.9, labelspacing=0.25, borderaxespad=0.0,
    )

    comparison = thresholds.loc[
        (thresholds["study"] == "fig3")
        & np.isclose(thresholds["g"], 0.05)
        & thresholds["quantity"].isin(["metric_1e-2", "pair"])
    ].copy()
    x_locations = {24: 0.0, 40: 1.0}
    comparison_handles = [
        Line2D([], [], color="0.20", marker="o", linestyle="None", label=r"$M_{\rm mc}=10^{-2}$"),
        Line2D([], [], color="0.20", marker="s", linestyle="None", markerfacecolor="white", label=r"$\delta P_{\rm bulk}=10^{-2}$"),
    ]
    for length in (24, 40):
        colour, _, _ = _branch_style(length, 0.05)
        metric = comparison.loc[(comparison["L"] == length) & (comparison["quantity"] == "metric_1e-2")]
        pair = comparison.loc[(comparison["L"] == length) & (comparison["quantity"] == "pair")]
        if not metric.empty and np.isfinite(metric["chi_c"].iloc[0]):
            scale_axis.scatter(x_locations[length] - 0.11, metric["chi_c"].iloc[0], color=colour, marker="o", s=34, zorder=3)
        if not pair.empty and np.isfinite(pair["chi_c"].iloc[0]):
            scale_axis.scatter(x_locations[length] + 0.11, pair["chi_c"].iloc[0], color=colour, marker="s", facecolors="white", linewidths=1.0, s=36, zorder=3)
        elif not pair.empty:
            endpoint = branch.loc[(branch["L"] == length) & np.isclose(branch["g"], 0.05), "chi"].max()
            scale_axis.scatter(x_locations[length] + 0.11, endpoint, color=colour, marker=">", facecolors="white", linewidths=1.0, s=40, zorder=3)
            scale_axis.annotate(
                "not reached\nby PBC", xy=(x_locations[length] + 0.11, endpoint), xytext=(6, -5),
                textcoords="offset points", fontsize=6.9, ha="left", va="top",
            )
    scale_axis.set(
        xticks=[0.0, 1.0], xticklabels=[r"$L=24$", r"$L=40$"],
        ylabel=r"crossover scale $\chi_x$", yscale="log", xlim=(-0.42, 1.58),
    )
    scale_axis.text(0.02, 0.96, r"$g=0.05$", transform=scale_axis.transAxes, va="top")
    scale_axis.grid(axis="y", which="major", color="0.90", lw=0.55)
    scale_axis.legend(handles=comparison_handles, loc="center left", bbox_to_anchor=(0.02, 0.68), frameon=False, handletextpad=0.45, labelspacing=0.3)

    for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        _panel_label(axis, label)
    _save(figure, output, "fig03_weak_link_crossover")


def _snapshot_label(row: pd.Series) -> str:
    if np.isclose(float(row["lambda"]), 0.0):
        return "OBC"
    if np.isclose(float(row["lambda"]), 1.0):
        return "PBC"
    return rf"$\chi={float(row['chi']):.2g}$"


def figure04(data: Path, output: Path) -> None:
    """Fig. 4: canonical post-crossover trajectory and PBC endpoint."""

    fig4 = _read(data / "fig4.csv")
    branch = _accepted(fig4)
    profiles = _read(data / "profiles.csv")
    spectra = _read(data / "spectra.csv")
    snapshots = _read(data / "fig4_snapshots.csv")

    figure = plt.figure(figsize=(7.15, 6.65))
    outer = figure.add_gridspec(2, 2, left=0.105, right=0.985, bottom=0.09, top=0.965, wspace=0.34, hspace=0.46)
    trajectory_axis = figure.add_subplot(outer[0, 0])
    profile_axis = figure.add_subplot(outer[0, 1])
    spectrum_grid = outer[1, 0].subgridspec(1, 3, wspace=0.12)
    spectrum_axes = [figure.add_subplot(spectrum_grid[0, index]) for index in range(3)]
    control_grid = outer[1, 1].subgridspec(1, 2, wspace=0.52)
    pair_control_axis = figure.add_subplot(control_grid[0, 0])
    gamma_control_axis = figure.add_subplot(control_grid[0, 1])

    trajectory = _unique_lambda(branch.loc[(branch["L"] == 40) & np.isclose(branch["g"], 0.05) & (branch["chi"] > 0.0)])
    diagnostics = (
        ("metric_violation", COLORS[0], "o", r"$M_{\rm mc}$"),
        ("delta_P_bulk", COLORS[1], "s", r"$\delta P_{\rm bulk}$"),
    )
    for column, colour, marker, label in diagnostics:
        source = trajectory.loc[trajectory[column] > 0.0]
        trajectory_axis.loglog(
            source["chi"], source[column], color=colour, marker=marker,
            markevery=_sparse_every(len(source), 10), markerfacecolor="white", markeredgewidth=0.85,
            ms=3.9, label=label,
        )
    gamma = trajectory.loc[trajectory["gamma_max_over_t"] > 1.0e-12]
    trajectory_axis.loglog(
        gamma["chi"], gamma["gamma_max_over_t"], color=COLORS[2], marker="^",
        markevery=_sparse_every(len(gamma), 10), markerfacecolor="white", markeredgewidth=0.85,
        ms=3.9, label=r"$\Gamma_{\max}/t$",
    )
    trajectory_axis.set(xlabel=r"$\chi=\lambda e^{gL}$", ylabel="diagnostic magnitude", ylim=(1.0e-10, 1.4))
    trajectory_axis.grid(which="major", color="0.90", lw=0.55)
    trajectory_axis.legend(loc="upper left", frameon=False, handlelength=1.55, labelspacing=0.3)
    trajectory_axis.text(0.97, 0.05, r"$L=40,\ g=0.05$", transform=trajectory_axis.transAxes, ha="right", va="bottom")

    all_snapshots = snapshots.loc[snapshots["L"] == 40].sort_values("lambda")
    snapshot_styles = ((COLORS[0], "-"), (COLORS[1], "--"), (COLORS[2], "-."), (COLORS[3], ":"))
    for (index, (_, row)) in enumerate(all_snapshots.iterrows()):
        profile = _run_profile(profiles, row["run_id"])
        if profile.empty:
            continue
        colour, linestyle = snapshot_styles[min(index, len(snapshot_styles) - 1)]
        profile_axis.plot(profile["j"], profile["P_real"], color=colour, linestyle=linestyle, lw=1.35, label=_snapshot_label(row))
    profile_axis.set(xlabel=r"site $j$", ylabel=r"$\operatorname{Re}P_j/t^2$")
    profile_axis.legend(loc="lower left", frameon=False, handlelength=1.75, labelspacing=0.3)

    # Three readable small multiples are more informative than four overlaid
    # spectra.  The intermediate snapshot is the first nonzero stored state.
    selected_spectra = []
    if not all_snapshots.empty:
        selected_spectra.append(all_snapshots.iloc[0])
        intermediate = all_snapshots.loc[(all_snapshots["lambda"] > 0.0) & (all_snapshots["lambda"] < 1.0)]
        if not intermediate.empty:
            selected_spectra.append(intermediate.iloc[0])
        endpoint = all_snapshots.loc[np.isclose(all_snapshots["lambda"], 1.0)]
        if not endpoint.empty:
            selected_spectra.append(endpoint.iloc[-1])
    selected_spectra = selected_spectra[:3]
    selected_values = spectra.loc[spectra["run_id"].isin([row["run_id"] for row in selected_spectra])]
    x_limit = max(1.0, float(np.nanmax(np.abs(selected_values["E_real"]))) * 1.08) if not selected_values.empty else 1.0
    y_limit = max(0.12, float(np.nanmax(np.abs(selected_values["E_imag"]))) * 1.18) if not selected_values.empty else 0.12
    for index, axis in enumerate(spectrum_axes):
        if index >= len(selected_spectra):
            axis.set_visible(False)
            continue
        row = selected_spectra[index]
        eigenvalues = spectra.loc[spectra["run_id"] == row["run_id"]]
        colour, _ = snapshot_styles[min(index, len(snapshot_styles) - 1)]
        axis.axhline(0.0, color="0.72", lw=0.65, zorder=0)
        axis.scatter(eigenvalues["E_real"], eigenvalues["E_imag"], color=colour, s=8.5, linewidths=0, zorder=2)
        axis.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit), title=_snapshot_label(row), xlabel=r"$\operatorname{Re}E/t$")
        if index == 0:
            axis.set_ylabel(r"$\operatorname{Im}E/t$")
        else:
            axis.tick_params(labelleft=False)
        axis.tick_params(pad=1.5)

    endpoints = branch.loc[np.isclose(branch["lambda"], 1.0)].groupby("L", as_index=False).first()
    controls = fig4.loc[fig4["kind"] == "bandwidth_control"]
    locations = np.array([0.0, 1.0])
    endpoint_pairs, control_pairs, endpoint_gamma, control_gamma = [], [], [], []
    for length in (24, 40):
        endpoint = endpoints.loc[endpoints["L"] == length]
        control = controls.loc[controls["L"] == length]
        endpoint_pairs.append(float(endpoint["bulk_pair_product_real"].iloc[0]) if not endpoint.empty else np.nan)
        control_pairs.append(float(control["bulk_pair_product_real"].iloc[0]) if not control.empty else np.nan)
        endpoint_gamma.append(float(endpoint["gamma_max_over_t"].iloc[0]) if not endpoint.empty else np.nan)
        control_gamma.append(float(control["gamma_max_over_t"].iloc[0]) if not control.empty else np.nan)
    control_handles = [
        Line2D([], [], color=COLORS[0], marker="o", markerfacecolor="white", label="NH PBC"),
        Line2D([], [], color="0.35", linestyle="--", marker="s", markerfacecolor="white", label="bandwidth-matched Hermitian"),
    ]
    for axis, nh_values, hermitian_values, ylabel in (
        (pair_control_axis, endpoint_pairs, control_pairs, r"$P_{\rm bulk}/t^2$"),
        (gamma_control_axis, endpoint_gamma, control_gamma, r"$\Gamma_{\max}/t$"),
    ):
        axis.plot(locations, nh_values, color=COLORS[0], marker="o", markerfacecolor="white", label="NH PBC")
        axis.plot(locations, hermitian_values, color="0.35", linestyle="--", marker="s", markerfacecolor="white", label="bandwidth-matched\nHermitian")
        axis.set(xticks=locations, xticklabels=[r"$L=24$", r"$L=40$"], ylabel=ylabel, xlim=(-0.25, 1.25))
        axis.tick_params(axis="x", pad=1.5)
    gamma_control_axis.set_ylim(bottom=-0.005)
    figure.legend(
        handles=control_handles, loc="lower center", bbox_to_anchor=(0.775, 0.485), ncol=2,
        frameon=False, handlelength=1.45, handletextpad=0.35, columnspacing=0.85,
        borderaxespad=0.0, fontsize=7.0,
    )

    _panel_label(trajectory_axis, "(a)")
    _panel_label(profile_axis, "(b)")
    _panel_label(spectrum_axes[0], "(c)", x=-0.52)
    _panel_label(pair_control_axis, "(d)", x=-0.50)
    _save(figure, output, "fig04_pbc_endpoint")


def figure_s1(data: Path, output: Path) -> None:
    """Supplementary Fig. S1: conditioning and covariance accuracy."""

    conditioning = _read(data / "conditioning.csv")
    raw = conditioning.loc[conditioning["kind"] == "raw"].sort_values("q")
    rescaled = conditioning.loc[conditioning["kind"] == "rescaled"].sort_values("q")
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.90, wspace=0.38)
    condition_axis, error_axis = axes

    representative = raw if not raw.empty else rescaled
    if not representative.empty:
        condition_axis.semilogy(representative["q"], representative["similarity_condition"], color=COLORS[0], marker="o", markerfacecolor="white", label=r"$\kappa(V)$")
        condition_axis.semilogy(representative["q"], representative["right_condition"], color=COLORS[1], linestyle="--", marker="s", markerfacecolor="white", label=r"$\kappa(R)$")
    condition_axis.set(xlabel=r"$q=g(L-1)$", ylabel="condition number")
    condition_axis.grid(which="major", color="0.90", lw=0.55)
    condition_axis.legend(loc="upper left", frameon=False, handlelength=1.55)

    error_source = raw if not raw.empty else rescaled
    if not error_source.empty:
        error_axis.semilogy(
            error_source["q"], error_source["covariance_error"], color=COLORS[0], linestyle="-", marker="o",
            markerfacecolor="white", label="direct / rescaled\n(indistinguishable)",
        )
    error_axis.set(xlabel=r"$q=g(L-1)$", ylabel="relative covariance error")
    error_axis.grid(which="major", color="0.90", lw=0.55)
    error_axis.legend(loc="lower right", frameon=False, handlelength=1.55)

    _panel_label(condition_axis, "(a)")
    _panel_label(error_axis, "(b)")
    _save(figure, output, "figS1_conditioning")


def figure_s2(data: Path, output: Path) -> None:
    """Supplementary Fig. S2: local invariance and directional Green covariance."""

    frequency = _read(data / "green_frequency.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.15))
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.745, wspace=0.39)
    ldos_axis, directional_axis, stripped_axis = axes

    q_handles = []
    for index, (q, source) in enumerate(frequency.groupby("q", sort=True)):
        colour = COLORS[index]
        source = source.sort_values("omega")
        ldos_axis.plot(source["omega"], source["center_ldos"], color=colour, lw=1.35)
        directional_axis.semilogy(source["omega"], source["G_1L_abs"], color=colour, linestyle="-", lw=1.15)
        directional_axis.semilogy(source["omega"], source["G_L1_abs"], color=colour, linestyle="--", lw=1.15)
        stripped_axis.semilogy(source["omega"], source["stripped_1L_abs"], color=colour, linestyle="-", lw=1.15)
        # |G_{L1}| has the opposite known OBC factor from |G_{1L}|.
        stripped_reverse = source["G_L1_abs"].to_numpy(float) * np.exp(-float(q))
        stripped_axis.semilogy(source["omega"], stripped_reverse, color=colour, linestyle="--", lw=1.15)
        q_handles.append(Line2D([], [], color=colour, lw=1.5, label=rf"$q={float(q):g}$"))
    direction_handles = [
        Line2D([], [], color="0.15", linestyle="-", lw=1.4, label=r"$1\!\to\!L$"),
        Line2D([], [], color="0.15", linestyle="--", lw=1.4, label=r"$L\!\to\!1$"),
    ]
    figure.legend(handles=q_handles, loc="upper center", bbox_to_anchor=(0.35, 0.995), ncol=3, frameon=False, title=r"$q=g(L-1)$", title_fontsize=7.4, handlelength=1.7, columnspacing=1.05)
    figure.legend(handles=direction_handles, loc="upper center", bbox_to_anchor=(0.80, 0.995), ncol=2, frameon=False, handlelength=1.8, columnspacing=0.9)

    ldos_axis.set(xlabel=r"$\omega/t$", ylabel=r"$\rho_{\rm center}(\omega)$")
    directional_axis.set(xlabel=r"$\omega/t$", ylabel=r"$|G^{pp}_{ij}(\omega)|$")
    stripped_axis.set(xlabel=r"$\omega/t$", ylabel=r"stripped $|G^{pp}_{ij}(\omega)|$")
    for axis in axes:
        axis.grid(which="major", color="0.91", lw=0.5)
    _panel_label(ldos_axis, "(a)")
    _panel_label(directional_axis, "(b)")
    _panel_label(stripped_axis, "(c)")
    _save(figure, output, "figS2_green_covariance")


def figure_s3(data: Path, output: Path) -> None:
    """Supplementary Fig. S3: accepted canonical branch diagnostics only."""

    branch = _accepted(_read(data / "fig3.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.35))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.84, wspace=0.34, hspace=0.40)
    axes = axes.ravel()
    quantities = (
        ("s_min", r"$s_{\min}$", False),
        ("minimum_eigenvalue_separation", r"minimum eigenvalue separation $/t$", True),
        ("right_condition", r"$\kappa(R)$", True),
        ("field_residual", "SCF field residual", True),
    )

    branch_handles: list[Line2D] = []
    for length, g in BRANCH_KEYS:
        source = _unique_lambda(branch.loc[(branch["L"] == length) & np.isclose(branch["g"], g) & (branch["lambda"] > 0.0)])
        if source.empty:
            continue
        colour, marker, linestyle = _branch_style(length, g)
        for axis, (column, _, logarithmic_y) in zip(axes, quantities):
            valid = source.loc[np.isfinite(source[column]) & (source[column] > 0.0 if logarithmic_y else np.ones(len(source), dtype=bool))]
            axis.plot(
                valid["lambda"], valid[column], color=colour, linestyle=linestyle, marker=marker,
                markevery=_sparse_every(len(valid), 10), markerfacecolor="white", markeredgewidth=0.85, ms=3.7,
            )
        branch_handles.append(Line2D([], [], color=colour, linestyle=linestyle, marker=marker, markerfacecolor="white", label=rf"$L={length},\ g={g:.2f}$"))

    for axis, (_, ylabel, logarithmic_y) in zip(axes, quantities):
        axis.set(xscale="log", xlabel=r"$\lambda$", ylabel=ylabel)
        if logarithmic_y:
            axis.set_yscale("log")
        axis.grid(which="major", color="0.90", lw=0.55)
    axes[0].axhline(0.70, color="0.38", linestyle="--", lw=0.9)
    axes[0].text(0.98, 0.08, "branch acceptance\nthreshold", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.0)
    axes[0].set_ylim(0.66, 1.03)
    axes[3].axhline(1.0e-10, color="0.38", linestyle="--", lw=0.9)
    axes[3].text(0.02, 0.10, "dashed: SCF tolerance", transform=axes[3].transAxes, ha="left", va="bottom", fontsize=7.0)
    figure.legend(handles=branch_handles, loc="upper center", bbox_to_anchor=(0.51, 0.987), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.15)

    for axis, label in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
        _panel_label(axis, label)
    _save(figure, output, "figS3_branch_quality")


def make(data: Path, output: Path, selected: str = "all") -> None:
    """Render one selected figure or the complete publication figure set."""

    functions = {
        "fig02": figure02,
        "fig03": figure03,
        "fig04": figure04,
        "figS1": figure_s1,
        "figS2": figure_s2,
        "figS3": figure_s3,
    }
    for name, function in functions.items():
        if selected in {"all", name}:
            function(data, output)
