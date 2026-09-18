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
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
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
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.4,
        "ytick.major.size": 3.4,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "lines.linewidth": 1.15,
        "lines.markersize": 4.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "axes.axisbelow": True,
        "figure.constrained_layout.h_pad": 0.09,
        "figure.constrained_layout.w_pad": 0.06,
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


def _panel_label(axis: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.015) -> None:
    """Place labels only on logical panels, never on nested mini-axes by default."""

    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="normal",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    # Use consistent journal-size typography, including compact nested panels.
    for axis in figure.axes:
        for legend in [axis.get_legend()]:
            if legend is not None:
                for label in legend.get_texts():
                    label.set_fontsize(max(7.0, label.get_fontsize()))
        for line in axis.get_xgridlines() + axis.get_ygridlines():
            line.set_color("0.93")
            line.set_linewidth(0.45)
    output.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight", "pad_inches": 0.035, "facecolor": "white"}
    figure.savefig(output / f"{stem}.pdf", **save_options)
    figure.savefig(output / f"{stem}.png", dpi=600, **save_options)
    plt.close(figure)


def _legend_above(axis: plt.Axes, *, ncol: int = 2, **kwargs) -> None:
    """Reserve layout space for a legend instead of covering numerical data."""
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.025), borderaxespad=0.0,
        frameon=False, ncol=ncol, handlelength=1.7, handletextpad=0.4,
        columnspacing=0.9, **kwargs,
    )


def _run_profile(profiles: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return profiles.loc[profiles["run_id"] == run_id].sort_values("j")


def _branch_source(branch: pd.DataFrame, length: int, g: float) -> pd.DataFrame:
    """Return the unique accepted nonzero points of one canonical branch."""

    return _unique_lambda(
        branch.loc[
            (branch["L"] == length)
            & np.isclose(branch["g"], g)
            & (branch["lambda"] > 0.0)
        ]
    )


def _branch_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for length, g in BRANCH_KEYS:
        colour, marker, linestyle = _branch_style(length, g)
        handles.append(
            Line2D(
                [],
                [],
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white",
                label=rf"$L={length},\ g={g:.2f}$",
            )
        )
    return handles


def _plot_validation_summary(axis: plt.Axes, summary: pd.DataFrame) -> None:
    """Plot the unchanged Fig. 2 validation summary in supplementary Fig. S1."""

    categories = (
        ("gap\nmap", "map_error"),
        ("pair\nproduct", "pair_product_error"),
        ("density", "density_error"),
        ("spectrum", "spectrum_error"),
    )
    validation = summary.loc[summary["kind"].isin(["raw", "rescaled"])]
    plot_styles = (
        ("raw", COLORS[0], "o", "direct solve"),
        ("rescaled", COLORS[1], "s", "rescaled solve"),
    )
    shown_labels: set[str] = set()
    for offset, (kind, colour, marker, label) in zip((-0.13, 0.13), plot_styles):
        values = validation.loc[validation["kind"] == kind]
        if values.empty:
            continue
        for index, (_, column) in enumerate(categories):
            value = float(values[column].max())
            if np.isfinite(value):
                axis.scatter(
                    index + offset,
                    value,
                    color=colour,
                    marker=marker,
                    s=23,
                    zorder=3,
                    label=label if label not in shown_labels else None,
                )
                shown_labels.add(label)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axis.set_ylim(bottom=0)
    axis.set_xticks(range(len(categories)), [label for label, _ in categories])
    axis.set_ylabel("max. relative discrepancy")
    axis.grid(axis="y", which="major", color="0.88", lw=0.5)
    _legend_above(axis)


def figure02(data: Path, output: Path) -> None:
    """Fig. 2: direct verification of self-consistent OBC covariance."""

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

    figure = plt.figure(figsize=(7.0, 4.3), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.12, 1.0), wspace=0.18, hspace=0.14)
    field_axis = figure.add_subplot(grid[0, :])
    pair_axis = figure.add_subplot(grid[1, 0])
    density_axis = figure.add_subplot(grid[1, 1])

    mapped = profile.loc[profile["kind"] == "mapped"].sort_values("j")
    raw = profile.loc[profile["kind"] == "raw"].sort_values("j")
    rescaled = profile.loc[profile["kind"] == "rescaled"].sort_values("j")
    for colour, column in zip(COLORS[:2], ("delta_plus_abs", "delta_minus_abs")):
        field_axis.semilogy(mapped["j"], mapped[column], color=colour, lw=1.25, zorder=2)
        for marker, source, marker_start in (("o", raw, 0), ("s", rescaled, _sparse_every(len(rescaled), 10) // 2)):
            field_axis.semilogy(
                source["j"],
                source[column],
                linestyle="None",
                marker=marker,
                markevery=(marker_start, _sparse_every(len(source), 10)),
                markerfacecolor="white",
                markeredgewidth=0.9,
                color=colour,
                ms=4.0,
                zorder=3,
            )
    figure.legend(
        handles=[
            Line2D([], [], color=COLORS[0], lw=1.3, label=r"$\Delta_+$"),
            Line2D([], [], color=COLORS[1], lw=1.3, label=r"$\Delta_-$"),
            Line2D([], [], color="none", linestyle="None", label=""),
            Line2D([], [], color="0.40", linestyle="--", lw=1.1, label="Hermitian"),
            Line2D([], [], color="0.15", lw=1.2, label="mapped"),
            Line2D([], [], color="0.15", marker="o", linestyle="None", markerfacecolor="white", label="direct"),
            Line2D([], [], color="0.15", marker="s", linestyle="None", markerfacecolor="white", label="rescaled"),
        ],
        loc="outside upper center",
        ncol=7,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    field_axis.set(xlabel=r"site $j$", ylabel=r"$|\Delta_\pm|/t$")
    field_axis.grid(which="major", color="0.91", lw=0.5)

    if not hermitian.empty:
        pair_axis.plot(hermitian["j"], hermitian["P_real"], color="0.40", linestyle="--", lw=1.10, label=r"Hermitian $|\widetilde\Delta_j|^2/t^2$")
    pair_axis.plot(mapped["j"], mapped["P_real"], color=COLORS[0], lw=1.35, label=r"mapped $\operatorname{Re}P_j/t^2$")
    for marker, source, label, marker_start in (("o", raw, "direct", 0), ("s", rescaled, "rescaled", _sparse_every(len(rescaled), 10) // 2)):
        pair_axis.plot(
            source["j"], source["P_real"], linestyle="None", marker=marker,
            markevery=(marker_start, _sparse_every(len(source), 10)), markerfacecolor="white",
            markeredgewidth=0.95, color=COLORS[0], ms=4.0, label=label,
        )
    pair_axis.set(xlabel=r"site $j$", ylabel=r"$\operatorname{Re}P_j/t^2$")
    pair_axis.grid(which="major", color="0.91", lw=0.5)

    if not hermitian.empty:
        density_axis.plot(hermitian["j"], hermitian["density"], color="0.40", linestyle="--", lw=1.10, label="Hermitian reference")
    density_axis.plot(mapped["j"], mapped["density"], color=COLORS[2], lw=1.35, label="mapped OBC")
    for marker, source, label, marker_start in (("o", raw, "direct", 0), ("s", rescaled, "rescaled", _sparse_every(len(rescaled), 10) // 2)):
        density_axis.plot(
            source["j"], source["density"], linestyle="None", marker=marker,
            markevery=(marker_start, _sparse_every(len(source), 10)), markerfacecolor="white",
            markeredgewidth=0.95, color=COLORS[2], ms=4.0, label=label,
        )
    density_axis.set(xlabel=r"site $j$", ylabel=r"$n_j$")
    density_axis.grid(which="major", color="0.91", lw=0.5)

    _panel_label(field_axis, "(a)", x=-0.055)
    _panel_label(pair_axis, "(b)", x=-0.13)
    _panel_label(density_axis, "(c)", x=-0.13)
    _save(figure, output, "fig02_obc_covariance")


def figure03(data: Path, output: Path) -> None:
    """Fig. 3: weak-link crossover using each physical branch only once."""

    branch = _accepted(_read(data / "fig3.csv"))
    audit = _accepted(_read(data / "branch_audit.csv"))
    collapse = _read(data / "collapse_quality.csv")
    matched = _read(data / "matched_gl_quality.csv")
    figure = plt.figure(figsize=(7.0, 4.8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.90), wspace=0.12, hspace=0.14)
    lambda_axis = figure.add_subplot(grid[0, 0])
    chi_axis = figure.add_subplot(grid[0, 1])
    quality_axis = figure.add_subplot(grid[1, 0])
    matched_axis = figure.add_subplot(grid[1, 1])

    for length, g in BRANCH_KEYS:
        source = _branch_source(branch, length, g)
        if source.empty:
            continue
        colour, marker, linestyle = _branch_style(length, g)
        for axis, x_column in ((lambda_axis, "lambda"), (chi_axis, "chi")):
            axis.loglog(
                source[x_column], source["metric_violation"], color=colour, linestyle=linestyle,
                marker=marker, markevery=_sparse_every(len(source), 10), markerfacecolor="white",
                markeredgewidth=0.95, ms=4.4, lw=1.45,
            )
    lambda_axis.set(xlabel=r"$\lambda$", ylabel=r"$M_{\rm mc}$")
    chi_axis.set(xlabel=r"$\chi=\lambda e^{gL}$", ylabel=r"$M_{\rm mc}$")
    for axis in (lambda_axis, chi_axis):
        axis.set_ylim(1.0e-5, 1.05)
        axis.grid(which="major", color="0.90", lw=0.55)
        axis.axhspan(1.0e-4, 1.0e-1, color="0.92", zorder=-3)
    lambda_axis.set_xlim(1.0e-4, 1.05)
    chi_axis.set_xlim(1.0e-3, 10.0)
    figure.legend(
        handles=_branch_handles(),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=1.85,
        handletextpad=0.4,
        columnspacing=1.05,
    )

    order = ["lambda", "chi"]
    quality = collapse.set_index("coordinate").reindex(order)
    x = np.arange(2)
    width = 0.27
    mean_bars = quality_axis.bar(x - width / 2, quality["mean_log10_std"], width, color=COLORS[0], edgecolor="0.25", linewidth=0.55, label="mean")
    max_bars = quality_axis.bar(x + width / 2, quality["max_log10_std"], width, color="0.85", hatch="//", edgecolor="0.25", linewidth=0.55, label="maximum")
    quality_axis.bar_label(mean_bars, fmt="%.3f", padding=4, fontsize=7.5)
    quality_axis.bar_label(max_bars, fmt="%.3f", padding=4, fontsize=7.5)
    quality_axis.set(
        xticks=x,
        xticklabels=[r"$\lambda$", r"$\chi$"],
        ylabel=r"std. dev. of $\log_{10}M_{\rm mc}$",
        ylim=(0.0, 1.25),
    )
    _legend_above(quality_axis)

    matched_spec = matched.iloc[0]
    matched_keys = (
        (int(matched_spec["L_first"]), float(matched_spec["g_first"]), COLORS[1], "o", "-"),
        (int(matched_spec["L_second"]), float(matched_spec["g_second"]), COLORS[2], "^", "--"),
    )
    for length, g, colour, marker, linestyle in matched_keys:
        source = _unique_lambda(audit.loc[(audit["L"] == length) & np.isclose(audit["g"], g) & (audit["chi"] > 0.0)])
        matched_axis.loglog(
            source["chi"], source["metric_violation"], color=colour, marker=marker,
            linestyle=linestyle, markevery=_sparse_every(len(source), 9), markerfacecolor="white",
            markeredgewidth=0.95, ms=4.4, lw=1.45, label=rf"$L={length},\ g={g:.2f}$",
        )
    matched_axis.axhspan(1.0e-4, 1.0e-1, color="0.92", zorder=-3)
    matched_axis.set(
        xlabel=r"$\chi=\lambda e^{gL}$", ylabel=r"$M_{\rm mc}$",
        xlim=(1.0e-5, 1.0), ylim=(1.0e-8, 1.05),
    )
    _legend_above(matched_axis)
    matched_axis.grid(which="major", color="0.90", lw=0.55)
    matched_annotation = rf"$gL={float(matched_spec['gL']):.1f}$"
    matched_axis.text(
        0.98, 0.06, matched_annotation,
        transform=matched_axis.transAxes, ha="right", va="bottom", fontsize=7.1,
    )

    _panel_label(lambda_axis, "(a)", x=-0.15)
    _panel_label(chi_axis, "(b)", x=-0.15)
    _panel_label(quality_axis, "(c)", x=-0.15)
    _panel_label(matched_axis, "(d)", x=-0.15)
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

    figure = plt.figure(figsize=(7.0, 5.0), layout="constrained")
    outer = figure.add_gridspec(2, 2, height_ratios=(0.86, 1.0), wspace=0.16, hspace=0.14)
    trajectory_axis = figure.add_subplot(outer[0, 0])
    profile_axis = figure.add_subplot(outer[0, 1])
    spectrum_grid = outer[1, 0].subgridspec(1, 3, wspace=0.08)
    spectrum_axes = [figure.add_subplot(spectrum_grid[0, index]) for index in range(3)]
    control_grid = outer[1, 1].subgridspec(3, 1, height_ratios=(0.16, 1.0, 1.0), hspace=0.12)
    control_header = figure.add_subplot(control_grid[0, 0])
    control_header.set_axis_off()
    pair_control_axis = figure.add_subplot(control_grid[1, 0])
    gamma_control_axis = figure.add_subplot(control_grid[2, 0])

    trajectory = _unique_lambda(branch.loc[(branch["L"] == 40) & np.isclose(branch["g"], 0.05) & (branch["chi"] > 0.0)])
    diagnostics = (
        ("metric_violation", COLORS[0], "o", r"$M_{\rm mc}$"),
        ("delta_P_bulk", COLORS[1], "s", r"$\delta P_{\rm bulk}$"),
    )
    for index, (column, colour, marker, label) in enumerate(diagnostics):
        source = trajectory.loc[trajectory[column] > 0.0]
        trajectory_axis.loglog(
            source["chi"], source[column], color=colour, marker=marker,
            markevery=_sparse_every(len(source), 10), markerfacecolor="white", markeredgewidth=0.85,
            ms=3.9, linestyle=LINESTYLES[index], label=label,
        )
    gamma = trajectory.loc[trajectory["gamma_max_over_t"] > 1.0e-12]
    trajectory_axis.loglog(
        gamma["chi"], gamma["gamma_max_over_t"], color=COLORS[2], marker="^",
        markevery=_sparse_every(len(gamma), 10), markerfacecolor="white", markeredgewidth=0.85,
        ms=3.9, linestyle="-.", label=r"$\Gamma_{\max}/t$",
    )
    trajectory_axis.set(xlabel=r"$\chi=\lambda e^{gL}$", ylabel="diagnostic magnitude", xlim=(1e-4, 10), ylim=(1.0e-8, 1.4))
    trajectory_axis.grid(which="major", color="0.90", lw=0.55)
    _legend_above(trajectory_axis, ncol=3)
    trajectory_axis.text(0.97, 0.05, r"$L=40,\ g=0.05$", transform=trajectory_axis.transAxes, ha="right", va="bottom")

    all_snapshots = snapshots.loc[snapshots["L"] == 40].sort_values("lambda")
    snapshot_styles = ((COLORS[0], "-"), (COLORS[1], "--"), (COLORS[2], "-."), (COLORS[3], ":"))
    for (index, (_, row)) in enumerate(all_snapshots.iterrows()):
        profile = _run_profile(profiles, row["run_id"])
        if profile.empty:
            continue
        colour, linestyle = snapshot_styles[min(index, len(snapshot_styles) - 1)]
        profile_axis.plot(profile["j"], profile["P_real"], color=colour, linestyle=linestyle, lw=1.2, label=_snapshot_label(row))
    profile_axis.set(xlabel=r"site $j$", ylabel=r"$\operatorname{Re}P_j/t^2$")
    _legend_above(profile_axis, ncol=2)

    # Three readable small multiples are more informative than overlaid spectra.
    # Select an existing state with a visibly complex spectrum, rather than the
    # first nonzero continuation point, which is still real within precision.
    selected_spectra = []
    if not all_snapshots.empty:
        selected_spectra.append(all_snapshots.iloc[0])
        complex_candidates = branch.loc[
            (branch["L"] == 40)
            & np.isclose(branch["g"], 0.05)
            & (branch["gamma_max_over_t"] > 1.0e-4)
            & (branch["lambda"] < 1.0)
        ]
        if not complex_candidates.empty:
            target_lambda = 0.08483428982440708
            selected_spectra.append(complex_candidates.loc[(complex_candidates["lambda"] - target_lambda).abs().idxmin()])
        endpoint = all_snapshots.loc[np.isclose(all_snapshots["lambda"], 1.0)]
        if not endpoint.empty:
            selected_spectra.append(endpoint.iloc[-1])
    selected_spectra = selected_spectra[:3]
    selected_values = spectra.loc[spectra["run_id"].isin([row["run_id"] for row in selected_spectra])]
    x_limit = max(1.0, float(np.nanmax(np.abs(selected_values["E_real"]))) * 1.08) if not selected_values.empty else 1.0
    y_limit = max(0.12, float(np.nanmax(np.abs(selected_values["E_imag"]))) * 1.18) if not selected_values.empty else 0.12
    spectrum_colours = (COLORS[0], COLORS[2], COLORS[3])
    for index, axis in enumerate(spectrum_axes):
        if index >= len(selected_spectra):
            axis.set_visible(False)
            continue
        row = selected_spectra[index]
        eigenvalues = spectra.loc[spectra["run_id"] == row["run_id"]]
        colour = spectrum_colours[index]
        axis.axhline(0.0, color="0.72", lw=0.65, zorder=0)
        axis.scatter(eigenvalues["E_real"], eigenvalues["E_imag"], color=colour, s=8.5, linewidths=0, zorder=2)
        spectrum_label = _snapshot_label(row) if index != 1 else rf"$\chi\simeq{float(row['chi']):.2f}$"
        axis.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit), title=spectrum_label)
        axis.set_title(spectrum_label, pad=2.0)
        if index == 0:
            axis.set_ylabel(r"$\operatorname{Im}E/t$")
        else:
            axis.tick_params(labelleft=False)
        axis.tick_params(pad=1.5)
    spectrum_axes[1].set_xlabel(r"$\operatorname{Re}E/t$")

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
        Patch(facecolor=COLORS[0], edgecolor="0.25", label="NH PBC"),
        Patch(facecolor="0.90", edgecolor="0.25", hatch="///", label="matched Hermitian"),
    ]
    category_offset = 0.17
    for axis, nh_values, hermitian_values, ylabel in (
        (pair_control_axis, endpoint_pairs, control_pairs, r"$P_{\rm bulk}/t^2$"),
        (gamma_control_axis, endpoint_gamma, control_gamma, r"$\Gamma_{\max}/t$"),
    ):
        axis.barh(locations - category_offset, nh_values, height=0.29, color=COLORS[0], edgecolor="0.25", linewidth=0.55)
        axis.barh(locations + category_offset, hermitian_values, height=0.29, color="0.90", edgecolor="0.25", hatch="///", linewidth=0.55)
        axis.set(yticks=locations, yticklabels=[r"$L=24$", r"$L=40$"], xlabel=ylabel, ylim=(1.55, -0.55))
        axis.grid(axis="x", color="0.93", linewidth=0.45)
        axis.tick_params(axis="y", left=False, right=False)
    pair_control_axis.set_xlim(0.0, 0.145)
    pair_control_axis.set_xticks([0.0, 0.05, 0.10])
    gamma_control_axis.set_xlim(0.0, 0.115)
    gamma_control_axis.set_xticks([0, 0.05, 0.10])
    # Zero-length bars are marked explicitly, without inventing a visible height.
    for y, value in zip(locations + category_offset, control_gamma):
        if np.isclose(value, 0, atol=1e-12):
            gamma_control_axis.plot(0, y, marker="|", color="0.25", ms=6, clip_on=False)
            gamma_control_axis.annotate("0", (0, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    control_header.legend(
        handles=control_handles,
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
        fontsize=7.0,
    )

    _panel_label(trajectory_axis, "(a)", x=-0.15)
    _panel_label(profile_axis, "(b)", x=-0.15)
    _panel_label(spectrum_axes[0], "(c)", x=-0.42, y=1.06)
    _panel_label(control_header, "(d)", x=-0.15, y=1.06)
    _save(figure, output, "fig04_pbc_endpoint")


def figure_s1(data: Path, output: Path) -> None:
    """Supplementary Fig. S1: conditioning and covariance accuracy."""

    conditioning = _read(data / "conditioning.csv")
    validation = _read(data / "fig2.csv")
    raw = conditioning.loc[conditioning["kind"] == "raw"].sort_values("q")
    rescaled = conditioning.loc[conditioning["kind"] == "rescaled"].sort_values("q")
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(3.35, 5.9),
        layout="constrained",
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.94)},
    )
    condition_axis, error_axis, validation_axis = axes

    representative = raw if not raw.empty else rescaled
    if not representative.empty:
        condition_axis.semilogy(representative["q"], representative["similarity_condition"], color=COLORS[0], marker="o", markerfacecolor="white", label=r"$\kappa(\mathcal{V})$")
        condition_axis.semilogy(representative["q"], representative["right_condition"], color=COLORS[1], linestyle="--", marker="s", markerfacecolor="white", label=r"$\kappa(R)$")
    condition_axis.set(xlabel=r"$q=g(L-1)$", ylabel="condition number")
    condition_axis.grid(which="major", color="0.90", lw=0.55)
    condition_axis.legend(loc="upper left", frameon=False, handlelength=1.55)

    error_source = raw if not raw.empty else rescaled
    if not error_source.empty:
        error_axis.plot(
            error_source["q"], error_source["covariance_error"], color=COLORS[0], linestyle="-", marker="o",
            markerfacecolor="white", label="direct / rescaled\n(indistinguishable)",
        )
    error_axis.set(xlabel=r"$q=g(L-1)$", ylabel="relative covariance error")
    error_axis.set_ylim(0, 5.3e-11)
    error_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    error_axis.grid(which="major", color="0.90", lw=0.55)
    _legend_above(error_axis, ncol=1)

    _plot_validation_summary(validation_axis, validation)
    for axis in (condition_axis, error_axis):
        axis.set_xticks([0, 4, 8, 12, 16])

    _panel_label(condition_axis, "(a)", x=-0.18)
    _panel_label(error_axis, "(b)", x=-0.18)
    _panel_label(validation_axis, "(c)", x=-0.18)
    _save(figure, output, "figS1_conditioning")


def figure_s2(data: Path, output: Path) -> None:
    """Supplementary Fig. S2: local invariance and directional Green covariance."""

    frequency = _read(data / "green_frequency.csv")
    figure, axes = plt.subplots(3, 1, figsize=(3.35, 5.9), sharex=True, layout="constrained")
    ldos_axis, directional_axis, stripped_axis = axes

    q_handles = []
    for index, (q, source) in enumerate(frequency.groupby("q", sort=True)):
        colour = COLORS[index]
        source = source.sort_values("omega")
        marker_every = _sparse_every(len(source), 10)
        marker_start = index * marker_every // 3
        ldos_axis.plot(
            source["omega"], source["center_ldos"], color=colour, lw=1.35,
            marker=MARKERS[index], markevery=(marker_start, marker_every), markerfacecolor="white",
            markeredgewidth=0.8, ms=3.4,
        )
        directional_axis.semilogy(source["omega"], source["G_1L_abs"], color=colour, linestyle="-", lw=1.15)
        directional_axis.semilogy(source["omega"], source["G_L1_abs"], color=colour, linestyle="--", lw=1.15)
        stripped_axis.semilogy(
            source["omega"], source["stripped_1L_abs"], color=colour, linestyle="-", lw=1.15,
            marker=MARKERS[index], markevery=(marker_start, marker_every), markerfacecolor="white",
            markeredgewidth=0.8, ms=3.4,
        )
        # |G_{L1}| has the opposite known OBC factor from |G_{1L}|.
        stripped_reverse = source["G_L1_abs"].to_numpy(float) * np.exp(-float(q))
        stripped_axis.semilogy(source["omega"], stripped_reverse, color=colour, linestyle="--", lw=1.15)
        q_handles.append(Line2D([], [], color=colour, marker=MARKERS[index], markerfacecolor="white", lw=1.5, label=rf"$q={float(q):g}$"))
    direction_handles = [
        Line2D([], [], color="0.15", linestyle="-", lw=1.4, label=r"$1\!\to\!L$"),
        Line2D([], [], color="0.15", linestyle="--", lw=1.4, label=r"$L\!\to\!1$"),
    ]
    # Matplotlib fills multi-column legends column-wise.  Interleave the
    # handles so the visible first row contains q=0,2,6 and the second row
    # contains the two direction styles.
    legend_handles = [q_handles[0], direction_handles[0], q_handles[1], direction_handles[1], q_handles[2]]
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.25,
    )

    ldos_axis.set_ylabel(r"$t\,\rho_{\rm center}(\omega)$")
    directional_axis.set_ylabel(r"$t\,|G^{pp}_{ij}(\omega)|$")
    stripped_axis.set(xlabel=r"$\omega/t$", ylabel=r"$t\times$ stripped $|G^{pp}_{ij}(\omega)|$")
    # Show the spectral window; the processed table retains the full tails.
    stripped_axis.set_xlim(-2.7, 2.7)
    directional_axis.set_ylim(1e-10, 1e4)
    stripped_axis.set_ylim(1e-8, 1e2)
    for axis in axes:
        axis.grid(which="major", color="0.91", lw=0.5)
    ldos_axis.tick_params(labelbottom=False)
    directional_axis.tick_params(labelbottom=False)
    _panel_label(ldos_axis, "(a)", x=-0.18)
    _panel_label(directional_axis, "(b)", x=-0.18)
    _panel_label(stripped_axis, "(c)", x=-0.18)
    _save(figure, output, "figS2_green_covariance")


S3_QUANTITIES = {
    "s_min": (r"$s_{\min}$", False),
    "minimum_eigenvalue_separation": (r"$d_{\min}/t$", True),
    "right_condition": (r"$\kappa(R)$", True),
    "field_residual": ("SCF field residual", True),
}


def _plot_s3_quantity(axis: plt.Axes, branch: pd.DataFrame, column: str) -> None:
    ylabel, logarithmic_y = S3_QUANTITIES[column]
    for length, g in BRANCH_KEYS:
        source = _branch_source(branch, length, g)
        if source.empty:
            continue
        finite = np.isfinite(source[column])
        if logarithmic_y:
            finite &= source[column] > 0.0
        valid = source.loc[finite]
        colour, marker, linestyle = _branch_style(length, g)
        axis.plot(
            valid["lambda"],
            valid[column],
            color=colour,
            linestyle=linestyle,
            marker=marker,
            markevery=_sparse_every(len(valid), 10),
            markerfacecolor="white",
            markeredgewidth=0.8,
            ms=4.0,
        )
    axis.set(xscale="log", xlabel=r"$\lambda$", ylabel=ylabel)
    if logarithmic_y:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=5))
        axis.yaxis.set_minor_formatter(NullFormatter())
    if column == "minimum_eigenvalue_separation":
        axis.set_yticks([0.002, 0.01, 0.05], [r"$2\times10^{-3}$", r"$10^{-2}$", r"$5\times10^{-2}$"])
    axis.grid(which="major", color="0.90", lw=0.5)


def _decorate_s3_threshold(axis: plt.Axes, column: str) -> None:
    if column == "s_min":
        axis.axhline(0.70, color="0.38", linestyle="--", lw=0.85)
        axis.text(0.03, 0.23, r"$s_{\min}\geq0.70$", transform=axis.transAxes, ha="left", va="bottom", fontsize=7.5)
        axis.set_ylim(0.66, 1.03)
    elif column == "field_residual":
        axis.axhline(1.0e-10, color="0.38", linestyle="--", lw=0.85)
        axis.text(0.03, 0.10, r"tolerance $10^{-10}$", transform=axis.transAxes, ha="left", va="bottom", fontsize=7.5)
        axis.set_ylim(top=1.3e-10)


def _make_s3_split(
    branch: pd.DataFrame,
    output: Path,
    stem: str,
    columns: tuple[str, str],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 4.5), layout="constrained")
    for axis, column, label in zip(axes, columns, ("(a)", "(b)")):
        _plot_s3_quantity(axis, branch, column)
        _decorate_s3_threshold(axis, column)
        _panel_label(axis, label, x=-0.18)
    figure.legend(
        handles=_branch_handles(),
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=1.75,
        handletextpad=0.35,
        columnspacing=0.85,
        labelspacing=0.2,
    )
    _save(figure, output, stem)


def figure_s3(data: Path, output: Path) -> None:
    """Supplementary Fig. S3: accepted canonical branch diagnostics only."""

    branch = _accepted(_read(data / "fig3.csv"))
    columns = (
        "s_min",
        "minimum_eigenvalue_separation",
        "right_condition",
        "field_residual",
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.45), layout="constrained")
    axes = axes.ravel()
    for axis, column, label in zip(axes, columns, ("(a)", "(b)", "(c)", "(d)")):
        _plot_s3_quantity(axis, branch, column)
        _decorate_s3_threshold(axis, column)
        _panel_label(axis, label, x=-0.15)
    figure.legend(
        handles=_branch_handles(),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    _save(figure, output, "figS3_branch_quality")

    _make_s3_split(
        branch,
        output,
        "figS3a_acceptance_convergence",
        ("s_min", "field_residual"),
    )
    _make_s3_split(
        branch,
        output,
        "figS3b_spectral_conditioning",
        ("minimum_eigenvalue_separation", "right_condition"),
    )


def figure_s4(data: Path, output: Path) -> None:
    """Supplementary Fig. S4: fixed-filling and projector audit."""

    audit = _read(data / "branch_audit.csv")
    forward = _accepted(audit)
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), layout="constrained")
    filling_axis, separation_axis, algebra_axis, reverse_axis = axes.ravel()

    filling_data = _read(data / "filling_audit.csv")
    filling = filling_data.loc[
        (filling_data["L"] == 40) & np.isclose(filling_data["g"], 0.05)
    ].copy()
    filling_styles = {
        "obc": (COLORS[0], "o", "-", "OBC"),
        "middle": (COLORS[1], "s", "--", "crossover"),
        "pbc": (COLORS[2], "^", "-.", "PBC"),
    }
    for location, (colour, marker, linestyle, label) in filling_styles.items():
        source = filling.loc[filling["audit_location"] == location].sort_values("mu")
        filling_axis.plot(
            source["mu"], source["achieved_filling"] - 0.8, color=colour, marker=marker,
            linestyle=linestyle, lw=1.4, markerfacecolor="white", markeredgewidth=0.9,
            label=label,
        )
    filling_axis.axhline(0.0, color="0.35", linestyle=":", lw=0.9)
    filling_axis.set(xlabel=r"$\mu/t$", ylabel=r"$N(\mu)/L-n_{\rm target}$")
    _legend_above(filling_axis, ncol=3)

    audit_keys = ((20, 0.10), (24, 0.05), (24, 0.10), (40, 0.05), (40, 0.10))
    audit_colours = ("#6A3D9A", *COLORS)
    for (length, g), colour, marker, linestyle in zip(audit_keys, audit_colours, ("v", "o", "s", "^", "D"), ("-", "--", "-.", ":", "-")):
        source = _unique_lambda(forward.loc[(forward["L"] == length) & np.isclose(forward["g"], g) & (forward["lambda"] > 0.0)])
        separation_axis.semilogx(
            source["lambda"], source["real_line_gap"], color=colour, marker=marker,
            linestyle=linestyle, markevery=_sparse_every(len(source), 8), markerfacecolor="white",
            markeredgewidth=0.9, ms=4.2, lw=1.35, label=rf"$L={length},\ g={g:.2f}$",
        )
    separation_axis.set(
        xlabel=r"$\lambda$", ylabel=r"real line gap $\Delta_{\mathrm{R}}/t$",
        xlim=(1.0e-4, 1.05), ylim=(0.345, 0.395),
    )
    separation_axis.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=2,
        handlelength=1.45, handletextpad=0.3, columnspacing=0.65, fontsize=6.35,
        borderaxespad=0.0,
    )

    representative = _unique_lambda(
        forward.loc[(forward["L"] == 40) & np.isclose(forward["g"], 0.05) & (forward["lambda"] > 0.0)]
    )
    for column, colour, marker, label in (
        ("projector_idempotency", COLORS[0], "o", r"$\|C^2-C\|/\|C\|$"),
        ("biorthogonality_error", COLORS[1], "s", r"$\|L^\dagger R-I\|/\|I\|$"),
        ("projector_trace_error", COLORS[2], "^", r"$|\operatorname{Tr}C-L|$"),
    ):
        source = representative.copy()
        display_values = np.maximum(source[column].to_numpy(float), 1.0e-16)
        algebra_axis.loglog(
            source["lambda"], display_values, color=colour, marker=marker,
            markerfacecolor="white", markeredgewidth=0.8, markevery=_sparse_every(len(source), 8), label=label,
        )
    algebra_axis.axhline(1.0e-8, color="0.35", linestyle=":", lw=0.9, label="acceptance bound")
    algebra_axis.set(xlabel=r"$\lambda$", ylabel="projector residual", ylim=(5.0e-17, 3.0e-7))
    _legend_above(algebra_axis, ncol=2)

    returned = audit.loc[audit["kind"] == "reverse_obc_endpoint"].sort_values(["L", "g"])
    labels = [rf"${int(row.L)},{float(row.g):.2f}$" for row in returned.itertuples()]
    positions = np.arange(len(returned), dtype=float)
    offsets = (-1.5, -0.5, 0.5, 1.5)
    bar_width = 0.18
    plotting_floor = 1.0e-10
    error_columns = (
        ("projector_error_to_endpoint", COLORS[0], "o", "projector"),
        ("density_error_to_endpoint", COLORS[1], "s", "density"),
        ("pair_product_error_to_endpoint", COLORS[2], "^", "pair product"),
        ("spectrum_error_to_endpoint", COLORS[3], "D", "spectrum"),
    )
    hatches = ("", "//", "xx", "..")
    for offset, (column, colour, _marker, label), hatch in zip(offsets, error_columns, hatches):
        values = np.maximum(returned[column].to_numpy(float), plotting_floor)
        reverse_axis.bar(
            positions + offset * bar_width, values - plotting_floor, width=bar_width,
            bottom=plotting_floor, color=colour, edgecolor="0.18", linewidth=0.55,
            hatch=hatch, label=label, zorder=2,
        )
    reverse_axis.set_yscale("log")
    reverse_axis.axhline(1.0e-7, color="0.35", linestyle=":", lw=0.9)
    reverse_axis.set(
        xticks=positions,
        xticklabels=labels,
        xlabel=r"$(L,g)$",
        ylabel=r"PBC$\to$OBC return error",
        ylim=(1.0e-10, 3.0e-7),
    )
    reverse_axis.grid(axis="y", which="major", color="0.90", lw=0.5, zorder=0)
    reverse_axis.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=2,
        handlelength=1.35, handletextpad=0.25, columnspacing=0.55, fontsize=6.35,
        borderaxespad=0.0,
    )

    for axis, label in zip(axes.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        _panel_label(axis, label, x=-0.15)
    _save(figure, output, "figS4_projector_filling_audit")


def make(data: Path, output: Path, selected: str = "all") -> None:
    """Render one selected figure or the complete publication figure set."""

    functions = {
        "fig02": figure02,
        "fig03": figure03,
        "fig04": figure04,
        "figS1": figure_s1,
        "figS2": figure_s2,
        "figS3": figure_s3,
        "figS4": figure_s4,
    }
    for name, function in functions.items():
        if selected in {"all", name}:
            function(data, output)
