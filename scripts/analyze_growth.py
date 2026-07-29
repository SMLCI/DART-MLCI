#!/usr/bin/env python
"""Analyze cell growth from processed experiment output.

Reads ``cells.csv`` files produced by ``process_folder.py``, computes
per-timepoint statistics, optionally fits exponential growth, and produces
publication-quality matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dart_mlci.analysis import (
    compute_growth_stats,
    discover_cells_csvs,
    filter_cells_by_area,
    fit_exponential_growth,
    fit_logistic_growth,
    load_cells_data,
    occupancy_cutoff_timepoint,
    summarize_by_group,
    truncate_at_cutoff,
)
from dart_mlci.chip import chamber_area_um2, load_chip_config
from dart_mlci.constants import CHAMBER_TYPE_NUMBERS

# Okabe-Ito colorblind-safe palette (no red+green)
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # pink
    "#009E73",  # teal
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#000000",  # black
]

LINE_STYLES = ["-", "--", "-.", ":"]

# Tripled vs matplotlib defaults so figures stay legible when shrunk into papers.
FS_TICK = 30
FS_LABEL = 30
FS_TITLE = 30
FS_LEGEND = 30
FS_ANNOTATION = 24

# Sizing for the growth-rate/doubling-time summary figure specifically. Unlike
# the per-ROI figures above (drawn oversized, then shrunk a lot when placed),
# this one is meant to be dropped in at close to its final size — e.g. a
# two-panel figure spanning a DIN A4 LaTeX page's text width (~6.3in with
# standard 1in margins) — so fonts are normal document scale, not tripled.
SUMMARY_FIGSIZE = (6.3, 3.0)
SUMMARY_FS_TICK = 8
SUMMARY_FS_LABEL = 9
SUMMARY_FS_TITLE = 10
SUMMARY_FS_ANNOTATION = 6.5

# Sizing for per-ROI growth-curve figures (cell count + area subplots). Width
# is fixed at half of SUMMARY_FIGSIZE's width, so each subplot is exactly 1/4
# of the ~6.3in LaTeX text width (two such figures side by side fill one page
# row). Height is chosen independently (no width-equivalent constraint) to
# leave room for a "Model parameters" footer below the axes, since there's no
# per-subplot title to repeat the y-axis label. Font sizes are picked for
# absolute print legibility, not scaled proportionally from SUMMARY_FS_* — the
# panel is narrower, but the text still needs to be readable at final print
# size.
PER_ROI_FIGSIZE = (SUMMARY_FIGSIZE[0] / 2, 1.8)  # (3.15, 1.8)
PER_ROI_FS_TICK = 4.5
PER_ROI_FS_LABEL = 5.25
PER_ROI_FS_TITLE = 5.625
PER_ROI_FS_ANNOTATION = 4.125

# Standalone single-panel exports (one chart per metric, no parameter footer).
SINGLE_PANEL_FIGSIZE = (PER_ROI_FIGSIZE[0] / 2, 1.5)  # (1.575, 1.5)

# Print-quality DPI for all saved raster figures — 300 looked soft/pixelated
# at these small physical sizes once actually printed/zoomed into.
SAVE_DPI = 600


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cell growth from processed experiment output.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path, help="Path to processed output directory")
    group.add_argument(
        "--config", type=Path, help="Path to folder config JSON (reads output_dir from it)"
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=0.75,
        help="Min cell area in µm² (default: 0.75, filters segmentation debris)",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=8.0,
        help="Max cell area in µm² (default: 8.0, filters fused-blob artifacts)",
    )
    parser.add_argument("--fit", action="store_true", help="Fit exponential growth curve")
    parser.add_argument(
        "--time-interval",
        type=float,
        default=None,
        help="Minutes per timepoint (converts x-axis to real time)",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for saved figures")
    parser.add_argument(
        "--format", choices=["pdf", "svg", "png"], default="pdf", help="Figure format"
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--folders", nargs="+", default=None, help="Filter to specific subfolders")
    parser.add_argument(
        "--exclude-chamber-types",
        nargs="+",
        default=None,
        help=(
            "Chamber-type folder names to exclude entirely (e.g. 'Mother Machines'), "
            "for designs unsuited to growth-rate fitting (e.g. cells leave immediately)."
        ),
    )
    parser.add_argument("--separate", action="store_true", help="One figure per ROI")
    parser.add_argument(
        "--model",
        choices=["logistic", "exponential"],
        default="logistic",
        help="Growth model for fitting (default: logistic)",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=None,
        help=(
            "Stop growth-rate fits once occupied chamber-area fraction "
            "(total_area_um2 / chamber_area_um2) reaches this value, e.g. 0.7. "
            "Requires --config or --chamber-config to resolve chip_config + folder mapping."
        ),
    )
    parser.add_argument(
        "--chamber-config",
        type=Path,
        default=None,
        help=(
            "Folder-config JSON providing 'chip_config' and 'folders' (chamber-type mapping), "
            "used with --occupancy-threshold. Defaults to --config if not given."
        ),
    )

    return parser.parse_args(argv)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    with open(args.config) as f:
        cfg = json.load(f)
    return Path(cfg["output_dir"])


def _load_all(output_dir: Path, args: argparse.Namespace) -> dict[str, dict]:
    """Load and filter all cells.csv files, return {label: {"stats": ..., "df": ...}}."""
    entries = discover_cells_csvs(output_dir)
    if not entries:
        print(f"No cells.csv files found under {output_dir}", file=sys.stderr)
        sys.exit(1)

    datasets: dict[str, dict] = {}
    for entry in entries:
        label = entry["folder"] or entry["path"].parent.name
        if args.folders and label not in args.folders:
            continue
        if args.exclude_chamber_types and entry["stack"] in args.exclude_chamber_types:
            continue
        try:
            df = load_cells_data(entry["path"])
        except ValueError as e:
            print(f"Skipping {entry['path']}: {e}", file=sys.stderr)
            continue

        df = filter_cells_by_area(df, min_area_um2=args.min_area, max_area_um2=args.max_area)
        if df.empty:
            print(f"Skipping {label}: all cells filtered out", file=sys.stderr)
            continue

        stats = compute_growth_stats(df)
        datasets[label] = {
            "stats": stats,
            "df": df,
            "entry": entry,
            "chamber_type": entry["stack"],
        }

    if not datasets:
        print("No datasets remain after filtering.", file=sys.stderr)
        sys.exit(1)

    return datasets


def _load_chamber_config(args: argparse.Namespace) -> dict | None:
    """Load the folder-config JSON (chip_config path + chamber folder mapping), if available.

    Reads ``--chamber-config``, falling back to ``--config``. Returns ``None`` if
    neither is given.
    """
    chamber_config_path = args.chamber_config or args.config
    if chamber_config_path is None:
        return None
    with open(chamber_config_path) as f:
        return json.load(f)


def _compute_cutoffs(datasets: dict[str, dict], args: argparse.Namespace) -> None:
    """Populate each dataset with an ``occupancy_cutoff`` timepoint (raw units, or None).

    Occupancy is only checked when ``--occupancy-threshold`` is given; it requires
    a chamber-config JSON (``--chamber-config`` or ``--config``) to map each
    dataset's chamber-type folder to a chip-config structure type and its area.
    """
    if args.occupancy_threshold is None:
        for data in datasets.values():
            data["occupancy_cutoff"] = None
        return

    cfg = _load_chamber_config(args)
    if cfg is None:
        print(
            "--occupancy-threshold requires --config or --chamber-config "
            "to resolve chip_config + chamber folder mapping",
            file=sys.stderr,
        )
        sys.exit(1)

    folders_map = cfg["folders"]
    chip_config = load_chip_config(cfg["chip_config"])

    for label, data in datasets.items():
        stack = data["entry"]["stack"]
        structure_type = folders_map.get(stack)
        if structure_type is None:
            print(
                f"Skipping occupancy cutoff for {label}: unknown chamber folder '{stack}'",
                file=sys.stderr,
            )
            data["occupancy_cutoff"] = None
            continue
        area = chamber_area_um2(chip_config, structure_type)
        data["occupancy_cutoff"] = occupancy_cutoff_timepoint(
            data["stats"], area, threshold=args.occupancy_threshold
        )


def _make_figures(
    datasets: dict[str, dict],
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    """Create cell count and cell area figures."""
    x_label = "Time (min)" if args.time_interval else "Timepoint"

    if args.separate:
        for i, (label, data) in enumerate(datasets.items()):
            _plot_single(label, data, args, save_dir, x_label, color_idx=i)
    else:
        _plot_combined(datasets, args, save_dir, x_label)


def _timepoints(stats, args):
    t = stats["timepoint"].values.astype(float)
    if args.time_interval:
        t = t * args.time_interval
    return t


def _plot_combined(datasets, args, save_dir, x_label):
    fig_count, ax_count = plt.subplots(figsize=(16, 10))
    fig_area, ax_area = plt.subplots(figsize=(16, 10))

    for i, (label, data) in enumerate(datasets.items()):
        stats = data["stats"]
        color = OKABE_ITO[i % len(OKABE_ITO)]
        t = _timepoints(stats, args)

        ax_count.plot(t, stats["cell_count"], marker="o", color=color, label=label)
        ax_area.plot(t, stats["total_area_um2"], marker="s", color=color, label=label)

        if args.fit:
            cutoff = data.get("occupancy_cutoff")
            _overlay_fit(
                ax_count,
                t,
                stats["cell_count"].values,
                color,
                label,
                args,
                cutoff=cutoff,
                marker="o",
                use_legend=False,
            )
            _overlay_fit(
                ax_area,
                t,
                stats["total_area_um2"].values,
                color,
                label,
                args,
                cutoff=cutoff,
                marker="s",
                use_legend=False,
            )

    ax_count.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_count.set_ylabel("Cell count", fontsize=FS_LABEL)
    ax_count.tick_params(axis="both", labelsize=FS_TICK)
    ax_count.legend(fontsize=FS_LEGEND)
    ax_count.set_title("Cell Count Over Time", fontsize=FS_TITLE)
    fig_count.tight_layout()

    ax_area.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_area.set_ylabel("Total single-cell area (µm²)", fontsize=FS_LABEL)
    ax_area.tick_params(axis="both", labelsize=FS_TICK)
    ax_area.legend(fontsize=FS_LEGEND)
    ax_area.set_title("Total Single-Cell Area Over Time", fontsize=FS_TITLE)
    fig_area.tight_layout()

    fmt = args.format
    fig_count.savefig(save_dir / f"cell_count.{fmt}", dpi=SAVE_DPI)
    fig_area.savefig(save_dir / f"cell_area.{fmt}", dpi=SAVE_DPI)

    if args.show:
        plt.show()
    else:
        plt.close(fig_count)
        plt.close(fig_area)


def _draw_growth_panel(ax, t, values, marker, ylabel, x_label, color, fit_color, args, cutoff):
    """Draw one growth-curve panel (measurement markers + optional fit line/legend) onto *ax*.

    Shared by the combined per-ROI figure and the standalone single-panel
    exports, so panel styling only lives in one place. Returns the fit's
    2-column parameter-grid text (or ``None`` if unfitted/fit failed) — the
    caller decides whether and where to display it.
    """
    ax.plot(
        t,
        values,
        marker=marker,
        linestyle="none",
        color=color,
        markersize=3,
        markeredgewidth=0.4,
        markeredgecolor="black",
    )
    ax.set_xlabel(x_label, fontsize=PER_ROI_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=PER_ROI_FS_LABEL)
    ax.tick_params(axis="both", labelsize=PER_ROI_FS_TICK, width=0.6, length=2.5)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    if not args.fit:
        return None
    return _overlay_fit(
        ax,
        t,
        values,
        color,
        "",
        args,
        fit_color=fit_color,
        annotation_fontsize=PER_ROI_FS_ANNOTATION,
        cutoff=cutoff,
        marker=marker,
        fit_linewidth=1.0,
    )


def _save_single_panel_png(
    t, values, marker, ylabel, x_label, color, fit_color, args, cutoff, save_path
):
    """Save one standalone growth-curve panel as its own PNG — measurement + fit line/legend,
    but no parameter-grid text (that only appears in the combined per-ROI figure's footer).
    """
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    _draw_growth_panel(ax, t, values, marker, ylabel, x_label, color, fit_color, args, cutoff)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=SAVE_DPI)
    plt.close(fig)


def _plot_single(label, data, args, save_dir, x_label, color_idx=0):
    stats = data["stats"]
    # For separate figures, always use blue (first color) for best readability
    color = OKABE_ITO[0]
    fit_color = "#000000"  # black fit line, distinct from blue measurements
    t = _timepoints(stats, args)
    cutoff = data.get("occupancy_cutoff")
    safe_label = label.replace("/", "_").replace(" ", "_")
    count_values = stats["cell_count"].values
    area_values = stats["total_area_um2"].values

    # No per-subplot title — it would just repeat the y-axis label at this
    # panel size; the y-axis label alone already says what's plotted.
    fig, (ax_count, ax_area) = plt.subplots(1, 2, figsize=PER_ROI_FIGSIZE)
    count_params = _draw_growth_panel(
        ax_count, t, count_values, "o", "Cell count", x_label, color, fit_color, args, cutoff
    )
    area_params = _draw_growth_panel(
        ax_area,
        t,
        area_values,
        "s",
        "Total single-cell area (µm²)",
        x_label,
        color,
        fit_color,
        args,
        cutoff,
    )

    # Reserve a footer band below the axes for the fitted parameters, shared
    # by both subplots under one "Model parameters" heading — keeps the
    # numbers off the data view entirely, rather than overlaid on the axes.
    footer_frac = 0.26 if (count_params or area_params) else 0.02
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, footer_frac, 1, 0.97))

    if count_params or area_params:
        fig.text(
            0.5,
            footer_frac - 0.03,
            "Model parameters",
            ha="center",
            va="top",
            fontsize=PER_ROI_FS_TITLE,
            fontweight="bold",
        )
        if count_params:
            fig.text(
                0.27,
                footer_frac - 0.12,
                count_params,
                ha="center",
                va="top",
                fontsize=PER_ROI_FS_ANNOTATION,
                fontfamily="monospace",
            )
        if area_params:
            fig.text(
                0.77,
                footer_frac - 0.12,
                area_params,
                ha="center",
                va="top",
                fontsize=PER_ROI_FS_ANNOTATION,
                fontfamily="monospace",
            )

    fig.savefig(save_dir / f"{safe_label}_growth.{args.format}", dpi=SAVE_DPI)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Standalone per-metric exports (always PNG, no parameter footer).
    _save_single_panel_png(
        t,
        count_values,
        "o",
        "Cell count",
        x_label,
        color,
        fit_color,
        args,
        cutoff,
        save_dir / f"{safe_label}_cell_count.png",
    )
    _save_single_panel_png(
        t,
        area_values,
        "s",
        "Total single-cell area (µm²)",
        x_label,
        color,
        fit_color,
        args,
        cutoff,
        save_dir / f"{safe_label}_area.png",
    )


def _scaled_cutoff(cutoff, args):
    """Convert a raw-timepoint occupancy cutoff to the plot's time units."""
    if cutoff is None:
        return None
    return cutoff * args.time_interval if args.time_interval else cutoff


def _fit_series(t, values, cutoff, args):
    """Fit the configured growth model, truncating at *cutoff* (raw timepoint units) if given.

    Returns ``(fit_t, result)``; ``result`` is ``None`` if the fit fails.
    """
    fit_func = fit_logistic_growth if args.model == "logistic" else fit_exponential_growth
    fit_t, fit_values = truncate_at_cutoff(t, values, _scaled_cutoff(cutoff, args))
    try:
        return fit_t, fit_func(fit_t, fit_values)
    except ValueError:
        return fit_t, None


def _overlay_fit(
    ax,
    t,
    counts,
    color,
    label,
    args,
    fit_color=None,
    annotation_fontsize=FS_ANNOTATION,
    cutoff=None,
    marker="o",
    fit_linewidth=2.5,
    use_legend=True,
):
    """Fit growth model and overlay on axis.

    ``fit_color`` overrides the line color (defaults to ``color``); ``annotation_fontsize``
    overrides the annotation size. Both default to the original combined-plot behavior.
    ``cutoff``, if given, truncates the data fed into the fit (see :func:`_fit_series`) —
    the fit line's dashed segment simply stops there, with no extra vertical
    marker line (which would be redundant with that visible endpoint).

    When ``use_legend`` is True (single-chamber figures), a compact in-axes
    legend explains the marker/line (data vs. model fit) via ``marker``, which
    should match the marker actually used to plot the measurements on this
    axis; the fitted parameters (as a 2-column grid string) are returned
    rather than drawn, so the caller can place them outside the axes (see
    ``_plot_single``) instead of crowding the data view. When False (the
    multi-chamber combined plot, which already has its own per-chamber-color
    legend), the parameters are drawn in-axes as a plain text box instead, and
    ``None`` is returned.

    Returns the parameter-grid string (``use_legend=True``) or ``None``.
    """
    fit_t, result = _fit_series(t, counts, cutoff, args)
    if result is None:
        return

    ax.plot(
        fit_t,
        result.fitted_values,
        linestyle="--",
        color=fit_color or color,
        alpha=1.0,
        linewidth=fit_linewidth,
    )

    rate_unit = "/min" if args.time_interval else "/tp"
    dt_unit = "min" if args.time_interval else "tp"
    rate_label = "r" if args.model == "logistic" else "λ"

    if not use_legend:
        params = (
            f"{rate_label}={result.growth_rate:.4f}{rate_unit}, t₂={result.doubling_time:.1f} {dt_unit}\n"
            f"R²={result.r_squared:.3f}"
        )
        if args.model == "logistic":
            params += f", K={result.carrying_capacity:.1f}"
        ax.annotate(
            params,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=annotation_fontsize,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        return

    # Compact legend explaining what the marker/line mean — stacked (not
    # side-by-side) so it stays narrow enough not to overflow this panel's
    # width; the actual "two columns" the parameter values sit in a separate
    # grid below (kept free of legend text, which would collide/overflow).
    measurement_handle = Line2D(
        [],
        [],
        marker=marker,
        linestyle="none",
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.4,
        markersize=4,
    )
    fit_handle = Line2D([], [], linestyle="--", color=fit_color or color, linewidth=fit_linewidth)
    ax.legend(
        handles=[measurement_handle, fit_handle],
        labels=["Measurement", "Model fit"],
        loc="upper left",
        ncol=1,
        framealpha=0.8,
        facecolor="white",
        edgecolor="black",
        handlelength=1.0,
        borderpad=0.6,
        labelspacing=0.3,
        prop=dict(size=annotation_fontsize),
    )

    # Build the 2-column parameter grid as text, but don't draw it here —
    # the caller places it below the whole figure (see _plot_single), since
    # overlaying it on the axes at this panel size crowds out the data.
    left_col = [
        f"{rate_label}={result.growth_rate:.4f}{rate_unit}",
        f"t₂={result.doubling_time:.1f} {dt_unit}",
    ]
    right_col = [f"R²={result.r_squared:.3f}"]
    if args.model == "logistic":
        right_col.append(f"K={result.carrying_capacity:.3g}")
    col_width = max(len(entry) for entry in left_col) + 1
    rows = []
    for i in range(max(len(left_col), len(right_col))):
        left = left_col[i] if i < len(left_col) else ""
        right = right_col[i] if i < len(right_col) else ""
        rows.append(f"{left:<{col_width}}{right}")
    return "\n".join(rows)


def _print_summary(datasets, args):
    """Print a summary table to stdout."""
    rate_col = "r" if args.model == "logistic" else "λ"
    print(f"\n{'Label':<30} {'Timepoints':>10} {'Max Count':>10} {'Max Area µm²':>12}", end="")
    if args.fit:
        print(f" {rate_col:>10} {'t_double':>10} {'R²':>8}", end="")
        if args.model == "logistic":
            print(f" {'K':>10}", end="")
    print()
    fit_width = 28 + (10 if args.model == "logistic" else 0) if args.fit else 0
    print("-" * (72 + fit_width))

    for label, data in datasets.items():
        stats = data["stats"]
        t = _timepoints(stats, args)
        line = f"{label:<30} {len(stats):>10} {stats['cell_count'].max():>10} {stats['total_area_um2'].max():>12.1f}"

        if args.fit:
            _, result = _fit_series(
                t, stats["cell_count"].values, data.get("occupancy_cutoff"), args
            )
            if result is not None:
                line += f" {result.growth_rate:>10.4f} {result.doubling_time:>10.1f} {result.r_squared:>8.3f}"
                if args.model == "logistic":
                    line += f" {result.carrying_capacity:>10.1f}"
            else:
                line += f" {'N/A':>10} {'N/A':>10} {'N/A':>8}"
                if args.model == "logistic":
                    line += f" {'N/A':>10}"

        print(line)


def _collect_group_metrics(datasets, args):
    """Fit each dataset's cell-count curve and group growth rate / doubling time by chamber type.

    Returns ``(rates_by_type, doubling_by_type)``, each a ``{chamber_type: [values]}`` dict.
    Datasets whose fit fails are skipped; non-finite doubling times (r <= 0) are
    excluded from ``doubling_by_type`` only.
    """
    rates_by_type: dict[str, list[float]] = {}
    doubling_by_type: dict[str, list[float]] = {}
    for data in datasets.values():
        stats = data["stats"]
        t = _timepoints(stats, args)
        _, result = _fit_series(t, stats["cell_count"].values, data.get("occupancy_cutoff"), args)
        if result is None:
            continue
        chamber_type = data["chamber_type"]
        rates_by_type.setdefault(chamber_type, []).append(result.growth_rate)
        if np.isfinite(result.doubling_time):
            doubling_by_type.setdefault(chamber_type, []).append(result.doubling_time)
    return rates_by_type, doubling_by_type


def _draw_group_bar_summary(ax, summaries, ylabel, title):
    """Draw a bar plot of group means with CI error bars and jittered replicate points onto *ax*.

    All bars share a single color — the x-axis labels already identify the
    chamber design, so per-bar color would be redundant. Sized for direct
    embedding at print scale (see ``SUMMARY_FIGSIZE``/``SUMMARY_FS_*``), not
    for the oversized-then-shrunk convention used by the per-ROI figures.
    """
    if not summaries:
        return

    groups = [s.group for s in summaries]
    means = [s.mean for s in summaries]
    err_low = [s.mean - s.ci_low for s in summaries]
    err_high = [s.ci_high - s.mean for s in summaries]

    x = np.arange(len(groups))
    ax.bar(
        x,
        means,
        yerr=[err_low, err_high],
        capsize=3,
        error_kw=dict(elinewidth=0.8, capthick=0.8),
        color=OKABE_ITO[0],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    rng = np.random.default_rng(0)
    for i, s in enumerate(summaries):
        jitter = rng.uniform(-0.15, 0.15, size=len(s.values))
        ax.scatter(
            np.full(len(s.values), x[i]) + jitter,
            s.values,
            color="black",
            s=8,
            linewidth=0,
            zorder=3,
            alpha=0.7,
        )
        ax.annotate(
            f"n={s.n}",
            xy=(x[i], max(s.values)),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=SUMMARY_FS_ANNOTATION,
        )

    # Headroom so the "n=" annotations above the tallest points aren't clipped.
    ax.margins(y=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0, ha="center", fontsize=SUMMARY_FS_TICK)
    ax.set_xlabel("RoI design on the SAK", fontsize=SUMMARY_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=SUMMARY_FS_LABEL)
    ax.tick_params(axis="both", labelsize=SUMMARY_FS_TICK, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=SUMMARY_FS_TITLE)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)


def _chamber_type_sort_key(chamber_type_label: str, folders_map: dict) -> tuple:
    """Sort key following the canonical 1-8 chamber-type numbering (see ``CHAMBER_TYPE_NUMBERS``).

    Unknown chamber types (no mapping, or not in the numbering) sort after all
    numbered ones, alphabetically.
    """
    structure_type = folders_map.get(chamber_type_label)
    number = CHAMBER_TYPE_NUMBERS.get(structure_type)
    if number is None:
        return (1, chamber_type_label)
    return (0, number)


def _number_group_labels(summaries, folders_map) -> None:
    """Replace each group's label with its canonical chamber-type number (e.g. 'Small Chambers' -> '1').

    Falls back to the original text for chamber types with no known number.
    """
    for s in summaries:
        structure_type = folders_map.get(s.group)
        number = CHAMBER_TYPE_NUMBERS.get(structure_type)
        if number is not None:
            s.group = str(number)


def _make_summary_bar_plots(datasets, args, save_dir):
    """Create a growth-rate + doubling-time summary figure (side by side), grouped by chamber type.

    Groups are ordered and numbered per the canonical 1-8 chamber-type numbering
    (matching ``process_folder.py``'s timing tables) when a chamber-config is
    available; otherwise groups keep discovery order with no number suffix.
    """
    rates_by_type, doubling_by_type = _collect_group_metrics(datasets, args)

    cfg = _load_chamber_config(args)
    folders_map = cfg.get("folders", {}) if cfg else {}

    rate_label = "r" if args.model == "logistic" else "λ"
    if args.time_interval:
        # growth_rate is per minute (t is scaled by --time-interval); report per hour instead.
        rates_by_type = {k: [v * 60 for v in vs] for k, vs in rates_by_type.items()}
        rate_unit = "1/h"
    else:
        rate_unit = "1/timepoint"
    rate_summaries = summarize_by_group(rates_by_type)
    rate_summaries.sort(key=lambda s: _chamber_type_sort_key(s.group, folders_map))
    _number_group_labels(rate_summaries, folders_map)

    dt_unit = "min" if args.time_interval else "timepoints"
    dt_summaries = summarize_by_group(doubling_by_type)
    dt_summaries.sort(key=lambda s: _chamber_type_sort_key(s.group, folders_map))
    _number_group_labels(dt_summaries, folders_map)

    if not rate_summaries and not dt_summaries:
        return

    fig, (ax_rate, ax_dt) = plt.subplots(1, 2, figsize=SUMMARY_FIGSIZE)
    _draw_group_bar_summary(
        ax_rate,
        rate_summaries,
        ylabel=f"Growth rate {rate_label} [{rate_unit}]",
        title="Growth rate (mean ± 95% CI)",
    )
    _draw_group_bar_summary(
        ax_dt,
        dt_summaries,
        ylabel=f"Doubling time [{dt_unit}]",
        title="Doubling time (mean ± 95% CI)",
    )
    fig.tight_layout(pad=0.6, w_pad=1.6)
    fig.savefig(save_dir / f"growth_summary.{args.format}", dpi=SAVE_DPI)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = _resolve_output_dir(args)
    save_dir = args.save_dir or output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_all(output_dir, args)
    _compute_cutoffs(datasets, args)
    _make_figures(datasets, args, save_dir)
    _print_summary(datasets, args)
    if args.fit:
        _make_summary_bar_plots(datasets, args, save_dir)


if __name__ == "__main__":
    main()
