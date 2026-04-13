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

from dart_mlci.analysis import (
    compute_growth_stats,
    discover_cells_csvs,
    filter_cells_by_area,
    fit_exponential_growth,
    fit_logistic_growth,
    load_cells_data,
)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cell growth from processed experiment output.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path, help="Path to processed output directory")
    group.add_argument(
        "--config", type=Path, help="Path to folder config JSON (reads output_dir from it)"
    )

    parser.add_argument("--min-area", type=float, default=None, help="Min cell area in µm²")
    parser.add_argument("--max-area", type=float, default=None, help="Max cell area in µm²")
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
    parser.add_argument("--separate", action="store_true", help="One figure per ROI")
    parser.add_argument(
        "--model",
        choices=["logistic", "exponential"],
        default="logistic",
        help="Growth model for fitting (default: logistic)",
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
        datasets[label] = {"stats": stats, "df": df, "entry": entry}

    if not datasets:
        print("No datasets remain after filtering.", file=sys.stderr)
        sys.exit(1)

    return datasets


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
    fig_count, ax_count = plt.subplots(figsize=(8, 5))
    fig_area, ax_area = plt.subplots(figsize=(8, 5))

    for i, (label, data) in enumerate(datasets.items()):
        stats = data["stats"]
        color = OKABE_ITO[i % len(OKABE_ITO)]
        t = _timepoints(stats, args)

        ax_count.plot(t, stats["cell_count"], marker="o", color=color, label=label)
        ax_area.plot(t, stats["total_area_um2"], marker="s", color=color, label=label)

        if args.fit:
            _overlay_fit(ax_count, t, stats["cell_count"].values, color, label, args)
            _overlay_fit(ax_area, t, stats["total_area_um2"].values, color, label, args)

    ax_count.set_xlabel(x_label)
    ax_count.set_ylabel("Cell count")
    ax_count.legend()
    ax_count.set_title("Cell Count Over Time")
    fig_count.tight_layout()

    ax_area.set_xlabel(x_label)
    ax_area.set_ylabel("Total single-cell area (µm²)")
    ax_area.legend()
    ax_area.set_title("Total Single-Cell Area Over Time")
    fig_area.tight_layout()

    fmt = args.format
    fig_count.savefig(save_dir / f"cell_count.{fmt}", dpi=300)
    fig_area.savefig(save_dir / f"cell_area.{fmt}", dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close(fig_count)
        plt.close(fig_area)


def _plot_single(label, data, args, save_dir, x_label, color_idx=0):
    stats = data["stats"]
    # For separate figures, always use blue (first color) for best readability
    color = OKABE_ITO[0]
    t = _timepoints(stats, args)

    fig, (ax_count, ax_area) = plt.subplots(1, 2, figsize=(12, 5))

    ax_count.plot(t, stats["cell_count"], marker="o", color=color)
    ax_count.set_xlabel(x_label)
    ax_count.set_ylabel("Cell count")
    ax_count.set_title(f"{label} — Cell Count")

    if args.fit:
        _overlay_fit(ax_count, t, stats["cell_count"].values, color, label, args)

    ax_area.plot(t, stats["total_area_um2"], marker="s", color=color)
    ax_area.set_xlabel(x_label)
    ax_area.set_ylabel("Total single-cell area (µm²)")
    ax_area.set_title(f"{label} — Total Single-Cell Area")

    if args.fit:
        _overlay_fit(ax_area, t, stats["total_area_um2"].values, color, label, args)

    fig.tight_layout()
    safe_label = label.replace("/", "_").replace(" ", "_")
    fig.savefig(save_dir / f"{safe_label}_growth.{args.format}", dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def _overlay_fit(ax, t, counts, color, label, args):
    """Fit growth model and overlay on axis."""
    fit_func = fit_logistic_growth if args.model == "logistic" else fit_exponential_growth
    try:
        result = fit_func(t, counts)
    except ValueError:
        return

    ax.plot(t, result.fitted_values, linestyle="--", color=color, alpha=0.7)

    rate_unit = "/min" if args.time_interval else "/tp"
    dt_unit = "min" if args.time_interval else "tp"
    rate_label = "r" if args.model == "logistic" else "λ"
    annotation = (
        f"{rate_label}={result.growth_rate:.4f}{rate_unit}\n"
        f"t₂={result.doubling_time:.1f} {dt_unit}\n"
        f"R²={result.r_squared:.3f}"
    )
    if args.model == "logistic":
        annotation += f"\nK={result.carrying_capacity:.1f}"
    ax.annotate(
        annotation,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        verticalalignment="top",
        fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


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

    fit_func = fit_logistic_growth if args.model == "logistic" else fit_exponential_growth
    for label, data in datasets.items():
        stats = data["stats"]
        t = _timepoints(stats, args)
        line = f"{label:<30} {len(stats):>10} {stats['cell_count'].max():>10} {stats['total_area_um2'].max():>12.1f}"

        if args.fit:
            try:
                result = fit_func(t, stats["cell_count"].values)
                line += f" {result.growth_rate:>10.4f} {result.doubling_time:>10.1f} {result.r_squared:>8.3f}"
                if args.model == "logistic":
                    line += f" {result.carrying_capacity:>10.1f}"
            except ValueError:
                line += f" {'N/A':>10} {'N/A':>10} {'N/A':>8}"
                if args.model == "logistic":
                    line += f" {'N/A':>10}"

        print(line)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = _resolve_output_dir(args)
    save_dir = args.save_dir or output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_all(output_dir, args)
    _make_figures(datasets, args, save_dir)
    _print_summary(datasets, args)


if __name__ == "__main__":
    main()
