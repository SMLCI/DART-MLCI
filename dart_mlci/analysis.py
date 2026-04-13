"""Cell growth analysis — load, filter, aggregate, and fit growth curves."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CELLS_CSV_REQUIRED_COLUMNS = {"timepoint", "cell_id", "area_px", "area_um2"}


def discover_cells_csvs(output_dir: Path) -> list[dict]:
    """Recursively find all ``cells.csv`` files under *output_dir*.

    Returns a list of dicts with keys ``path``, ``folder``, ``stack``.
    ``folder`` is the parent directory name and ``stack`` is the grandparent
    (or the same as ``folder`` when the CSV sits directly in *output_dir*).
    """
    output_dir = Path(output_dir)
    results: list[dict] = []
    for csv_path in sorted(output_dir.rglob("cells.csv")):
        rel = csv_path.relative_to(output_dir)
        parts = rel.parts[:-1]  # strip filename
        folder = parts[-1] if parts else ""
        stack = parts[-2] if len(parts) >= 2 else folder
        results.append({"path": csv_path, "folder": folder, "stack": stack})
    return results


def load_cells_data(path: Path) -> pd.DataFrame:
    """Read and validate a ``cells.csv`` file.

    Raises ``ValueError`` if required columns are missing or the file is empty.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = CELLS_CSV_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Empty cells.csv: {path}")
    return df


def filter_cells_by_area(
    df: pd.DataFrame,
    min_area_um2: float | None = None,
    max_area_um2: float | None = None,
) -> pd.DataFrame:
    """Filter cells by area bounds (inclusive)."""
    mask = pd.Series(True, index=df.index)
    if min_area_um2 is not None:
        mask &= df["area_um2"] >= min_area_um2
    if max_area_um2 is not None:
        mask &= df["area_um2"] <= max_area_um2
    return df.loc[mask].reset_index(drop=True)


def compute_growth_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-timepoint statistics.

    Returns a DataFrame with columns ``timepoint``, ``cell_count``,
    ``mean_area_um2``, ``total_area_um2``.
    """
    grouped = df.groupby("timepoint", sort=True)
    stats = grouped.agg(
        cell_count=("cell_id", "count"),
        mean_area_um2=("area_um2", "mean"),
        total_area_um2=("area_um2", "sum"),
    ).reset_index()
    return stats


@dataclasses.dataclass
class ExponentialFitResult:
    """Result of exponential growth fit."""

    n0: float
    """Estimated initial count (at t=0 of the provided data)."""
    growth_rate: float
    """Growth rate λ (per unit time)."""
    doubling_time: float
    """Doubling time = ln(2)/λ.  ``inf`` when λ ≤ 0."""
    r_squared: float
    """Coefficient of determination on the log-scale fit."""
    fitted_values: np.ndarray
    """Fitted counts at each input timepoint."""


def fit_exponential_growth(
    timepoints: np.ndarray | list,
    counts: np.ndarray | list,
) -> ExponentialFitResult:
    """Fit exponential growth via linear regression on log-transformed data.

    ``log(N) = log(N₀) + λ·t``

    Timepoints with count ≤ 0 are skipped.

    Raises ``ValueError`` if fewer than 2 valid data points remain.
    """
    t = np.asarray(timepoints, dtype=float)
    n = np.asarray(counts, dtype=float)

    valid = n > 0
    t_valid = t[valid]
    n_valid = n[valid]

    if len(t_valid) < 2:
        raise ValueError("Need at least 2 timepoints with positive counts for exponential fit")

    log_n = np.log(n_valid)

    # Linear fit: log(N) = intercept + slope * t
    slope, intercept = np.polyfit(t_valid, log_n, 1)

    growth_rate = slope
    n0 = np.exp(intercept)
    doubling_time = np.log(2) / growth_rate if growth_rate > 0 else float("inf")

    # R² on log scale
    log_n_pred = intercept + slope * t_valid
    ss_res = np.sum((log_n - log_n_pred) ** 2)
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Fitted values for all original timepoints
    fitted_values = n0 * np.exp(growth_rate * t)

    return ExponentialFitResult(
        n0=n0,
        growth_rate=growth_rate,
        doubling_time=doubling_time,
        r_squared=r_squared,
        fitted_values=fitted_values,
    )


@dataclasses.dataclass
class LogisticFitResult:
    """Result of logistic growth fit."""

    n0: float
    """Estimated initial population."""
    growth_rate: float
    """Intrinsic growth rate r."""
    doubling_time: float
    """Doubling time = ln(2)/r.  ``inf`` when r ≤ 0."""
    carrying_capacity: float
    """Carrying capacity K (the plateau)."""
    r_squared: float
    """Coefficient of determination on the original scale."""
    fitted_values: np.ndarray
    """Fitted counts at each input timepoint."""


def fit_logistic_growth(
    timepoints: np.ndarray | list,
    counts: np.ndarray | list,
) -> LogisticFitResult:
    """Fit logistic growth model: N(t) = K / (1 + ((K - N₀)/N₀) · exp(-r·t)).

    Timepoints with count ≤ 0 are skipped.

    Raises ``ValueError`` if fewer than 3 valid data points remain or if the
    optimiser fails to converge.
    """
    t = np.asarray(timepoints, dtype=float)
    n = np.asarray(counts, dtype=float)

    valid = n > 0
    t_valid = t[valid]
    n_valid = n[valid]

    if len(t_valid) < 3:
        raise ValueError("Need at least 3 timepoints with positive counts for logistic fit")

    def _logistic(t, n0, r, K):
        return K / (1.0 + ((K - n0) / n0) * np.exp(-r * t))

    # Initial guesses
    n0_guess = float(np.min(n_valid[: max(1, len(n_valid) // 4 + 1)]))
    K_guess = float(np.max(n_valid)) * 1.1

    # Estimate r from log-slope of first half
    half = max(2, len(t_valid) // 2)
    t_half = t_valid[:half]
    n_half = n_valid[:half]
    log_n_half = np.log(n_half)
    slope, _ = np.polyfit(t_half, log_n_half, 1)
    r_guess = max(slope, 0.01)

    try:
        popt, _ = curve_fit(
            _logistic,
            t_valid,
            n_valid,
            p0=[n0_guess, r_guess, K_guess],
            bounds=(0, np.inf),
            maxfev=10000,
        )
    except RuntimeError as e:
        raise ValueError(f"Logistic fit failed to converge: {e}") from e

    n0_fit, r_fit, K_fit = popt
    doubling_time = np.log(2) / r_fit if r_fit > 0 else float("inf")

    fitted_all = _logistic(t, n0_fit, r_fit, K_fit)

    # R² on original scale (using valid points only)
    fitted_valid = _logistic(t_valid, n0_fit, r_fit, K_fit)
    ss_res = np.sum((n_valid - fitted_valid) ** 2)
    ss_tot = np.sum((n_valid - np.mean(n_valid)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return LogisticFitResult(
        n0=n0_fit,
        growth_rate=r_fit,
        doubling_time=doubling_time,
        carrying_capacity=K_fit,
        r_squared=r_squared,
        fitted_values=fitted_all,
    )
