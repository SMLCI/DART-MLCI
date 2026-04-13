"""Tests for dart_mlci.analysis — cell growth analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dart_mlci.analysis import (
    ExponentialFitResult,
    LogisticFitResult,
    compute_growth_stats,
    discover_cells_csvs,
    filter_cells_by_area,
    fit_exponential_growth,
    fit_logistic_growth,
    load_cells_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cells_csv(path, rows=None):
    """Write a minimal cells.csv at *path*."""
    if rows is None:
        rows = [
            {"timepoint": 0, "cell_id": 1, "area_px": 100, "area_um2": 25.0},
            {"timepoint": 0, "cell_id": 2, "area_px": 120, "area_um2": 30.0},
            {"timepoint": 1, "cell_id": 1, "area_px": 110, "area_um2": 27.5},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# TestDiscoverCellsCsvs
# ---------------------------------------------------------------------------


class TestDiscoverCellsCsvs:
    def test_finds_nested_csvs(self, tmp_path):
        d = tmp_path / "stack1" / "roi1"
        d.mkdir(parents=True)
        _make_cells_csv(d / "cells.csv")
        result = discover_cells_csvs(tmp_path)
        assert len(result) == 1
        assert result[0]["path"] == d / "cells.csv"
        assert result[0]["folder"] == "roi1"
        assert result[0]["stack"] == "stack1"

    def test_finds_multiple(self, tmp_path):
        for name in ["a", "b"]:
            d = tmp_path / name
            d.mkdir()
            _make_cells_csv(d / "cells.csv")
        result = discover_cells_csvs(tmp_path)
        assert len(result) == 2

    def test_empty_directory(self, tmp_path):
        assert discover_cells_csvs(tmp_path) == []


# ---------------------------------------------------------------------------
# TestLoadCellsData
# ---------------------------------------------------------------------------


class TestLoadCellsData:
    def test_valid(self, tmp_path):
        p = tmp_path / "cells.csv"
        _make_cells_csv(p)
        df = load_cells_data(p)
        assert set(df.columns) >= {"timepoint", "cell_id", "area_px", "area_um2"}
        assert len(df) == 3

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "cells.csv"
        pd.DataFrame({"timepoint": [0], "cell_id": [1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_cells_data(p)

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "cells.csv"
        pd.DataFrame(columns=["timepoint", "cell_id", "area_px", "area_um2"]).to_csv(p, index=False)
        with pytest.raises(ValueError, match="Empty"):
            load_cells_data(p)

    def test_extra_columns_ok(self, tmp_path):
        p = tmp_path / "cells.csv"
        rows = [{"timepoint": 0, "cell_id": 1, "area_px": 50, "area_um2": 12.0, "extra": "x"}]
        pd.DataFrame(rows).to_csv(p, index=False)
        df = load_cells_data(p)
        assert "extra" in df.columns


# ---------------------------------------------------------------------------
# TestFilterCellsByArea
# ---------------------------------------------------------------------------


class TestFilterCellsByArea:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame(
            {
                "timepoint": [0, 0, 1, 1, 2],
                "cell_id": [1, 2, 3, 4, 5],
                "area_px": [100, 200, 150, 50, 300],
                "area_um2": [10.0, 20.0, 15.0, 5.0, 30.0],
            }
        )

    def test_min_only(self, df):
        result = filter_cells_by_area(df, min_area_um2=15.0)
        assert list(result["area_um2"]) == [20.0, 15.0, 30.0]

    def test_max_only(self, df):
        result = filter_cells_by_area(df, max_area_um2=15.0)
        assert list(result["area_um2"]) == [10.0, 15.0, 5.0]

    def test_both(self, df):
        result = filter_cells_by_area(df, min_area_um2=10.0, max_area_um2=20.0)
        assert list(result["area_um2"]) == [10.0, 20.0, 15.0]

    def test_no_filter(self, df):
        result = filter_cells_by_area(df)
        assert len(result) == len(df)

    def test_all_filtered(self, df):
        result = filter_cells_by_area(df, min_area_um2=100.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestComputeGrowthStats
# ---------------------------------------------------------------------------


class TestComputeGrowthStats:
    def test_basic_aggregation(self):
        df = pd.DataFrame(
            {
                "timepoint": [0, 0, 1, 1, 1],
                "cell_id": [1, 2, 3, 4, 5],
                "area_px": [100, 200, 150, 50, 300],
                "area_um2": [10.0, 20.0, 15.0, 5.0, 30.0],
            }
        )
        stats = compute_growth_stats(df)
        assert list(stats["timepoint"]) == [0, 1]
        assert list(stats["cell_count"]) == [2, 3]
        np.testing.assert_allclose(stats["mean_area_um2"], [15.0, 50.0 / 3])
        np.testing.assert_allclose(stats["total_area_um2"], [30.0, 50.0])

    def test_single_timepoint(self):
        df = pd.DataFrame(
            {
                "timepoint": [0, 0],
                "cell_id": [1, 2],
                "area_px": [100, 200],
                "area_um2": [10.0, 20.0],
            }
        )
        stats = compute_growth_stats(df)
        assert len(stats) == 1
        assert stats["cell_count"].iloc[0] == 2

    def test_sorted_output(self):
        df = pd.DataFrame(
            {
                "timepoint": [2, 0, 1],
                "cell_id": [1, 2, 3],
                "area_px": [100, 200, 150],
                "area_um2": [10.0, 20.0, 15.0],
            }
        )
        stats = compute_growth_stats(df)
        assert list(stats["timepoint"]) == [0, 1, 2]

    def test_columns_present(self):
        df = pd.DataFrame({"timepoint": [0], "cell_id": [1], "area_px": [100], "area_um2": [10.0]})
        stats = compute_growth_stats(df)
        assert set(stats.columns) == {"timepoint", "cell_count", "mean_area_um2", "total_area_um2"}


# ---------------------------------------------------------------------------
# TestFitExponentialGrowth
# ---------------------------------------------------------------------------


class TestFitExponentialGrowth:
    def test_perfect_exponential(self):
        """Exact exponential data should give R²≈1."""
        n0, lam = 10.0, 0.05
        t = np.arange(20, dtype=float)
        counts = n0 * np.exp(lam * t)
        result = fit_exponential_growth(t, counts)
        assert isinstance(result, ExponentialFitResult)
        np.testing.assert_allclose(result.n0, n0, rtol=1e-6)
        np.testing.assert_allclose(result.growth_rate, lam, rtol=1e-6)
        assert result.r_squared > 0.9999

    def test_noisy_data(self):
        rng = np.random.default_rng(42)
        n0, lam = 50.0, 0.03
        t = np.arange(30, dtype=float)
        counts = n0 * np.exp(lam * t) * (1 + 0.05 * rng.standard_normal(len(t)))
        counts = np.maximum(counts, 1)  # ensure positive
        result = fit_exponential_growth(t, counts)
        assert result.r_squared > 0.9
        np.testing.assert_allclose(result.growth_rate, lam, atol=0.01)

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_exponential_growth([0], [5])

    def test_doubling_time(self):
        lam = 0.1
        t = np.arange(50, dtype=float)
        counts = 10 * np.exp(lam * t)
        result = fit_exponential_growth(t, counts)
        expected_dt = np.log(2) / lam
        np.testing.assert_allclose(result.doubling_time, expected_dt, rtol=1e-4)

    def test_zero_counts_skipped(self):
        """Timepoints with count=0 should be skipped, not cause errors."""
        t = np.array([0, 1, 2, 3, 4], dtype=float)
        counts = np.array([0, 5, 10, 20, 40], dtype=float)
        result = fit_exponential_growth(t, counts)
        assert result.growth_rate > 0
        assert len(result.fitted_values) == len(t)


# ---------------------------------------------------------------------------
# TestFitLogisticGrowth
# ---------------------------------------------------------------------------


class TestFitLogisticGrowth:
    def test_perfect_logistic(self):
        """Exact logistic data should give R²≈1 and recover parameters."""
        n0, r, K = 10.0, 0.05, 1000.0
        t = np.arange(100, dtype=float)
        counts = K / (1.0 + ((K - n0) / n0) * np.exp(-r * t))
        result = fit_logistic_growth(t, counts)
        assert isinstance(result, LogisticFitResult)
        assert result.r_squared > 0.999
        np.testing.assert_allclose(result.n0, n0, rtol=0.05)
        np.testing.assert_allclose(result.growth_rate, r, rtol=0.05)
        np.testing.assert_allclose(result.carrying_capacity, K, rtol=0.05)

    def test_noisy_data(self):
        """Noisy logistic data should still fit reasonably."""
        rng = np.random.default_rng(42)
        n0, r, K = 20.0, 0.08, 500.0
        t = np.arange(80, dtype=float)
        counts = K / (1.0 + ((K - n0) / n0) * np.exp(-r * t))
        counts = counts * (1 + 0.05 * rng.standard_normal(len(t)))
        counts = np.maximum(counts, 1)
        result = fit_logistic_growth(t, counts)
        assert result.r_squared > 0.95

    def test_too_few_points(self):
        """Fewer than 3 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            fit_logistic_growth([0, 1], [5, 10])

    def test_doubling_time(self):
        """Doubling time should equal ln(2)/r."""
        n0, r, K = 10.0, 0.1, 1000.0
        t = np.arange(100, dtype=float)
        counts = K / (1.0 + ((K - n0) / n0) * np.exp(-r * t))
        result = fit_logistic_growth(t, counts)
        expected_dt = np.log(2) / r
        np.testing.assert_allclose(result.doubling_time, expected_dt, rtol=0.05)

    def test_carrying_capacity_recovery(self):
        """Carrying capacity K should be recovered near true value."""
        n0, r, K = 5.0, 0.06, 800.0
        t = np.arange(120, dtype=float)
        counts = K / (1.0 + ((K - n0) / n0) * np.exp(-r * t))
        result = fit_logistic_growth(t, counts)
        np.testing.assert_allclose(result.carrying_capacity, K, rtol=0.05)
