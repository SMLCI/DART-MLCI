"""Integration tests for registration methods.

Tests end-to-end usage in realistic scenarios including synthetic time-lapse
sequences, performance benchmarks, and robustness checks.  Tests are
parametrized over both NCC and Phase Correlation where behaviour should be
equivalent.
"""

import time

import numpy as np
import pytest

from dart_mlci.registration import PhaseCorrelationRegistration, TimelapseRegistration
from tests.fixtures.synthetic_markers import (
    apply_known_translation,
    create_marker_group_pixel,
    create_synthetic_marker_image,
    create_synthetic_timelapse,
)

# ---------------------------------------------------------------------------
# Parametrized fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ncc", "phase"])
def reg_factory(request):
    """Factory that builds a registration instance from a marker group.

    Returns a callable ``(marker_group, **kwargs) -> registration_instance``.
    """

    def _make(marker_group, **kw):
        if request.param == "ncc":
            return TimelapseRegistration(
                marker_group_pixel=marker_group,
                max_translation=kw.get("max_translation", 20),
                padding=kw.get("padding", 50),
                device="cpu",
            )
        else:
            return PhaseCorrelationRegistration(
                marker_group_pixel=marker_group,
                padding=kw.get("padding", 50),
            )

    _make.method = request.param
    return _make


# ===================================================================
# Synthetic time-lapse tests
# ===================================================================


class TestSyntheticTimelapse:
    """Parametrized time-lapse tests."""

    def test_register_synthetic_timelapse(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        num_frames = 10
        drift_per_frame = (1.0, 0.5)
        frames, true_translations = create_synthetic_timelapse(
            num_frames=num_frames,
            width=640,
            height=480,
            marker_positions=marker_positions,
            drift_per_frame=drift_per_frame,
            background="cells",
        )
        reg = reg_factory(marker_group, max_translation=20, padding=50)
        reference = frames[0]

        for i in range(1, num_frames):
            dx, dy, _score = reg.compute_translation(reference, frames[i])
            true_dx, true_dy = true_translations[i]
            assert abs(dx - true_dx) <= 1.0, f"Frame {i}: dx~{true_dx}, got {dx}"
            assert abs(dy - true_dy) <= 1.0, f"Frame {i}: dy~{true_dy}, got {dy}"

    def test_register_and_align_timelapse(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        num_frames = 5
        drift_per_frame = (2.0, 1.0)
        frames, _ = create_synthetic_timelapse(
            num_frames=num_frames,
            width=640,
            height=480,
            marker_positions=marker_positions,
            drift_per_frame=drift_per_frame,
            background="gradient",
        )
        reg = reg_factory(marker_group, max_translation=30, padding=50)
        reference = frames[0]
        aligned_frames = [reference]

        for i in range(1, num_frames):
            dx, dy, _ = reg.compute_translation(reference, frames[i])
            aligned = reg.apply_translation(frames[i], -dx, -dy)
            aligned_frames.append(aligned)

        # Aligned marker regions should be close to reference
        for i in range(1, num_frames):
            ref_region = reg.extract_marker_region(reference)
            aligned_region = reg.extract_marker_region(aligned_frames[i])
            diff = np.abs(ref_region.astype(float) - aligned_region.astype(float))
            assert diff.mean() < 10.0, f"Frame {i}: mean diff = {diff.mean()}"

    def test_varying_drift(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reference = create_synthetic_timelapse(
            num_frames=1,
            width=640,
            height=480,
            marker_positions=marker_positions,
            drift_per_frame=(0, 0),
        )[0][0]
        reg = reg_factory(marker_group, max_translation=30, padding=50)
        drift_patterns = [(0, 0), (2, 1), (5, 2), (3, 4), (-2, 3), (1, -1)]

        for i, (dx_true, dy_true) in enumerate(drift_patterns):
            target = apply_known_translation(reference, dx_true, dy_true)
            dx, dy, _ = reg.compute_translation(reference, target)
            assert abs(dx - dx_true) < 1.0, f"Pattern {i}: dx~{dx_true}, got {dx}"
            assert abs(dy - dy_true) < 1.0, f"Pattern {i}: dy~{dy_true}, got {dy}"


# ===================================================================
# Accuracy statistics
# ===================================================================


class TestAccuracyStatistics:
    """Statistical accuracy over many random translations."""

    def test_translation_accuracy_statistics(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = reg_factory(marker_group, max_translation=20, padding=50)
        reference = create_synthetic_marker_image(640, 480, marker_positions, background="uniform")

        num_samples = 50
        errors_x, errors_y = [], []
        np.random.seed(42)

        for _ in range(num_samples):
            true_dx = np.random.uniform(-15, 15)
            true_dy = np.random.uniform(-15, 15)
            target = apply_known_translation(reference, true_dx, true_dy)
            dx, dy, _ = reg.compute_translation(reference, target)
            errors_x.append(abs(dx - true_dx))
            errors_y.append(abs(dy - true_dy))

        assert np.mean(errors_x) < 0.5
        assert np.mean(errors_y) < 0.5
        assert np.percentile(errors_x, 95) < 1.0
        assert np.percentile(errors_y, 95) < 1.0


# ===================================================================
# Performance benchmarks
# ===================================================================


class TestPerformance:
    """Performance benchmarks."""

    def test_registration_speed_benchmark(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = reg_factory(marker_group, max_translation=20, padding=50)
        reference = create_synthetic_marker_image(640, 480, marker_positions)

        # Warm-up
        target = apply_known_translation(reference, 5, 3)
        reg.compute_translation(reference, target)

        num_iterations = 10
        times = []
        for i in range(num_iterations):
            target = apply_known_translation(reference, i, i * 0.5)
            start = time.time()
            reg.compute_translation(reference, target)
            times.append(time.time() - start)

        avg_ms = np.mean(times) * 1000
        assert avg_ms < 500, f"Average time {avg_ms:.1f}ms > 500ms"

    def test_memory_efficiency(self, reg_factory):
        import os

        import psutil

        marker_positions = {"cross": (1000, 1000), "circle": (1500, 1200)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = reg_factory(marker_group, max_translation=20, padding=50)

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        reference = create_synthetic_marker_image(2048, 2048, marker_positions)
        target = apply_known_translation(reference, 10, 5)
        reg.compute_translation(reference, target)

        mem_after = process.memory_info().rss / 1024 / 1024
        assert (mem_after - mem_before) < 100


# ===================================================================
# Robustness
# ===================================================================


class TestRobustness:
    """Robustness tests."""

    def test_with_noise(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = reg_factory(marker_group, max_translation=20, padding=50)
        reference = create_synthetic_marker_image(640, 480, marker_positions, background="noise")

        true_dx, true_dy = 7, -4
        target = apply_known_translation(reference, true_dx, true_dy)
        noise = np.random.randint(-20, 20, target.shape, dtype=np.int16)
        target_noisy = np.clip(target.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        dx, dy, _ = reg.compute_translation(reference, target_noisy)
        assert abs(dx - true_dx) < 2.0
        assert abs(dy - true_dy) < 2.0

    def test_different_backgrounds(self, reg_factory):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = reg_factory(marker_group, max_translation=20, padding=50)

        for bg in ["uniform", "gradient", "noise", "cells"]:
            reference = create_synthetic_marker_image(640, 480, marker_positions, background=bg)
            target = apply_known_translation(reference, 5, -3)
            dx, dy, _ = reg.compute_translation(reference, target)
            assert abs(dx - 5) < 1.0, f"Background '{bg}': dx~5, got {dx}"
            assert abs(dy - (-3)) < 1.0, f"Background '{bg}': dy~-3, got {dy}"
