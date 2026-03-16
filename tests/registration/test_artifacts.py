"""Registration tests using real artifact images (SAK sequence, image stack).

This module consolidates all tests that require real microscopy data from the
artifacts directory.  It covers:

- SAK sequence registration with both NCC and Phase Correlation
- Method comparison (speed, accuracy, consistency)
- Translation accuracy with ground-truth shifts
- Alignment quality (SSIM) verification
- Warp backend isolation tests
- Marker detection integration

All classes are marked with ``@pytest.mark.artifacts`` so they can be
selected/excluded via ``-m artifacts``.
"""

import math
import time

import numpy as np
import pytest

from dart_mlci.registration import PhaseCorrelationRegistration, TimelapseRegistration
from tests.fixtures.artifact_helpers import (
    create_marker_group_from_detection,
    detect_markers_in_image,
    load_image_stack,
    load_sak_sequence,
    normalize_to_uint8,
)
from tests.fixtures.synthetic_markers import apply_known_translation

pytestmark = pytest.mark.artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_marker_groups(markers):
    """Create flat marker-group dict from detected markers (both methods share format now)."""
    mg = {}
    for i, marker in enumerate(markers):
        center = marker.get("mask_center", marker["bbox_center"])
        label = marker.get("label", f"marker{i}")
        mg[label] = center
    return mg


def _translation_error(dx_est, dy_est, dx_true, dy_true):
    return math.sqrt((dx_est - dx_true) ** 2 + (dy_est - dy_true) ** 2)


def _align_ncc(reg, target, dx, dy):
    return reg.apply_translation(target, -dx, -dy, return_tensor=False)


def _align_phase(reg, target, dx, dy):
    return reg.apply_translation(target, -dx, -dy)


def _interior_ssim(reference, aligned, dx_true, dy_true, extra_border=5):
    from tests.utils.image_comparison import compute_ssim

    border = int(max(abs(dx_true), abs(dy_true))) + extra_border
    h, w = reference.shape[:2]
    if border * 2 >= h or border * 2 >= w:
        border = min(h, w) // 4
    ref_crop = reference[border : h - border, border : w - border]
    aln_crop = aligned[border : h - border, border : w - border]
    channel_axis = -1 if ref_crop.ndim == 3 else None
    return compute_ssim(ref_crop, aln_crop, channel_axis=channel_axis)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sak_reference_setup():
    """Load SAK frame 0, detect markers, create both registration instances."""
    frames, _ = load_sak_sequence(max_frames=1)
    reference = normalize_to_uint8(frames[0])
    markers = detect_markers_in_image(reference)
    assert len(markers) >= 2
    mg = _create_marker_groups(markers)

    reg_ncc = TimelapseRegistration(mg, max_translation=25, padding=50)
    reg_phase = PhaseCorrelationRegistration(mg, padding=100)

    return {
        "reference": reference,
        "markers": markers,
        "marker_group": mg,
        "reg_ncc": reg_ncc,
        "reg_phase": reg_phase,
    }


# ===================================================================
# 1. SAK sequence registration (NCC)
# ===================================================================


class TestSAKSequenceRegistration:
    """Tests with real SAK microscopy sequence."""

    def test_load_sak_sequence(self):
        frames, frame_numbers = load_sak_sequence()
        assert len(frames) == 10
        assert 2 not in frame_numbers
        for frame in frames:
            assert frame.ndim == 3
            assert frame.shape[2] == 3

    def test_detect_markers(self):
        frames, _ = load_sak_sequence(max_frames=1)
        markers = detect_markers_in_image(normalize_to_uint8(frames[0]))
        assert len(markers) >= 2
        for m in markers:
            assert "bbox_center" in m
            assert m["label"] in ["cross", "circle"]

    def test_register_frame_pairs(self):
        frames, _ = load_sak_sequence(max_frames=3)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = create_marker_group_from_detection(markers)
        reg = TimelapseRegistration(mg, max_translation=30, padding=50, device="cpu")
        dx, dy, score = reg.compute_translation(frames_u8[0], frames_u8[1])
        assert abs(dx) <= 30 and abs(dy) <= 30 and score > 0.05

    def test_cumulative_drift(self):
        frames, _ = load_sak_sequence()
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = create_marker_group_from_detection(markers)
        reg = TimelapseRegistration(mg, max_translation=30, padding=50, device="cpu")
        reference = frames_u8[0]
        translations = [(0.0, 0.0)]
        for i in range(1, len(frames_u8)):
            dx, dy, _ = reg.compute_translation(reference, frames_u8[i])
            translations.append((dx, dy))
            assert abs(dx) <= 30 and abs(dy) <= 30
        magnitudes = np.sqrt(np.array(translations)[:, 0] ** 2 + np.array(translations)[:, 1] ** 2)
        assert np.max(magnitudes) < 50


# ===================================================================
# 2. Method comparison
# ===================================================================


class TestRegistrationMethodComparison:
    """Compare phase correlation vs NCC on real data."""

    def test_both_methods_on_first_pair(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = _create_marker_groups(markers)

        reg_ncc = TimelapseRegistration(mg, padding=50)
        reg_phase = PhaseCorrelationRegistration(mg, padding=100)

        dx_ncc, _dy_ncc, _score_ncc = reg_ncc.compute_translation(frames_u8[0], frames_u8[1])
        dx_phase, dy_phase, _conf_phase = reg_phase.compute_translation(frames_u8[0], frames_u8[1])

        assert isinstance(dx_ncc, float) and isinstance(dx_phase, float)
        assert abs(dx_phase) < 50 and abs(dy_phase) < 50

    def test_translation_consistency(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = _create_marker_groups(markers)

        reg_ncc = TimelapseRegistration(mg, padding=50)
        reg_phase = PhaseCorrelationRegistration(mg, padding=100)

        dx_ncc, dy_ncc, score_ncc = reg_ncc.compute_translation(frames_u8[0], frames_u8[1])
        dx_phase, dy_phase, conf_phase = reg_phase.compute_translation(frames_u8[0], frames_u8[1])

        if conf_phase > 0.3 and score_ncc > 0.05:
            assert abs(dx_ncc - dx_phase) < 10
            assert abs(dy_ncc - dy_phase) < 10

    def test_performance_comparison(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = _create_marker_groups(markers)

        reg_ncc = TimelapseRegistration(mg, padding=50)
        reg_phase = PhaseCorrelationRegistration(mg, padding=100)

        n_runs = 5
        start = time.time()
        for _ in range(n_runs):
            reg_ncc.compute_translation(frames_u8[0], frames_u8[1])
        time_ncc = (time.time() - start) / n_runs

        start = time.time()
        for _ in range(n_runs):
            reg_phase.compute_translation(frames_u8[0], frames_u8[1])
        time_phase = (time.time() - start) / n_runs

        # Just document — no hard assertion on which is faster
        print(f"\nNCC: {time_ncc * 1000:.1f}ms, Phase: {time_phase * 1000:.1f}ms")


# ===================================================================
# 3. Translation accuracy with ground-truth shifts
# ===================================================================

INTEGER_TRANSLATIONS = [(5, 0), (0, 5), (5, -3), (10, 7), (-8, 4)]
SUBPIXEL_TRANSLATIONS = [(2.5, 1.5), (0.7, -0.3)]
LARGER_TRANSLATIONS = [(15, 10), (18, -15)]
ALL_TRANSLATIONS = INTEGER_TRANSLATIONS + SUBPIXEL_TRANSLATIONS + LARGER_TRANSLATIONS


class TestTranslationAccuracy:
    """Parametrized tests for translation detection accuracy."""

    @pytest.mark.parametrize("dx_true,dy_true", INTEGER_TRANSLATIONS)
    def test_ncc_integer(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 1.5

    @pytest.mark.parametrize("dx_true,dy_true", INTEGER_TRANSLATIONS)
    def test_phase_integer(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 1.5

    @pytest.mark.parametrize("dx_true,dy_true", SUBPIXEL_TRANSLATIONS)
    def test_ncc_subpixel(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 1.5

    @pytest.mark.parametrize("dx_true,dy_true", SUBPIXEL_TRANSLATIONS)
    def test_phase_subpixel(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 1.0

    @pytest.mark.parametrize("dx_true,dy_true", LARGER_TRANSLATIONS)
    def test_ncc_larger(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 2.0

    @pytest.mark.parametrize("dx_true,dy_true", LARGER_TRANSLATIONS)
    def test_phase_larger(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, dx_true, dy_true) < 2.0


# ===================================================================
# 4. Alignment quality (SSIM)
# ===================================================================


class TestAlignmentQuality:
    """Verify corrected images match the reference."""

    @pytest.mark.parametrize("dx_true,dy_true", ALL_TRANSLATIONS)
    def test_ncc_alignment(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        aligned = np.clip(_align_ncc(reg, target, dx_est, dy_est), 0, 255).astype(np.uint8)
        assert _interior_ssim(ref, aligned, dx_true, dy_true) > 0.88

    @pytest.mark.parametrize("dx_true,dy_true", ALL_TRANSLATIONS)
    def test_phase_alignment(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, dx_true, dy_true)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        aligned = np.clip(_align_phase(reg, target, dx_est, dy_est), 0, 255).astype(np.uint8)
        assert _interior_ssim(ref, aligned, dx_true, dy_true) > 0.90


# ===================================================================
# 5. Warp backend isolation
# ===================================================================


class TestWarpBackends:
    """Test warp/apply in isolation (no detection)."""

    @pytest.mark.parametrize("dx_true,dy_true", ALL_TRANSLATIONS)
    def test_ncc_warp_roundtrip(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, dx_true, dy_true)
        corrected = np.clip(_align_ncc(reg, target, dx_true, dy_true), 0, 255).astype(np.uint8)
        assert _interior_ssim(ref, corrected, dx_true, dy_true) > 0.90

    @pytest.mark.parametrize("dx_true,dy_true", ALL_TRANSLATIONS)
    def test_phase_warp_roundtrip(self, sak_reference_setup, dx_true, dy_true):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, dx_true, dy_true)
        corrected = np.clip(_align_phase(reg, target, dx_true, dy_true), 0, 255).astype(np.uint8)
        assert _interior_ssim(ref, corrected, dx_true, dy_true) > 0.95

    def test_warp_direction_ncc(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        warped = np.clip(reg.apply_translation(ref, 5.0, 0.0), 0, 255).astype(np.uint8)
        h = ref.shape[0]
        strip_orig = ref[h // 4 : 3 * h // 4, 100:110]
        strip_warp = warped[h // 4 : 3 * h // 4, 105:115]
        diff = np.abs(strip_orig.astype(float) - strip_warp.astype(float)).mean()
        assert diff < 10

    def test_warp_direction_phase(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        warped = reg.apply_translation(ref, 5.0, 0.0)
        h = ref.shape[0]
        strip_orig = ref[h // 4 : 3 * h // 4, 100:110]
        strip_warp = warped[h // 4 : 3 * h // 4, 105:115]
        diff = np.abs(strip_orig.astype(float) - strip_warp.astype(float)).mean()
        assert diff < 10

    def test_both_backends_same_direction(self, sak_reference_setup):
        from tests.utils.image_comparison import compute_ssim

        ref = sak_reference_setup["reference"]
        reg_ncc = sak_reference_setup["reg_ncc"]
        reg_phase = sak_reference_setup["reg_phase"]

        warped_ncc = np.clip(reg_ncc.apply_translation(ref, 5.0, 0.0), 0, 255).astype(np.uint8)
        warped_phase = reg_phase.apply_translation(ref, 5.0, 0.0)

        h, w = ref.shape[:2]
        border = 15
        ncc_i = warped_ncc[border : h - border, border : w - border]
        phase_i = warped_phase[border : h - border, border : w - border]
        channel_axis = -1 if ncc_i.ndim == 3 else None
        assert compute_ssim(ncc_i, phase_i, channel_axis=channel_axis) > 0.93


# ===================================================================
# 6. NCC max-translation boundary
# ===================================================================


class TestNCCMaxTranslationBoundary:
    def test_at_boundary(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_ncc"]
        target = apply_known_translation(ref, 20, 0)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, 20, 0) < 2.0

    def test_beyond_boundary(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg_small = TimelapseRegistration(
            sak_reference_setup["marker_group"], max_translation=10, padding=50
        )
        target = apply_known_translation(ref, 18, -15)
        dx_est, dy_est, _ = reg_small.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, 18, -15) > 5.0

    def test_phase_handles_large_shift(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg = sak_reference_setup["reg_phase"]
        target = apply_known_translation(ref, 18, -15)
        dx_est, dy_est, _ = reg.compute_translation(ref, target)
        assert _translation_error(dx_est, dy_est, 18, -15) < 2.0


# ===================================================================
# 7. Phase correlation on real data
# ===================================================================


class TestPhaseCorrelationOnRealData:
    def test_phase_corr_on_sak_sequence(self):
        frames, _ = load_sak_sequence(max_frames=3)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = _create_marker_groups(markers)
        reg = PhaseCorrelationRegistration(mg, padding=100, preprocess=True)

        confidences = []
        for i in range(len(frames_u8) - 1):
            _, _, conf = reg.compute_translation(frames_u8[i], frames_u8[i + 1])
            confidences.append(conf)

        assert len(confidences) == len(frames_u8) - 1
        assert all(isinstance(c, float) for c in confidences)

    def test_preprocessing_effect(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = _create_marker_groups(markers)

        reg_no = PhaseCorrelationRegistration(mg, padding=100, preprocess=False)
        reg_yes = PhaseCorrelationRegistration(mg, padding=100, preprocess=True)

        _, _, conf_no = reg_no.compute_translation(frames_u8[0], frames_u8[1])
        _, _, conf_yes = reg_yes.compute_translation(frames_u8[0], frames_u8[1])

        assert isinstance(conf_no, float) and isinstance(conf_yes, float)


# ===================================================================
# 8. Image stack tests
# ===================================================================


class TestImageStackRegistration:
    def test_load_image_stack(self):
        frames = load_image_stack(channel=0, max_frames=5)
        assert len(frames) == 5
        for f in frames:
            assert f.ndim == 3 and f.shape[2] == 3

    def test_register_first_10_frames(self):
        frames = load_image_stack(channel=0, max_frames=10)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        if len(markers) < 2:
            pytest.skip("No markers detected")
        mg = create_marker_group_from_detection(markers)
        if "cross" not in mg or "circle" not in mg:
            pytest.skip("Missing markers")

        reg = TimelapseRegistration(mg, max_translation=40, padding=50, device="cpu")
        reference = frames_u8[0]
        translations = []
        for i in range(1, len(frames_u8)):
            dx, dy, _ = reg.compute_translation(reference, frames_u8[i])
            translations.append((dx, dy))
            assert abs(dx) < 50 and abs(dy) < 50

        magnitudes = np.sqrt(np.array(translations)[:, 0] ** 2 + np.array(translations)[:, 1] ** 2)
        assert np.max(magnitudes) > 1.0


# ===================================================================
# 9. Marker detection integration
# ===================================================================


class TestMarkerDetectionIntegration:
    def test_end_to_end_sak_pipeline(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = create_marker_group_from_detection(markers)
        assert "cross" in mg and "circle" in mg

        reg = TimelapseRegistration(mg, max_translation=30, padding=50, device="cpu")
        dx, dy, score = reg.compute_translation(frames_u8[0], frames_u8[1])
        aligned = reg.apply_translation(frames_u8[1], dx, dy)
        assert aligned.shape == frames_u8[1].shape
        assert score > 0.05

    def test_marker_detection_robustness(self):
        frames, _ = load_sak_sequence()
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        success = sum(1 for f in frames_u8 if len(detect_markers_in_image(f)) >= 2)
        assert success / len(frames_u8) > 0.8


# ===================================================================
# 10. Performance with real images
# ===================================================================


class TestArtifactPerformance:
    def test_high_resolution_performance(self):
        frames, _ = load_sak_sequence(max_frames=2)
        frames_u8 = [normalize_to_uint8(f) for f in frames]
        markers = detect_markers_in_image(frames_u8[0])
        mg = create_marker_group_from_detection(markers)

        reg = TimelapseRegistration(mg, max_translation=30, padding=50, device="cpu")
        start = time.time()
        reg.compute_translation(frames_u8[0], frames_u8[1])
        assert (time.time() - start) < 2.0

    def test_summary_table(self, sak_reference_setup):
        ref = sak_reference_setup["reference"]
        reg_ncc = sak_reference_setup["reg_ncc"]
        reg_phase = sak_reference_setup["reg_phase"]

        lines = []
        for dx_true, dy_true in ALL_TRANSLATIONS:
            target = apply_known_translation(ref, dx_true, dy_true)

            dx_n, dy_n, _ = reg_ncc.compute_translation(ref, target)
            err_n = _translation_error(dx_n, dy_n, dx_true, dy_true)
            aln_n = np.clip(_align_ncc(reg_ncc, target, dx_n, dy_n), 0, 255).astype(np.uint8)
            ssim_n = _interior_ssim(ref, aln_n, dx_true, dy_true)

            dx_p, dy_p, _ = reg_phase.compute_translation(ref, target)
            err_p = _translation_error(dx_p, dy_p, dx_true, dy_true)
            aln_p = np.clip(_align_phase(reg_phase, target, dx_p, dy_p), 0, 255).astype(np.uint8)
            ssim_p = _interior_ssim(ref, aln_p, dx_true, dy_true)

            winner = "NCC" if err_n < err_p else ("Phase" if err_p < err_n else "tie")
            lines.append(
                f"({dx_true:+5.1f},{dy_true:+5.1f}): NCC err={err_n:.3f} ssim={ssim_n:.4f}  Phase err={err_p:.3f} ssim={ssim_p:.4f}  {winner}"
            )

        print("\n".join(lines))
