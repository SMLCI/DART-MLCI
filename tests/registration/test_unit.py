"""Unit tests for registration classes (NCC and Phase Correlation).

Tests are parametrized over both methods using shared synthetic data wherever
the expected behaviour is identical. Method-specific tests (e.g. NCC's
``apply_translation_to_mask`` with kornia, GPU tensor handling) are kept
separate.
"""

import cv2
import numpy as np
import pytest
import torch

from dmc_masking.registration import (
    BaseRegistration,
    PhaseCorrelationRegistration,
    TimelapseRegistration,
)
from tests.fixtures.synthetic_markers import (
    apply_known_translation,
    create_marker_group_pixel,
    create_synthetic_marker_image,
)

# ---------------------------------------------------------------------------
# Parametrized fixture: yields both methods
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ncc", "phase"])
def reg_instance(request, marker_group_fixture):
    """Registration instance parametrized over both methods."""
    if request.param == "ncc":
        return TimelapseRegistration(
            marker_group_pixel=marker_group_fixture,
            max_translation=20,
            padding=50,
            device="cpu",
        )
    else:
        return PhaseCorrelationRegistration(
            marker_group_pixel=marker_group_fixture,
            padding=50,
        )


# ===================================================================
# 1. Initialization tests
# ===================================================================


class TestRegistrationInit:
    """Initialization tests parametrized over both methods."""

    def test_init_stores_marker_group(self, reg_instance, marker_group_fixture):
        assert reg_instance.marker_group_pixel == marker_group_fixture

    def test_init_stores_padding(self, reg_instance):
        assert reg_instance.padding == 50

    def test_init_computes_bbox(self, reg_instance):
        assert reg_instance.marker_bbox is not None
        assert len(reg_instance.marker_bbox) == 4

    def test_isinstance_base(self, reg_instance):
        assert isinstance(reg_instance, BaseRegistration)

    def test_init_empty_markers_raises(self):
        with pytest.raises(ValueError):
            TimelapseRegistration(marker_group_pixel={}, device="cpu")
        with pytest.raises(ValueError):
            PhaseCorrelationRegistration(marker_group_pixel={})

    def test_marker_bbox_computation(self):
        marker_group = create_marker_group_pixel({"cross": (100, 100), "circle": (200, 150)})
        reg = TimelapseRegistration(marker_group_pixel=marker_group, padding=50, device="cpu")
        expected = (50, 50, 250, 200)
        assert reg.marker_bbox == expected

        reg_phase = PhaseCorrelationRegistration(marker_group_pixel=marker_group, padding=50)
        assert reg_phase.marker_bbox == expected

    def test_marker_bbox_negative_clipping(self):
        marker_group = create_marker_group_pixel({"cross": (20, 30), "circle": (100, 100)})
        reg = TimelapseRegistration(marker_group_pixel=marker_group, padding=50, device="cpu")
        x_min, y_min, _, _ = reg.marker_bbox
        assert x_min >= 0
        assert y_min >= 0

    def test_get_marker_bbox(self, reg_instance):
        bbox = reg_instance.get_marker_bbox()
        assert bbox == reg_instance.marker_bbox

    def test_get_marker_region_size(self, reg_instance):
        w, h = reg_instance.get_marker_region_size()
        x_min, y_min, x_max, y_max = reg_instance.marker_bbox
        assert w == x_max - x_min
        assert h == y_max - y_min


class TestNCCSpecificInit:
    """NCC-specific initialization tests."""

    def test_auto_device_detection(self, marker_group_fixture):
        reg = TimelapseRegistration(marker_group_pixel=marker_group_fixture, device=None)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert reg.device == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explicit_cuda_device(self, marker_group_fixture):
        reg = TimelapseRegistration(marker_group_pixel=marker_group_fixture, device="cuda")
        assert reg.device == "cuda"


class TestPhaseSpecificInit:
    """Phase correlation-specific initialization tests."""

    def test_default_preprocessing_flags(self, marker_group_fixture):
        reg = PhaseCorrelationRegistration(marker_group_pixel=marker_group_fixture)
        assert reg.preprocess is True
        assert reg.use_hanning is True


# ===================================================================
# 2. Marker region extraction
# ===================================================================


class TestMarkerRegionExtraction:
    """Tests for extract_marker_region (shared behaviour)."""

    def test_extract_rgb(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)
        region = reg_instance.extract_marker_region(image)
        assert region.ndim == 3
        assert region.shape[0] > 0
        assert region.shape[1] > 0
        assert region.shape[2] == 3

    def test_extract_grayscale(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)[:, :, 0]
        region = reg_instance.extract_marker_region(image)
        assert region.ndim == 2
        assert region.shape[0] > 0
        assert region.shape[1] > 0

    def test_extract_boundary_clipping(self):
        marker_group = create_marker_group_pixel({"cross": (50, 50), "circle": (100, 80)})
        reg = TimelapseRegistration(marker_group_pixel=marker_group, padding=50, device="cpu")
        image = create_synthetic_marker_image(200, 150, {"cross": (50, 50), "circle": (100, 80)})
        region = reg.extract_marker_region(image)
        assert region.shape[0] > 0
        assert region.shape[1] > 0

    def test_extract_matches_bbox(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)
        region = reg_instance.extract_marker_region(image)
        x_min, y_min, x_max, y_max = reg_instance.marker_bbox
        expected = image[y_min:y_max, x_min:x_max]
        np.testing.assert_array_equal(region, expected)


# ===================================================================
# 3. Translation computation
# ===================================================================


class TestTranslationComputation:
    """Tests for compute_translation (shared behaviour)."""

    def test_no_shift(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions, background="cells")
        dx, dy, _score = reg_instance.compute_translation(image, image)
        assert abs(dx) < 0.5
        assert abs(dy) < 0.5

    def test_known_shift(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        reference = create_synthetic_marker_image(640, 480, marker_positions, background="uniform")
        true_dx, true_dy = 5, -3
        target = apply_known_translation(reference, true_dx, true_dy)
        dx, dy, _score = reg_instance.compute_translation(reference, target)
        assert abs(dx - true_dx) < 1.0, f"Expected dx~{true_dx}, got {dx}"
        assert abs(dy - true_dy) < 1.0, f"Expected dy~{true_dy}, got {dy}"

    @pytest.mark.parametrize(
        "shift",
        [(10, 0), (0, 10), (-5, 7), (8, -6), (-10, -10)],
    )
    def test_multiple_shifts(self, reg_instance, shift):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        reference = create_synthetic_marker_image(640, 480, marker_positions, background="gradient")
        true_dx, true_dy = shift
        target = apply_known_translation(reference, true_dx, true_dy)
        dx, dy, _score = reg_instance.compute_translation(reference, target)
        assert abs(dx - true_dx) < 1.0, f"Shift {shift}: dx~{true_dx}, got {dx}"
        assert abs(dy - true_dy) < 1.0, f"Shift {shift}: dy~{true_dy}, got {dy}"

    def test_grayscale_vs_rgb_consistency(self):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        marker_group = create_marker_group_pixel(marker_positions)
        reg = TimelapseRegistration(
            marker_group_pixel=marker_group, max_translation=20, padding=50, device="cpu"
        )
        reference_rgb = create_synthetic_marker_image(640, 480, marker_positions)
        true_dx, true_dy = 7, -4
        target_rgb = apply_known_translation(reference_rgb, true_dx, true_dy)
        reference_gray = reference_rgb[:, :, 0]
        target_gray = target_rgb[:, :, 0]

        dx_rgb, dy_rgb, _ = reg.compute_translation(reference_rgb, target_rgb)
        dx_gray, dy_gray, _ = reg.compute_translation(reference_gray, target_gray)

        assert abs(dx_rgb - dx_gray) < 0.5
        assert abs(dy_rgb - dy_gray) < 0.5


# ===================================================================
# 4. Apply translation
# ===================================================================


class TestApplyTranslation:
    """Tests for apply_translation (shared behaviour)."""

    def test_returns_same_shape(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)
        translated = reg_instance.apply_translation(image, 10, -5)
        assert translated.shape == image.shape

    def test_identity_translation(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)
        translated = reg_instance.apply_translation(image, 0, 0)
        np.testing.assert_allclose(translated, image, atol=1.0)

    def test_roundtrip_translation(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        image = create_synthetic_marker_image(640, 480, marker_positions)
        dx, dy = 12, -8
        translated = reg_instance.apply_translation(image, dx, dy)
        back = reg_instance.apply_translation(translated, -dx, -dy)
        # Compare interior to avoid border effects
        diff = np.abs(back[20:-20, 20:-20].astype(float) - image[20:-20, 20:-20].astype(float))
        assert diff.mean() < 5


class TestRegisterToReference:
    """Tests for batch register_to_reference."""

    def test_register_multiple(self, reg_instance):
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        ref = create_synthetic_marker_image(640, 480, marker_positions, background="gradient")
        targets = [apply_known_translation(ref, dx, dy) for dx, dy in [(3, 2), (-5, 4)]]
        results = reg_instance.register_to_reference(ref, targets)
        assert len(results) == 2
        for aligned, _dx, _dy, _score in results:
            assert aligned.shape == ref.shape


# ===================================================================
# 5. NCC-specific tests
# ===================================================================


class TestNCCApplyTranslation:
    """NCC-specific apply_translation tests (tensor handling, GPU)."""

    def test_numpy_hwc(self, registration_instance_fixture):
        image = create_synthetic_marker_image(640, 480, {"cross": (200, 200), "circle": (300, 250)})
        translated = registration_instance_fixture.apply_translation(image, 10, -5)
        assert isinstance(translated, np.ndarray)
        assert translated.shape == image.shape

    def test_tensor_chw(self, registration_instance_fixture):
        image = create_synthetic_marker_image(640, 480, {"cross": (200, 200), "circle": (300, 250)})
        image_chw = torch.from_numpy(image).permute(2, 0, 1).float()
        translated = registration_instance_fixture.apply_translation(
            image_chw, 7, 3, return_tensor=True
        )
        assert isinstance(translated, torch.Tensor)
        assert translated.shape == image_chw.shape

    def test_grayscale(self, registration_instance_fixture):
        image_rgb = create_synthetic_marker_image(
            640, 480, {"cross": (200, 200), "circle": (300, 250)}
        )
        image_gray = image_rgb[:, :, 0]
        translated = registration_instance_fixture.apply_translation(image_gray, 6, -4)
        assert translated.shape == image_gray.shape
        assert translated.ndim == 2

    def test_deprecated_apply_translation_to_image(self, registration_instance_fixture):
        image = create_synthetic_marker_image(640, 480, {"cross": (200, 200), "circle": (300, 250)})
        with pytest.warns(DeprecationWarning, match="apply_translation_to_image"):
            translated = registration_instance_fixture.apply_translation_to_image(image, 5, 3)
        assert isinstance(translated, np.ndarray)

    def test_invalid_tensor_shape(self, registration_instance_fixture):
        invalid = torch.randn(480, 640)
        with pytest.raises(ValueError, match="Unexpected tensor shape"):
            registration_instance_fixture.apply_translation(invalid, 5, 3)

    def test_numpy_tensor_consistency(self, registration_instance_fixture):
        image_np = create_synthetic_marker_image(
            640, 480, {"cross": (200, 200), "circle": (300, 250)}
        )
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        dx, dy = 8, -6

        translated_np = registration_instance_fixture.apply_translation(image_np, dx, dy)
        translated_tensor = registration_instance_fixture.apply_translation(
            image_tensor, dx, dy, return_tensor=True
        )
        translated_tensor_np = translated_tensor.cpu().numpy().transpose(1, 2, 0)
        np.testing.assert_allclose(translated_np, translated_tensor_np, atol=2.0)


class TestNCCApplyTranslationToMask:
    """NCC mask translation tests."""

    def test_binary_mask(self, registration_instance_fixture):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 150:250] = 1
        translated = registration_instance_fixture.apply_translation_to_mask(mask, 8, -6)
        assert isinstance(translated, np.ndarray)
        assert translated.shape == mask.shape
        unique = np.unique(translated)
        assert all(v in [0, 1] for v in unique)

    def test_labeled_mask(self, registration_instance_fixture):
        mask = np.zeros((480, 640), dtype=np.uint16)
        mask[50:100, 50:100] = 1
        mask[200:250, 200:250] = 2
        mask[300:350, 400:450] = 3
        translated = registration_instance_fixture.apply_translation_to_mask(mask, 5, 5)
        unique = np.unique(translated)
        assert all(v in [0, 1, 2, 3] for v in unique)

    def test_identity(self, registration_instance_fixture):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 150:250] = 1
        translated = registration_instance_fixture.apply_translation_to_mask(mask, 0, 0)
        np.testing.assert_array_equal(translated, mask)

    def test_tensor_input(self, registration_instance_fixture):
        mask = torch.zeros(480, 640)
        mask[100:200, 150:250] = 1
        translated = registration_instance_fixture.apply_translation_to_mask(
            mask, 8, -6, return_tensor=True
        )
        assert isinstance(translated, torch.Tensor)

    def test_invalid_mask_shape(self, registration_instance_fixture):
        invalid = torch.randn(3, 480, 640)
        with pytest.raises(ValueError, match="Unexpected mask shape"):
            registration_instance_fixture.apply_translation_to_mask(invalid, 5, 3)


# ===================================================================
# 6. Phase-specific tests
# ===================================================================


class TestPhaseApplyTranslation:
    """Phase correlation-specific apply_translation tests."""

    def test_grayscale(self, phase_registration_instance_fixture):
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (100, 100), 30, 255, -1)
        aligned = phase_registration_instance_fixture.apply_translation(image, 10, -15)
        assert aligned.shape == image.shape
        assert aligned.dtype == image.dtype

    def test_rgb(self, phase_registration_instance_fixture):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 30, (255, 128, 0), -1)
        aligned = phase_registration_instance_fixture.apply_translation(image, 5, 8)
        assert aligned.shape == image.shape
        assert aligned.dtype == image.dtype


class TestPhaseApplyTranslationToMask:
    """Phase correlation mask translation tests."""

    def test_binary_mask(self, phase_registration_instance_fixture):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 1
        translated = phase_registration_instance_fixture.apply_translation_to_mask(mask, 5, 5)
        assert translated.shape == mask.shape
        assert translated.dtype == mask.dtype
        unique = np.unique(translated)
        assert all(v in [0, 1] for v in unique)


class TestPhasePreprocessing:
    """Phase correlation preprocessing option tests."""

    def _create_low_contrast(self, size=(300, 300), positions=None):
        if positions is None:
            positions = [(100, 100), (200, 100), (100, 200), (200, 200)]
        image = np.ones(size, dtype=np.uint8) * 110
        for x, y in positions:
            cv2.circle(image, (x, y), 12, 130, -1)
        return image

    def test_with_clahe(self):
        positions = [(100, 100), (200, 100), (100, 200), (200, 200)]
        marker_group = create_marker_group_pixel({f"m{i}": pos for i, pos in enumerate(positions)})
        ref = self._create_low_contrast(positions=positions)
        tx, ty = 5, 8
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        target = cv2.warpAffine(ref, M, (300, 300))

        reg = PhaseCorrelationRegistration(marker_group, padding=50, preprocess=True)
        dx, dy, _conf = reg.compute_translation(ref, target)
        assert abs(dx - tx) < 1.5
        assert abs(dy - ty) < 1.5

    def test_without_clahe(self):
        positions = [(100, 100), (200, 100), (100, 200), (200, 200)]
        marker_group = create_marker_group_pixel({f"m{i}": pos for i, pos in enumerate(positions)})
        ref = self._create_low_contrast(positions=positions)
        tx, ty = 5, 8
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        target = cv2.warpAffine(ref, M, (300, 300))

        reg = PhaseCorrelationRegistration(marker_group, padding=50, preprocess=False)
        dx, _dy, _conf = reg.compute_translation(ref, target)
        assert isinstance(dx, float)

    def test_with_hanning(self):
        positions = [(150, 150), (250, 150), (150, 250), (250, 250)]
        marker_group = create_marker_group_pixel({f"m{i}": pos for i, pos in enumerate(positions)})
        ref = np.ones((400, 400), dtype=np.uint8) * 128
        for x, y in positions:
            cv2.circle(ref, (x, y), 15, 200, -1)

        tx, ty = 7, -6
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        target = cv2.warpAffine(ref, M, (400, 400))

        reg = PhaseCorrelationRegistration(marker_group, padding=50, use_hanning=True)
        dx, dy, _ = reg.compute_translation(ref, target)
        assert abs(dx - tx) < 1.0
        assert abs(dy - ty) < 1.0

    def test_without_hanning(self):
        positions = [(150, 150), (250, 150), (150, 250), (250, 250)]
        marker_group = create_marker_group_pixel({f"m{i}": pos for i, pos in enumerate(positions)})
        ref = np.ones((400, 400), dtype=np.uint8) * 128
        for x, y in positions:
            cv2.circle(ref, (x, y), 15, 200, -1)

        tx, ty = 7, -6
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        target = cv2.warpAffine(ref, M, (400, 400))

        reg = PhaseCorrelationRegistration(marker_group, padding=50, use_hanning=False)
        dx, _dy, _ = reg.compute_translation(ref, target)
        assert isinstance(dx, float)


# ===================================================================
# 7. Parametric & edge-case tests
# ===================================================================


class TestParametric:
    """Parametric tests for different configurations."""

    @pytest.mark.parametrize("padding", [10, 30, 50, 100])
    def test_padding_effect(self, marker_group_fixture, padding):
        reg = TimelapseRegistration(
            marker_group_pixel=marker_group_fixture,
            max_translation=20,
            padding=padding,
            device="cpu",
        )
        x_min, _y_min, x_max, _y_max = reg.marker_bbox
        assert (x_max - x_min) >= 2 * padding

    @pytest.mark.parametrize("max_translation", [5, 10, 20, 50])
    def test_max_translation_effect(self, marker_group_fixture, max_translation):
        reg = TimelapseRegistration(
            marker_group_pixel=marker_group_fixture,
            max_translation=max_translation,
            padding=50,
            device="cpu",
        )
        marker_positions = {"cross": (200, 200), "circle": (300, 250)}
        reference = create_synthetic_marker_image(640, 480, marker_positions)
        small_shift = max_translation - 2
        target = apply_known_translation(reference, small_shift, 0)
        dx, _dy, _ = reg.compute_translation(reference, target)
        assert abs(dx - small_shift) < 1.0


class TestEdgeCases:
    """Edge cases shared across methods."""

    def test_very_small_marker_region(self):
        marker_group = create_marker_group_pixel({"cross": (100, 100), "circle": (105, 105)})
        reg = TimelapseRegistration(
            marker_group_pixel=marker_group,
            max_translation=20,
            padding=10,
            device="cpu",
        )
        reference = create_synthetic_marker_image(
            300, 300, {"cross": (100, 100), "circle": (105, 105)}
        )
        target = apply_known_translation(reference, 3, 2)
        dx, dy, _ = reg.compute_translation(reference, target)
        assert abs(dx - 3) < 2.0
        assert abs(dy - 2) < 2.0

    def test_markers_at_image_edge(self):
        marker_group = create_marker_group_pixel({"cross": (10, 10), "circle": (590, 10)})
        reg = TimelapseRegistration(
            marker_group_pixel=marker_group,
            max_translation=20,
            padding=50,
            device="cpu",
        )
        image = create_synthetic_marker_image(600, 400, {"cross": (10, 10), "circle": (590, 10)})
        region = reg.extract_marker_region(image)
        assert region.shape[0] > 0
        assert region.shape[1] > 0
