"""
Regression tests to ensure calibration behavior is preserved during refactoring.
Run BEFORE and AFTER refactoring - outputs must be identical.
"""

import json
import unittest
from pathlib import Path

import numpy as np


class TestCalibrationRegression(unittest.TestCase):
    """End-to-end calibration regression tests."""

    @classmethod
    def setUpClass(cls):
        """Run calibration once and store results."""
        from scripts.calibrate_map import calibrate_map

        cls.config_path = Path("scripts/calibration_test.json")
        cls.golden_dir = Path("tests/golden")

        # Skip if config doesn't exist (CI environment without test data)
        if not cls.config_path.exists():
            cls.result = None
            cls.blueprint_map = None
            return

        # Run calibration with current code
        cls.result, cls.blueprint_map = calibrate_map(
            config=cls.config_path,
            verbose=False,
        )

    def setUp(self):
        if self.result is None:
            self.skipTest("Calibration test config not available")

    def test_calibration_succeeds(self):
        """Calibration should complete without errors."""
        self.assertIsNotNone(self.result)
        self.assertIsNotNone(self.result.calibrated_map)

    def test_calibrated_positions_match_golden(self):
        """Calibrated positions must match stored golden values."""
        golden_file = self.golden_dir / "calibration_positions.json"

        if not golden_file.exists():
            self.skipTest("Golden file not found - run generate_golden_files.py first")

        with open(golden_file) as f:
            golden = json.load(f)

        for roi_id, expected in golden.items():
            actual = self.result.calibrated_map.roi_positions[roi_id].position
            np.testing.assert_array_almost_equal(
                actual,
                expected["position"],
                decimal=2,  # 2 decimal places = 0.01 micron tolerance
                err_msg=f"Position mismatch for {roi_id}",
            )

    def test_transform_parameters_stable(self):
        """Affine transform parameters must not change."""
        golden_file = self.golden_dir / "transform_params.json"

        if not golden_file.exists():
            self.skipTest("Golden file not found - run generate_golden_files.py first")

        with open(golden_file) as f:
            golden = json.load(f)

        # Reconstruct current matrix from transform function
        transform_fn = self.result.transform_result.transform
        origin = np.array([0.0, 0.0])
        x_unit = np.array([1.0, 0.0])
        y_unit = np.array([0.0, 1.0])

        t_origin = transform_fn(origin)
        t_x = transform_fn(x_unit)
        t_y = transform_fn(y_unit)

        a, c = t_x - t_origin
        b, d = t_y - t_origin
        tx, ty = t_origin

        current_matrix = [[float(a), float(b), float(tx)], [float(c), float(d), float(ty)]]

        np.testing.assert_array_almost_equal(current_matrix, golden["matrix"], decimal=2)
        self.assertAlmostEqual(self.result.transform_result.rmse, golden["rmse"], places=4)

    def test_rmse_within_threshold(self):
        """RMSE should be reasonable (not regressed)."""
        self.assertLess(self.result.transform_result.rmse, 50.0)  # microns

    def test_all_expected_rois_calibrated(self):
        """All ROIs from blueprint should be in calibrated map."""
        blueprint_ids = set(self.blueprint_map.roi_positions.keys())
        calibrated_ids = set(self.result.calibrated_map.roi_positions.keys())
        self.assertEqual(blueprint_ids, calibrated_ids)

    def test_image_results_match_golden(self):
        """Individual image results should match golden values."""
        golden_file = self.golden_dir / "image_results.json"

        if not golden_file.exists():
            self.skipTest("Golden file not found - run generate_golden_files.py first")

        with open(golden_file) as f:
            golden = json.load(f)

        for img_result in self.result.image_results:
            roi_id = img_result.roi_id
            expected = golden.get(roi_id)

            if expected is None:
                self.fail(f"Missing golden data for {roi_id}")

            if img_result.success:
                self.assertIn("microscope_position", expected, f"{roi_id} should have succeeded")
                np.testing.assert_array_almost_equal(
                    img_result.microscope_position,
                    expected["microscope_position"],
                    decimal=2,
                    err_msg=f"Microscope position mismatch for {roi_id}",
                )
            else:
                self.assertIn("success", expected, f"{roi_id} should have failed")
                self.assertFalse(expected["success"])


class TestCalibrationRoundTrip(unittest.TestCase):
    """Test that transforms are mathematically consistent."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        from dmc_masking.map import Map, RoIPosition

        # Create a simple test map with known positions
        positions = [
            RoIPosition("0000", np.array([0.0, 0.0])),
            RoIPosition("0001", np.array([100.0, 0.0])),
            RoIPosition("0002", np.array([0.0, 100.0])),
            RoIPosition("0003", np.array([100.0, 100.0])),
        ]
        cls.source_map = Map(positions)

        # Create a target map with transformed positions (translation + scale)
        transformed_positions = [
            RoIPosition("0000", np.array([50.0, 50.0])),
            RoIPosition("0001", np.array([250.0, 50.0])),  # 2x scale
            RoIPosition("0002", np.array([50.0, 250.0])),
            RoIPosition("0003", np.array([250.0, 250.0])),
        ]
        cls.target_map = Map(transformed_positions)

    def test_transform_roundtrip_identity(self):
        """Transform from source to source should be identity."""
        result = self.source_map.compute_affine_transform(self.source_map)

        # Transform should have zero error
        self.assertAlmostEqual(result.rmse, 0.0, places=6)

        # Applying transform should give same positions
        transformed = self.source_map.apply_transform(result)
        for roi_id in self.source_map.roi_positions:
            np.testing.assert_array_almost_equal(
                transformed.roi_positions[roi_id].position,
                self.source_map.roi_positions[roi_id].position,
                decimal=6,
            )

    def test_transform_known_scale_translation(self):
        """Test transform with known scale and translation."""
        result = self.source_map.compute_affine_transform(self.target_map)

        # Should have zero error (exact correspondence)
        self.assertAlmostEqual(result.rmse, 0.0, places=6)

        # Apply to source and check against target
        transformed = self.source_map.apply_transform(result)
        for roi_id in self.target_map.roi_positions:
            np.testing.assert_array_almost_equal(
                transformed.roi_positions[roi_id].position,
                self.target_map.roi_positions[roi_id].position,
                decimal=6,
            )


if __name__ == "__main__":
    unittest.main()
