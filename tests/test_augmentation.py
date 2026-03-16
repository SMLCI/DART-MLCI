"""Testcases for image augmentation and transformation consistency."""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dart_mlci
from dart_mlci import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dart_mlci.match import marker_group_to_pixel_coordinates, match_markers
from dart_mlci.rotation import compute_marker_group_angles
from dart_mlci.visualization import plot_markers_on_image, rotate_image_no_crop

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


class TestRotationAngleConsistency(unittest.TestCase):
    """Test that measured rotation angles are consistent when applying known rotations."""

    def test_rotation_angle_consistency(self):
        """Test rotation angle measurement consistency.

        This test:
        1. Loads an image and measures the initial rotation angle
        2. Applies multiple known rotations to the original image
        3. Re-measures the rotation angle after each rotation
        4. Verifies that the measured angle changes match the applied rotations
        """
        # Create subfolder for this test
        output_dir = TEST_RESULTS_DIR / "rotation_consistency"
        output_dir.mkdir(exist_ok=True)

        # Config
        pixel_size = 0.065789
        marker_group = {
            "cross": np.array((4, 8), dtype=float),
            "circle": np.array((56, 8), dtype=float),
        }
        marker_group_pixels = marker_group_to_pixel_coordinates(marker_group, pixel_size)

        # Load model with label mapping for new model format
        label_mapping = {"class_0": "cross", "class_1": "circle"}
        model = MarkerDetectionModel(
            DEFAULT_MODEL_PATH,
            label_mapping=label_mapping,
        )

        # Confidence threshold to filter out false positives
        confidence_threshold = 0.5

        def filter_markers_by_confidence(markers, threshold):
            return [m for m in markers if m.get("conf", 0.0) >= threshold]

        # Load original image
        import cv2

        original_image = cv2.imread(
            str(Path(dart_mlci.__file__).parent.parent / "artifacts/images/sak/0000.png")
        )

        # Measure initial angle
        markers = model.predict_markers(original_image)
        markers = filter_markers_by_confidence(markers, confidence_threshold)
        matched_indices = match_markers(markers, marker_group=marker_group_pixels, tolerance=60)

        self.assertGreater(len(matched_indices), 0, "No markers matched in original image")

        initial_angles = compute_marker_group_angles(markers, matched_indices, marker_group_pixels)
        initial_angle = np.mean(initial_angles)

        # Save original image with detected markers
        plot_markers_on_image(
            original_image,
            markers,
            matched_indices,
            title=f"Original Image (measured angle: {initial_angle:.2f}°)",
            output_path=output_dir / "00_original.png",
        )

        # Test rotations to apply (in degrees)
        test_rotations = [5, 10, 15, 20, 25, 30, 45, -5, -10, -15, -20, -25, -30, -45]

        applied_rotations = []
        measured_angle_changes = []
        skipped_rotations = []
        base_angle_tolerance = 3.0

        for i, rotation in enumerate(test_rotations):
            rotated_image = rotate_image_no_crop(original_image, rotation)

            rotated_markers = model.predict_markers(rotated_image)
            rotated_markers = filter_markers_by_confidence(rotated_markers, confidence_threshold)
            rotated_matched_indices = match_markers(
                rotated_markers, marker_group=marker_group_pixels, tolerance=60
            )

            if len(rotated_matched_indices) == 0:
                plot_markers_on_image(
                    rotated_image,
                    rotated_markers,
                    [],
                    title=f"Rotation {rotation}° - NO MATCHES",
                    output_path=output_dir / f"{i + 1:02d}_rot_{rotation:+d}_no_match.png",
                )
                skipped_rotations.append(rotation)
                continue

            rotated_angles = compute_marker_group_angles(
                rotated_markers, rotated_matched_indices, marker_group_pixels
            )
            rotated_angle = np.mean(rotated_angles)
            measured_change = rotated_angle - initial_angle

            applied_rotations.append(rotation)
            measured_angle_changes.append(measured_change)

            plot_markers_on_image(
                rotated_image,
                rotated_markers,
                rotated_matched_indices,
                title=f"Rotation {rotation}° | Measured: {rotated_angle:.2f}° | Change: {measured_change:.2f}°",
                output_path=output_dir / f"{i + 1:02d}_rot_{rotation:+d}.png",
            )

            error_direct = abs(measured_change - rotation)
            error_inverted = abs(measured_change + rotation)
            angle_error = min(error_direct, error_inverted)

            print(
                f"Rotation {rotation:+3d}°: change={measured_change:+.2f}°, error={angle_error:.2f}°"
            )

            self.assertLess(
                angle_error,
                base_angle_tolerance,
                f"Rotation of {rotation}° error={angle_error:.2f}° exceeds tolerance",
            )

        # Save summary plot
        if len(applied_rotations) > 0:
            _, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(applied_rotations, measured_angle_changes, s=100)
            axes[0].plot([-35, 35], [-35, 35], "r--", label="Perfect", alpha=0.5)
            axes[0].plot([-35, 35], [35, -35], "b--", label="Inverted", alpha=0.5)
            axes[0].set_xlabel("Applied Rotation (°)")
            axes[0].set_ylabel("Measured Change (°)")
            axes[0].set_title("Rotation Consistency")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            errors = [
                abs(abs(m) - abs(a))
                for a, m in zip(applied_rotations, measured_angle_changes, strict=True)
            ]
            axes[1].bar(range(len(errors)), errors)
            axes[1].set_xticks(range(len(errors)))
            axes[1].set_xticklabels([f"{r}°" for r in applied_rotations], rotation=45)
            axes[1].axhline(y=base_angle_tolerance, color="r", linestyle="--")
            axes[1].set_title("Measurement Error")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "rotation_consistency.png", dpi=150)
            plt.close()

        if skipped_rotations:
            print(f"Skipped rotations: {skipped_rotations}")

        self.assertGreaterEqual(
            len(applied_rotations), 10, f"Too few rotations tested: {skipped_rotations}"
        )


if __name__ == "__main__":
    unittest.main()
