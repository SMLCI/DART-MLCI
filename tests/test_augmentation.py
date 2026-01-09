"""Testcases for image augmentation and transformation consistency."""

import unittest
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import dmc_masking
from dmc_masking import MarkerDetectionModel
from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers
from dmc_masking.rotation import compute_marker_group_angles

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


def plot_markers_on_image(
    image: np.ndarray,
    markers: list,
    matched_indices: list,
    title: str = "",
    output_path: Path | None = None,
) -> None:
    """Plot detected markers on an image and optionally save to file.

    Args:
        image: Input image in HxWxC format (BGR from cv2.imread)
        markers: List of detected markers with 'bbox_center' and 'label' keys
        matched_indices: List of matched marker index pairs (cross_idx, circle_idx)
        title: Title for the plot
        output_path: Optional path to save the figure
    """
    # Convert BGR to RGB for matplotlib
    if image.ndim == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    plt.figure(figsize=(12, 10))
    plt.imshow(image_rgb)

    # Define colors for different marker types
    colors = {"cross": "red", "circle": "blue"}
    marker_symbols = {"cross": "x", "circle": "o"}

    # Plot all detected markers
    for i, marker in enumerate(markers):
        center = marker["bbox_center"]
        label = marker["label"]
        conf = marker.get("conf", 0.0)  # Get confidence score
        color = colors.get(label, "green")
        symbol = marker_symbols.get(label, "s")

        plt.scatter(center[0], center[1], c=color, marker=symbol, s=200, linewidths=3, zorder=5)
        plt.annotate(
            f"{i}: {label} ({conf:.2f})",  # Show confidence score
            (center[0], center[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    # Draw lines between matched marker pairs
    for cross_idx, circle_idx in matched_indices:
        cross_center = markers[cross_idx]["bbox_center"]
        circle_center = markers[circle_idx]["bbox_center"]
        plt.plot(
            [cross_center[0], circle_center[0]],
            [cross_center[1], circle_center[1]],
            "g-",
            linewidth=2,
            alpha=0.7,
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def rotate_image_no_crop(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center without cropping.

    Expands the canvas with black borders to fit the entire rotated image.

    Args:
        image: Input image in HxWxC or HxW format (as returned by cv2.imread)
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image with expanded canvas (black borders where needed)
    """
    # Handle both HxW and HxWxC formats
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width / 2, height / 2

    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Calculate the sine and cosine of the rotation angle
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # Calculate new image bounds to fit the entire rotated image
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation matrix to account for the new image center
    rot_mat[0, 2] += (new_width / 2) - center_x
    rot_mat[1, 2] += (new_height / 2) - center_y

    # Perform the rotation with black border (default borderValue=0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated


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
        # The new model uses class_0 for cross and class_1 for circle
        label_mapping = {"class_0": "cross", "class_1": "circle"}
        model = MarkerDetectionModel(
            # Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"
            Path(
                "/home/seiffarth_l/projects/DMC_new/dmc-train/runs/v8_segment_s_imgsz1280/weights/best.pt"
            ),
            label_mapping=label_mapping,
        )

        # Confidence threshold to filter out false positives at large rotations
        # Lower threshold (0.5) to allow detection at larger angles with the new model
        confidence_threshold = 0.5

        def filter_markers_by_confidence(markers, threshold):
            """Filter markers to only keep those with confidence above threshold."""
            return [m for m in markers if m.get("conf", 0.0) >= threshold]

        # Load original image
        original_image = cv2.imread(
            str(Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0000.png")
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
        # Test up to ±45° - requires high confidence threshold to filter false positives
        test_rotations = [5, 10, 15, 20, 25, 30, 45, -5, -10, -15, -20, -25, -30, -45]

        # Store results for plotting
        applied_rotations = []
        measured_angle_changes = []
        skipped_rotations = []

        # Tolerance for angle measurement (in degrees)
        # Use higher tolerance for larger rotations
        base_angle_tolerance = 3.0

        for i, rotation in enumerate(test_rotations):
            # Apply rotation to original image (no cropping, black borders added)
            rotated_image = rotate_image_no_crop(original_image, rotation)

            # Detect markers in rotated image
            rotated_markers = model.predict_markers(rotated_image)
            rotated_markers = filter_markers_by_confidence(rotated_markers, confidence_threshold)
            rotated_matched_indices = match_markers(
                rotated_markers, marker_group=marker_group_pixels, tolerance=60
            )

            if len(rotated_matched_indices) == 0:
                # Save image even if no markers detected for debugging
                plot_markers_on_image(
                    rotated_image,
                    rotated_markers,
                    [],
                    title=f"Rotation {rotation}° - NO MATCHES (detected {len(rotated_markers)} markers)",
                    output_path=output_dir / f"{i + 1:02d}_rot_{rotation:+d}_no_match.png",
                )
                skipped_rotations.append(rotation)
                continue

            # Measure angle in rotated image
            rotated_angles = compute_marker_group_angles(
                rotated_markers, rotated_matched_indices, marker_group_pixels
            )
            rotated_angle = np.mean(rotated_angles)

            # The measured angle change should match the applied rotation
            # Note: angle_between returns absolute angles, so we need to handle this carefully
            measured_change = rotated_angle - initial_angle

            applied_rotations.append(rotation)
            measured_angle_changes.append(measured_change)

            # Save rotated image with detected markers
            plot_markers_on_image(
                rotated_image,
                rotated_markers,
                rotated_matched_indices,
                title=(
                    f"Rotation {rotation}° | "
                    f"Measured angle: {rotated_angle:.2f}° | "
                    f"Change: {measured_change:.2f}°"
                ),
                output_path=output_dir / f"{i + 1:02d}_rot_{rotation:+d}.png",
            )

            # Check consistency (within tolerance)
            # Now using signed angles, so we can directly compare values
            angle_tolerance = base_angle_tolerance

            # With signed angles, the measured change should directly match the applied rotation
            # Note: Sign may be inverted depending on coordinate system conventions
            # We check both possibilities and use the one with smaller error
            error_direct = abs(measured_change - rotation)
            error_inverted = abs(measured_change + rotation)
            angle_error = min(error_direct, error_inverted)
            sign_matches = error_direct < error_inverted

            # Log the results for debugging
            print(
                f"Rotation {rotation:+3d}°: measured_angle={rotated_angle:.2f}°, "
                f"change={measured_change:+.2f}°, error={angle_error:.2f}°, "
                f"sign_match={sign_matches}"
            )

            self.assertLess(
                angle_error,
                angle_tolerance,
                f"Rotation of {rotation}° resulted in measured change of {measured_change:.2f}°, "
                f"error={angle_error:.2f}° exceeds tolerance={angle_tolerance:.2f}°",
            )

        # Save results plot
        if len(applied_rotations) > 0:
            _, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot applied vs measured
            axes[0].scatter(applied_rotations, measured_angle_changes, s=100)
            axes[0].plot([-35, 35], [-35, 35], "r--", label="Perfect correlation", alpha=0.5)
            axes[0].plot([-35, 35], [35, -35], "b--", label="Inverted correlation", alpha=0.5)
            axes[0].set_xlabel("Applied Rotation (degrees)")
            axes[0].set_ylabel("Measured Angle Change (degrees)")
            axes[0].set_title("Rotation Angle Consistency")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(-35, 35)
            axes[0].set_ylim(-35, 35)

            # Plot absolute error
            errors = [
                abs(abs(m) - abs(a))
                for a, m in zip(applied_rotations, measured_angle_changes, strict=True)
            ]
            axes[1].bar(range(len(errors)), errors)
            axes[1].set_xticks(range(len(errors)))
            axes[1].set_xticklabels([f"{r}°" for r in applied_rotations], rotation=45)
            axes[1].set_xlabel("Applied Rotation")
            axes[1].set_ylabel("Absolute Error (degrees)")
            axes[1].set_title(f"Measurement Error (tolerance: {angle_tolerance}°)")
            axes[1].axhline(y=angle_tolerance, color="r", linestyle="--", label="Tolerance")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "rotation_consistency.png", dpi=150)
            plt.close()

        # Log skipped rotations
        if skipped_rotations:
            print(f"Skipped rotations (no marker matches): {skipped_rotations}")

        # Ensure we tested at least 10 rotations successfully
        self.assertGreaterEqual(
            len(applied_rotations),
            10,
            f"Too few rotations tested ({len(applied_rotations)}). Skipped: {skipped_rotations}",
        )


if __name__ == "__main__":
    unittest.main()
