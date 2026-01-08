"""Testcases for full masking pipeline"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

import dmc_masking
from dmc_masking import SingleStructureRoIMasker

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


class TestFullPipeline(unittest.TestCase):
    """Test cases for full masking pipeline"""

    def test_ssrm(self):
        """test cropping of an image stack with a single structure masker"""
        # Create subfolder for this test
        output_dir = TEST_RESULTS_DIR / "stack_ssrm"
        output_dir.mkdir(exist_ok=True)

        ssrm = SingleStructureRoIMasker()

        image_path = Path(dmc_masking.__file__).parent.parent / "artifacts/images/image_stack.tif"

        image = tifffile.imread(image_path)

        image_stack = image[:200]

        cropped_images, _ = ssrm(image_stack=image_stack, roi_id="0010")

        # self.assertEqual(len(cropped_images), 200)

        cropped_images = np.moveaxis(cropped_images, [1, 2, 3], [3, 1, 2])

        # plot
        _, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].imshow(cropped_images[0, ..., 0])
        axes[1].imshow(cropped_images[1, ..., 0])
        axes[2].imshow(cropped_images[2, ..., 0])

        plt.tight_layout()

        plt.savefig(output_dir / "result.jpg")


if __name__ == "__main__":
    unittest.main()
