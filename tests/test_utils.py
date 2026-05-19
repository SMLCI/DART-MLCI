"""Testcases for utils."""

import unittest

import numpy as np
import pytest

from dart_mlci.utils import center_of_mask_mass, homogenize_image_size, to_hwc_numpy


class TestUtils(unittest.TestCase):
    """Test case for utils."""

    @staticmethod
    def test_homogenize_image_size():
        """test array dimension homogeneization"""

        test_sequence = [
            np.zeros((10, 10, 2), dtype=np.uint8),
            np.zeros((100, 100, 2), dtype=np.uint8),
            np.zeros((512, 512, 2), dtype=np.uint8),
        ]

        output_stack = homogenize_image_size(test_sequence)

        np.testing.assert_array_equal(output_stack.shape, (3, 512, 512, 2))

    def test_homogenize_image_size_2(self):
        """test array dimension homogeneization - fail: inconsistent number of channels"""

        with self.assertRaises(ValueError) as e:
            test_sequence = [
                np.zeros((10, 10, 3), dtype=np.uint8),
                np.zeros((100, 100, 2), dtype=np.uint8),
                np.zeros((512, 512, 2), dtype=np.uint8),
            ]

            _ = homogenize_image_size(test_sequence)

        print("Inhomogeneous number of channels" in e.exception.args[0])


class TestToHwcNumpy:
    def test_chw_is_transposed(self):
        chw = np.zeros((3, 8, 16), dtype=np.uint8)
        chw[0] = 5
        out = to_hwc_numpy(chw)
        assert out.shape == (8, 16, 3)
        assert (out[..., 0] == 5).all()

    def test_hwc_is_passed_through(self):
        hwc = np.zeros((8, 16, 3), dtype=np.uint8)
        out = to_hwc_numpy(hwc)
        assert out.shape == (8, 16, 3)
        assert out is hwc or np.shares_memory(out, hwc)

    def test_2d_is_passed_through(self):
        gray = np.zeros((8, 16), dtype=np.uint8)
        out = to_hwc_numpy(gray)
        assert out.shape == (8, 16)

    def test_torch_tensor_converted(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        t = torch.zeros((3, 8, 16), dtype=torch.uint8)
        out = to_hwc_numpy(t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8, 16, 3)


class TestCenterOfMaskMass:
    def test_centered_blob(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        x, y = center_of_mask_mass(mask)
        assert 8 <= x <= 11
        assert 8 <= y <= 11


if __name__ == "__main__":
    unittest.main()
