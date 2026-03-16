"""Testcases for utils."""

import unittest

import numpy as np

from dart_mlci.utils import homogenize_image_size


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


if __name__ == "__main__":
    unittest.main()
