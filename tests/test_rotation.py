""" Testcases for rotation """

import unittest

import numpy as np

from dmc_masking.rotation import angle_between, rotate_point


class TestAngleComputation(unittest.TestCase):
    """testcase for angle computation"""

    @staticmethod
    def test_angle_0():
        """test 0 degree angel"""
        v1 = np.array([0, 1])
        v2 = np.array([0, 2])

        np.testing.assert_almost_equal(angle_between(v1, v2), 0)

    @staticmethod
    def test_angle_90():
        """test 90 degree angle"""
        v1 = np.array([0, 1])
        v2 = np.array([1, 0])

        np.testing.assert_almost_equal(angle_between(v1, v2), np.pi / 2 * 57.29578)
        # np.testing.assert_almost_equal(angle_between(v2, v1), -np.pi/2)

    @staticmethod
    def test_angle_180():
        """test 180 degree angle"""
        v1 = np.array([0, 1])
        v2 = np.array([0, -1])

        np.testing.assert_almost_equal(angle_between(v1, v2), np.pi * 57.29578)


class TestPointRotation(unittest.TestCase):
    """testcase for rotating points."""

    @staticmethod
    def test_angle_90():
        """testing 90 degree rotation"""

        origin = np.array([0.0, 0.0])
        point = np.array([1, 1])

        np.testing.assert_almost_equal(
            rotate_point(point, origin, 90), np.array([1, -1])
        )

        origin = np.array([10, 0.0])
        point = np.array([0, 0])
        np.testing.assert_almost_equal(
            rotate_point(point, origin, 90), np.array([10, 10])
        )

    @staticmethod
    def test_angle_180():
        """testing 180 degree rotation"""

        origin = np.array([0.0, 0.0])
        point = np.array([1, 1])

        np.testing.assert_almost_equal(
            rotate_point(point, origin, 180), np.array([-1, -1])
        )

        origin = np.array([10, 0.0])
        point = np.array([0, 0])
        np.testing.assert_almost_equal(
            rotate_point(point, origin, 180), np.array([20, 0])
        )


if __name__ == "__main__":
    unittest.main()
