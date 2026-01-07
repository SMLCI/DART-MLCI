"""Testcases for mask shape handling."""

import unittest

import numpy as np
from shapely.geometry import Point, Polygon

from dmc_masking.mask import RoIPolygon


def build_polygon():
    chamber_width = 60
    chamber_height = 60

    interior = Point(30, 30).buffer(5)

    chamber_polygon = Polygon(
        [
            (0, 0),
            (chamber_width, 0),
            (chamber_width, chamber_height),
            (0, chamber_height),
        ]
    ).difference(interior)

    return RoIPolygon(chamber_polygon)


polygon_area = 60 * 60 - np.pi * 5**2


class TestMask(unittest.TestCase):
    """Test case for RoI mask shapes."""

    def test_area(self):
        """test area computation"""

        self.assertLess(np.abs(build_polygon().area - polygon_area), 1)

    @staticmethod
    def test_translation():
        """test translation operation"""

        roi_polygon = build_polygon()

        np.testing.assert_almost_equal(roi_polygon.center, np.array([30, 30]))

        new_roi_polygon = roi_polygon.translate(15, -15)

        np.testing.assert_almost_equal(roi_polygon.center, np.array([30, 30]))
        np.testing.assert_almost_equal(new_roi_polygon.center, np.array([30 + 15, 30 - 15]))

    def test_scaling(self):
        """test scaling operation"""

        roi_polygon = build_polygon()

        self.assertLess(np.abs(roi_polygon.area - polygon_area), 0.3)

        new_roi_polygon = roi_polygon.scale(2)

        self.assertLess(np.abs(roi_polygon.area - polygon_area), 0.3)
        self.assertLess(np.abs(new_roi_polygon.area - 4 * polygon_area), 0.6)

    def test_difference(self):
        """test difference operation."""
        A = build_polygon()
        B = A.copy()
        zeroish = A.difference(B)

        self.assertLess(np.abs(A.area - polygon_area), 0.3)
        self.assertLess(np.abs(B.area - polygon_area), 0.3)
        self.assertLess(np.abs(zeroish.area), 0.1)

    def test_union(self):
        """test union operation."""
        A = build_polygon().translate(x=60)
        B = build_polygon()

        both = A.union(B)

        self.assertLess(np.abs(A.area - polygon_area), 0.3)
        self.assertLess(np.abs(B.area - polygon_area), 0.3)
        self.assertLess(np.abs(both.area - 2 * polygon_area), 0.3)


if __name__ == "__main__":
    unittest.main()
