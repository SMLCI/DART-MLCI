""" Testcases for mask shape handling. """

import unittest
from pathlib import Path

import numpy as np

import dmc_masking
from dmc_masking.mask import SAKRoIStructureLibrary


class TestMask(unittest.TestCase):
    """Test case for RoI mask shapes."""

    def test_area(self):
        """test area computation"""

        # load the sak library
        sakl = SAKRoIStructureLibrary(
            Path(dmc_masking.__file__).parent.parent
            / "artifacts/chamber_structure.json",
            1,
        )

        sn, sp, _ = sakl("0000")

        self.assertEqual(sn, "NormaleBox-inner")
        self.assertEqual(sp.area, 60 * 60)

        sn, sp, _ = sakl("0100")

        self.assertEqual(sn, "BigBox-inner")
        self.assertEqual(sp.area, 60 * 100)

        sn, sp, _ = sakl("0200")

        self.assertEqual(sn, "OpenBox-inner")
        self.assertEqual(sp.area, 60 * 80)

        sn, sp, _ = sakl("0300")

        self.assertEqual(sn, "Mothermachine-inner")
        np.testing.assert_almost_equal(sp.area, 1378.28, decimal=1)

        sn, sp, _ = sakl("1000")

        self.assertEqual(sn, "NormaleBox-pillar-inner")
        np.testing.assert_almost_equal(sp.area, 3521.55, decimal=1)

        sn, sp, _ = sakl("1100")

        self.assertEqual(sn, "BigBox-pillar-inner")
        np.testing.assert_almost_equal(sp.area, 5843.11, decimal=1)

        sn, sp, _ = sakl("1200")

        self.assertEqual(sn, "OpenBox-collector-inner")
        np.testing.assert_almost_equal(sp.area, 4509.11, decimal=1)

        sn, sp, _ = sakl("1300")

        self.assertEqual(sn, "Mothermachine-2x-inner")
        np.testing.assert_almost_equal(sp.area, 1298.59, decimal=1)


if __name__ == "__main__":
    unittest.main()
