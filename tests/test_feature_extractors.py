from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fractal_extrema import (  # noqa: E402
    extract_advanced_feature_dict,
    extract_feature_dict,
    extract_multizone_feature_dict,
)
from fractal_extrema.component_tree import (  # noqa: E402
    extract_component_tree_feature_dict,
)


class FeatureExtractorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        size = 256
        rows, columns = np.ogrid[:size, :size]
        cls.mask = (
            ((columns - 128.0) / 54.0) ** 2
            + ((rows - 128.0) / 42.0) ** 2
            <= 1.0
        )
        cls.image = (
            0.35
            + 0.25 * columns / size
            + 0.12 * np.sin(rows / 11.0)
            + 0.08 * np.cos(columns / 9.0)
        )

    def assert_finite_mapping(self, values: dict[str, float]) -> None:
        self.assertTrue(values)
        self.assertTrue(np.isfinite(np.asarray(list(values.values()))).all())

    def test_reference_features(self) -> None:
        self.assert_finite_mapping(extract_feature_dict(self.image, self.mask))

    def test_multizone_features(self) -> None:
        self.assert_finite_mapping(
            extract_multizone_feature_dict(self.image, self.mask)
        )

    def test_scale_bus_features(self) -> None:
        values = extract_advanced_feature_dict(self.image, self.mask)
        self.assertEqual(len(values), 76)
        self.assert_finite_mapping(values)

    def test_component_tree_features(self) -> None:
        values = extract_component_tree_feature_dict(self.image, self.mask)
        self.assertEqual(len(values), 54)
        self.assert_finite_mapping(values)


if __name__ == "__main__":
    unittest.main()
