"""Regression checks for the recovered 39-class PlantVillage mapping.

Set PLANT_TEST_IMAGES_DIR to run the fixture inventory check against the
companion project's images. Inference accuracy is intentionally not asserted:
the TensorFlow model and the companion PyTorch model are different artifacts.
"""

import os
import unittest
from pathlib import Path

from app.services.prediction_service import CLASS_NAMES, determine_status


class ModelMappingTests(unittest.TestCase):
    def test_mapping_has_all_39_verified_classes(self):
        self.assertEqual(39, len(CLASS_NAMES))
        self.assertEqual("Background without leaves", CLASS_NAMES[4])
        self.assertEqual("Tomato healthy", CLASS_NAMES[38])

    def test_reference_fixture_directory_has_category_coverage(self):
        directory = os.getenv("PLANT_TEST_IMAGES_DIR")
        if not directory:
            self.skipTest("Set PLANT_TEST_IMAGES_DIR to validate the external test-image fixtures.")
        images = [path for path in Path(directory).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.assertGreaterEqual(len(images), 39)

    def test_prediction_status_precedence(self):
        self.assertEqual("healthy", determine_status(20, "Bell pepper healthy", 92, 60))
        self.assertEqual("diseased", determine_status(31, "Tomato Late blight", 95, 60))
        self.assertEqual("uncertain", determine_status(20, "Bell pepper healthy", 38, 60))
        self.assertEqual("no_leaf", determine_status(4, "Background without leaves", 20, 60))
        self.assertEqual("no_leaf", determine_status(4, "Background without leaves", 98, 60))


if __name__ == "__main__":
    unittest.main()
