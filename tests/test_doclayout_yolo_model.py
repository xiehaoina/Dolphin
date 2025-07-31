import unittest
import sys
import os
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.layout.doclayout_yolo.DocLayoutYOLO import DocLayoutYOLOModel

class TestDocLayoutYOLOModel(unittest.TestCase):
    """
    Test suite for the DocLayoutYOLOModel.
    """

    def setUp(self):
        """Set up the test case by creating a model instance."""
        # Config for the model, based on test_factory.py
        self.config = OmegaConf.create({
            "device": "cpu",
            "weight": "./model_weight/Structure/layout_zh.pt"
        })

        # Instantiate the model directly for unit testing
        self.model_instance = DocLayoutYOLOModel(self.config)

    def test_process_inference(self):
        """Tests the process method for inference."""
        # Load the test image
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'bill.png')
        self.assertTrue(os.path.exists(image_path), f"Test image not found at {image_path}")
        image = Image.open(image_path).convert('RGB')

        # Call the model's process method
        layout_elements = self.model_instance.inference(image)

        # Assertions to validate the output structure
        self.assertIsInstance(layout_elements, list, "The output should be a list.")
        print(layout_elements)
        if layout_elements:
            element = layout_elements[0]
            self.assertIn('bbox', element)
            self.assertIn('label', element)
            self.assertIn('score', element)
            self.assertIsInstance(element['bbox'], list)
            self.assertEqual(len(element['bbox']), 4)

if __name__ == '__main__':
    unittest.main()