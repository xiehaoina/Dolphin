import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from PIL import Image
import torch

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.chat.dolphin.hf_dolphin_model import HFDolphinModel

class TestHFDolphinModel(unittest.TestCase):
    """
    Test suite for the HFDolphinModel.
    """

    def setUp(self):
        """Set up the test case by mocking the model and processor loading."""
        # Config
        self.config = OmegaConf.create({
            "model_id_or_path": "/Users/bytedance/git/ocr/Dolphin/hf_model"
        })

        # Instantiate the model
        self.model_instance = HFDolphinModel(self.config)



    def test_batch_inference(self):
        """Tests the batch_inference method."""
        # Prepare mock inputs and outputs
        prompts = ["prompt1", "prompt2"]
        images = [Image.new('RGB', (10, 10)), Image.new('RGB', (10, 10))]
        full_prompts = ["<s>prompt1 <Answer/>", "<s>prompt2 <Answer/>"]
        
        # Call the method
        results = self.model_instance.batch_inference(full_prompts, images)
        print(results)

if __name__ == '__main__':
    unittest.main()