import unittest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from omegaconf import OmegaConf
from PIL import Image
import httpx

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.chat.gateway.gateway_model import GatewayModel

class TestGatewayModel(unittest.TestCase):
    """
    Test suite for the GatewayModel.
    """

    def setUp(self):
        """Set up the test case."""
        self.config = OmegaConf.create({"url": "https://ai-gateway.vei.volces.com/v1", 
                                 "model_name": "doubao-1.5-vision-pro", 
                                 "api_key": "sk-b187cc92d38040cbbf76839f2ea5980cag5bv9m2r4j17vah",
                                 "max_concurrency": 30})

    #@patch('src.models.chat.gateway.gateway_model.OpenAI')
    def test_initialization_success(self, mock_openai = MagicMock()):
        """Tests successful initialization of the GatewayModel."""
        mock_client = mock_openai.return_value
        mock_client.models.list.return_value = MagicMock()  # Simulate successful validation
        model = GatewayModel(self.config)
        self.assertEqual(model.model_name, self.config.model_name)
        self.assertEqual(model.base_url, self.config.url)
        #mock_openai.assert_called_once()
        #model.client.models.list.assert_called_once()



    #@patch('src.models.chat.gateway.gateway_model.GatewayModel.async_batch_inference')
    def test_batch_inference(self, mock_async_batch = AsyncMock()):
        """Tests the batch_inference method."""
        mock_async_batch.return_value = ["result1", "result2"]
        
        with patch.object(GatewayModel, 'validate', return_value=None):
            model = GatewayModel(self.config)
            prompts = ["p1", "p2"]
            images = [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))]
            results = model.batch_inference(prompts, images)      
            print(results)
            #mock_async_batch.assert_called_once()
            
    #@patch('src.models.chat.gateway.gateway_model.GatewayModel.async_batch_inference')
    def test_inference(self, mock_async_batch = AsyncMock()):
        """Tests the inference method."""
        
        with patch.object(GatewayModel, 'validate', return_value=None):
            model = GatewayModel(self.config)
            prompt = "p1"
            image = Image.new('RGB', (100, 100))
            result = model.inference(prompt, image)      
            print(result)
            #mock_async_batch.assert_called_once()

   

  

if __name__ == '__main__':
    # To run async tests
    unittest.main()