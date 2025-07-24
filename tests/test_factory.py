import unittest
import sys
import os
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.chat.factory import ChatModelFactory
from src.models.chat.dolphin.raw_dolphin_model import RawDolphinModel
from src.models.chat.dolphin.hf_dolphin_model import HFDolphinModel
from src.models.chat.gateway.gateway_model import GatewayModel

class TestChatModelFactory(unittest.TestCase):
    """
    Test suite for the ChatModelFactory.
    """

    def setUp(self):
        """Set up the test case."""
        self.factory = ChatModelFactory()
        # Since it's a singleton, clear the cache for each test
        self.factory._models = {}

    #@patch('src.models.chat.dolphin.raw_dolphin_model.RawDolphinModel.__init__', return_value=None)
    def test_create_raw_dolphin_model(self, mock_init = MagicMock()):
        """
        Tests the creation of a RawDolphinModel instance.
        """
        config = OmegaConf.load("./config/Dolphin.yaml")
        model = self.factory.create_model("raw_dolphin", config)
        self.assertIsInstance(model, RawDolphinModel)
        #mock_init.assert_called_once_with(config)

    #@patch('src.models.chat.dolphin.hf_dolphin_model.HFDolphinModel.__init__', return_value=None)
    def test_create_hf_dolphin_model(self, mock_init = MagicMock()):
        """
        Tests the creation of an HFDolphinModel instance.
        """
        config = OmegaConf.create({"model_id_or_path": "/Users/bytedance/git/ocr/Dolphin/hf_model"})
        model = self.factory.create_model("hf_dolphin", config)
        self.assertIsInstance(model, HFDolphinModel)
        #mock_init.assert_called_once_with(config)

    #@patch('src.models.chat.gateway.gateway_model.GatewayModel.__init__', return_value=None)
    def test_create_gateway_model(self, mock_init = MagicMock()) :
        """
        Tests the creation of a GatewayModel instance.
        """
        config = OmegaConf.create({"url": "https://ai-gateway.vei.volces.com/v1", 
                                 "model_name": "doubao-1.5-vision-pro", 
                                 "api_key": "sk-b187cc92d38040cbbf76839f2ea5980cag5bv9m2r4j17vah",
                                 "max_concurrency": 30})
        model = self.factory.create_model("gateway", config)
        self.assertIsInstance(model, GatewayModel)
        #mock_init.assert_called_once_with(config)

    def test_invalid_model_type(self):
        """
        Tests that creating an invalid model type raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.factory.create_model("invalid_type", {})

    def test_singleton_instance(self):
        """Tests that the factory is a singleton."""
        factory1 = ChatModelFactory()
        self.assertIs(factory1, self.factory)

    @patch('src.models.chat.dolphin.raw_dolphin_model.RawDolphinModel.__init__', return_value=None)
    def test_model_caching(self, mock_init):
        """Tests that models are cached within the factory."""
        config = {"some_config_key": "some_config_value"}
        model1 = self.factory.create_model("raw_dolphin", config)
        model2 = self.factory.create_model("raw_dolphin", config)
        self.assertIs(model1, model2)
        mock_init.assert_called_once()