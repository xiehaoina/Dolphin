from typing import Any

from src.models.chat.dolphin.raw_dolphin_model import RawDolphinModel
from src.models.chat.dolphin.hf_dolphin_model import HFDolphinModel
from src.models.chat.gateway.gateway_model import GatewayModel
from src.models.chat.chat_model import ChatModel

class ChatModelFactory:
    """
    A singleton factory class for creating and managing different types of ChatModel instances.
    """
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatModelFactory, cls).__new__(cls)
        return cls._instance

    def create_model(self, model_type: str, config: Any) -> ChatModel:
        """
        Creates and returns a ChatModel instance based on the specified type.
        It ensures that only one instance of each model type is created.

        Args:
            model_type: A string indicating the type of model to create (e.g., "raw_dolphin", "hf_dolphin", "gateway").
            config: A configuration object specific to the model type. Used only on the first creation.

        Returns:
            An instance of a class inheriting from ChatModel.

        Raises:
            ValueError: If an unsupported model_type is provided.
        """
        if model_type not in self._models:
            if model_type == "raw_dolphin":
                self._models[model_type] = RawDolphinModel(config)
            elif model_type == "hf_dolphin":
                self._models[model_type] = HFDolphinModel(config)
            elif model_type == "gateway":
                self._models[model_type] = GatewayModel(config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        return self._models[model_type]