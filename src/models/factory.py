import importlib
from typing import Any
from src.models.model import Model

class ModelFactory:
    """
    A singleton factory class for creating and managing different types of ChatModel instances.
    """
    _instance = None
    _models = {}
    _model_map = {
        "raw_dolphin": ("src.models.chat.dolphin.raw_dolphin_model", "RawDolphinModel"),
        "hf_dolphin": ("src.models.chat.dolphin.hf_dolphin_model", "HFDolphinModel"),
        "gateway": ("src.models.chat.gateway.gateway_model", "GatewayModel"),
        "doclayout_yolo": ("src.models.layout.doclayout_yolo.DocLayoutYOLO", "DocLayoutYOLOModel"),
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def create_model(self, model_type: str, config: Any) -> Model:
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
            if model_type in self._model_map:
                module_name, class_name = self._model_map[model_type]
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                self._models[model_type] = model_class(config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        return self._models[model_type]