from abc import ABC, abstractmethod
from typing import Any, List
from src.models.model import Model
from PIL import Image

class LayoutModel(Model):
    """
    Abstract base class for a multi-modal Vision Language Model (VLM).
    
    This class defines the standard interface for interacting with different
    VLM implementations. Subclasses must implement the abstract methods
    defined here to provide specific model functionality.
    """
    @abstractmethod
    def __init__(self, config: Any) -> None:
        """
        Initializes the chat model with a given configuration.

        Args:
            config: A configuration object containing model-specific settings,
                    such as model paths, API keys, or other parameters.
        """

    @abstractmethod
    def validate(self) -> None:
        """
        Validates the model and its configuration.

        This method should check if the model is loaded correctly, if the
        necessary credentials are in place, and if the configuration is valid.
        It should raise an exception if validation fails.
        """
        pass

    @abstractmethod
    def inference(self,  image: Image.Image) -> dict:
        """
        Performs inference on a single prompt and image.

        Args:
            prompt: The text prompt to guide the model's generation.
            image: A PIL Image object to be processed by the model.

        Returns:
            The generated text output from the model as a string.
        """
        pass

    @abstractmethod
    def batch_inference(self, images: List[Image.Image]) -> List[dict]:
        """
        Performs inference on a batch of prompts and images.

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Image objects.

        Returns:
            A list of generated text outputs from the model.
        """
        pass
    

