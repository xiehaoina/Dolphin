from PIL import Image


from src.models.chat.chat_model import ChatModel
from src.pipelines.processor.processor import Processor
from typing import List



class DolphinTextProcessor(Processor):
    """
    Processor for performing layout analysis on a document image.

    This processor uses a ChatModel to infer the layout of a document, then processes the
    output to extract individual elements, their bounding boxes, and reading order.
    """

    def __init__(self, model: ChatModel):
        """
        Initializes the DolphinLayoutAnalysisProcessor.

        Args:
            model: A ChatModel instance used for layout inference.
        """
        self.model = model
        self.prompt = "Read text in the image."
    
    def batch_process(self, images: List[Image.Image]) -> List[str]:
        prompts = [self.prompt] * len(images)
        return  self.model.batch_inference(prompts, images)
 

    

    def process(self, image: Image.Image) -> str:
        """
        Processes a single image to perform layout analysis.

        This method takes a PIL image, sends it to the model for layout inference,
        and then post-processes the model's output to extract structured data.

        Args:
            image: The input PIL Image.

        Returns:
            A list of dictionaries, where each dictionary represents a detected element
            and contains its cropped image, label, bounding box, and reading order.
        """
        
        return  self.model.inference(self.prompt, image) 
