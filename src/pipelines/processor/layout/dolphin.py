import cv2
import re
from PIL import Image
from typing import Any, List, Dict
from loguru import logger

from src.models.chat.chat_model import ChatModel
from src.pipelines.processor.processor import Processor
from src.utils.enums.doc_element_type import SpanType
from src.utils.utils import (
    process_coordinates,
    prepare_image,
)
class DolphinLayoutProcessor(Processor):
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
    
    def batch_process(self, images: List[Image.Image]) -> List[List[Dict]]:
        raise NotImplementedError("Batch processing is not implemented for this processor.")
    def process(self, image: Image.Image) -> List[Dict]:
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
        # Inference the layout from the image
        layout_str = self.model.inference(
            "Parse the reading order of this document.", image
        )
        # Prepare image for cropping
        padded_image, dims = prepare_image(image)
        # Post-process the layout string to extract elements
        elements = self._post_process(layout_str, padded_image, dims)
        logger.info(f"Extracted {len(elements)} elements from the image.")
        return elements
    def _post_process(
        self, layout_str: str, padded_image: Any, dims: Dict
    ) -> List[Dict]:
        """
        Post-processes the raw layout string from the model.
        This method parses the layout string, extracts bounding boxes and labels,
        and creates a list of element dictionaries.
        Args:
            layout_str: The raw layout string from the model.
            padded_image: The padded image used for cropping.
            dims: A dictionary containing original and padded dimensions.
        Returns:
            A list of dictionaries representing the extracted elements.
        """
        layout_results = self.parse_layout_string(layout_str)
        elements = []
        previous_box = None
        reading_order = 0
        for bbox, label in layout_results:
            try:
                # Process coordinates to get bounding boxes for cropping
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    orig_x1,
                    orig_y1,
                    orig_x2,
                    orig_y2,
                    previous_box,
                ) = process_coordinates(bbox, padded_image, dims, previous_box)
                # Crop the element from the padded image
                cropped = padded_image[y1:y2, x1:x2]
                # Ensure the cropped image is valid before adding it
                if cropped.size > 3 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    # Convert from OpenCV BGR format to PIL RGB format
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": self._to_block_type(label),
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                        "score": None,
                    }
                    elements.append(element_info)
                reading_order += 1
            except Exception as e:
                logger.error(f"Error processing bbox with label {label}: {e}")
                continue
        return elements        
    
    def parse_layout_string(self, bbox_str):
        """Parse layout string using regular expressions"""
        pattern = r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
        matches = re.finditer(pattern, bbox_str)
        parsed_results = []
        for match in matches:
            coords = [float(match.group(i)) for i in range(1, 5)]
            label = match.group(5).strip()
            parsed_results.append((coords, label))
        return parsed_results

    def _to_block_type(self, label: str) -> str:
        """Converts a raw model label to a structured BlockType value."""
        label_map = {
            'tab': SpanType.Table,
            'fig': SpanType.Image,
            'title': SpanType.Title,
            'sec': SpanType.Section,
            'sub_sec': SpanType.SubSection,
            'list': SpanType.List,
            'formula': SpanType.Formula,
            'reference': SpanType.Text,
            'alg': SpanType.Algorithm,
            'para': SpanType.Text,
            'header': SpanType.Header,
            'footer': SpanType.Footer,
        }
        block_type = label_map.get(label, SpanType.Unknown)
        if block_type == SpanType.Unknown:
            logger.warning(f"Unknown label '{label}' encountered, mapping to 'unknown'.")
        return block_type.value