import cv2
import numpy as np
import re
from PIL import Image
from typing import Any, List, Dict
from loguru import logger

from src.models.layout.layout_model import LayoutModel
from src.pipelines.processor.processor import Processor
from src.utils.enums.doc_element_type import SpanType
from src.utils.utils import (
    prepare_image,
)
class DocLayoutYOLOProcessor(Processor):
    """
    Processor for performing layout analysis on a document image.
    This processor uses a ChatModel to infer the layout of a document, then processes the
    output to extract individual elements, their bounding boxes, and reading order.
    """
    def __init__(self, model: LayoutModel):
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
        raw_dict = self.model.inference(image)
        image  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elements = self._post_process(raw_dict, image)
        logger.info(f"Extractehttps://file+.vscode-resource.vscode-cdn.net/Users/bytedance/git/ocr/Dolphin/results/volc/bill/hn1/bbox_images/layout.png?version%3D1754022229084d {len(elements)} elements from the image.")
        return elements
    def _post_process(
        self, raw_dict: Dict, image: Image
    ) -> Dict:      
        elements = []
        reading_order = 0
        for element in raw_dict:
            try:
                [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax] = element['poly']
                score = element['score']
                if score < 0.35:
                    continue
                category_id = element['category_id']
                # Crop the element from the padded image
                cropped = image[ymin:ymax, xmin:xmax]
                pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                element_info = {
                    "crop": pil_crop,
                    "label": self._to_block_type(category_id),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "reading_order": reading_order,
                    "score": score,
                }
                elements.append(element_info)
                reading_order += 1
            except Exception as e:
                logger.error(f"Error processing bbox with category_id {category_id}: {e}")
                continue
        return elements        
    

    def _to_block_type(self, label: int) -> str:
        """Converts a raw model label to a structured BlockType value."""
        label_map = {
            # 文档结构元素
            0: SpanType.Title,
            1: SpanType.Text,
            # 图像相关元素
            3: SpanType.Image,
            4: SpanType.ImageCaption,
            101: SpanType.ImageFootnote,
            # 表格相关元素
            5: SpanType.Table,
            6: SpanType.TableCaption,
            7: SpanType.TableFootnote,
            # 公式相关元素
            8: SpanType.Formula,
            13: SpanType.InlineEquation,
            # 布局辅助元素
            2: SpanType.Discarded
        }
        block_type = label_map.get(label, SpanType.Unknown)
        if block_type == SpanType.Unknown:
            logger.warning(f"Unknown label '{label}' encountered, mapping to 'unknown'.")
        return block_type.value