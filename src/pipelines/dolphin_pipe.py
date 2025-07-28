import os
import glob
from re import T
import cv2
from PIL import Image
from typing import Any, List, Dict
from loguru import logger
import time
from omegaconf import OmegaConf
from torch import true_divide

from src.pipelines.pipeline import Pipeline
from src.models.factory import ChatModelFactory
from src.utils.perf_timer import PerfTimer
from src.utils.utils import (
    setup_output_dirs,
    save_figure_to_local,
    prepare_image,
    convert_pdf_to_images,
    save_combined_pdf_results,
    parse_layout_string,
    process_coordinates,
    save_outputs,
)


class DolphinPipeline(Pipeline):
    """
    An end-to-end pipeline for document processing using the Dolphin model.
    This pipeline handles file input, layout analysis, and element-level parsing.
    """

    def __init__(self, config: Any):
        """
        Initializes the DolphinPipeline with a configuration object.

        Args:
            config: A configuration object, expected to have 'model_config'
                    and 'max_batch_size'.
        """
        super().__init__(config)
        self.timer = PerfTimer(debug=True)
        
        logger.info("Creating model...")
        
        self.timer.start_timer("create_dolphin_model")
        factory = ChatModelFactory()
        self.model = factory.create_model("hf_dolphin", self.config)
        self.timer.stop_timer("create_dolphin_model")
        
        self.timer.start_timer("create_gateway_model")
        config = OmegaConf.create({"url": "https://ai-gateway.vei.volces.com/v1", 
                                 "model_name": "doubao-1.5-vision-pro", 
                                 "api_key": "sk-b187cc92d38040cbbf76839f2ea5980cag5bv9m2r4j17vah",
                                 "max_concurrency": 30})
        self.gateway_model = factory.create_model("gateway", config)
        self.timer.stop_timer("create_gateway_model")
        
        self.timer.log_timings()
        self.max_batch_size = getattr(self.config, "max_batch_size", 16)

    def run(self, debug: bool, input_path: str, save_dir: str, **kwargs) -> List[str]:
        """
        Runs the full document processing pipeline on a given input path.

        Args:
            debug: If True, enables performance logging.
            input_path: Path to an input image/PDF or a directory of files.
            save_dir: Directory to save the parsing results.
            **kwargs: Additional arguments (not used in this implementation).

        Returns:
            A list of paths to the generated JSON result files.
        """
        if debug:
            self.timer.enable()
        else:
            self.timer.disable()

        self.timer.start_timer("total_run")

        document_files = self._collect_files(input_path)

        if not document_files:
            logger.warning(f"No supported files found in {input_path}")
            return []

        setup_output_dirs(save_dir)
        logger.info(f"Total files to process: {len(document_files)}")

        result_paths = []
        for file_path in document_files:
            logger.info(f"Processing {file_path}")
            try:
                json_path, _ = self._process_document(file_path, save_dir)
                if json_path:
                    result_paths.append(json_path)
                logger.info(f"Processing completed. Results saved to {save_dir}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        self.timer.stop_timer("total_run")
        self.timer.log_timings()
        return result_paths

    def _collect_files(self, input_path: str) -> List[str]:
        """Collects all supported document files from the input path."""
        if os.path.isdir(input_path):
            extensions = [".jpg", ".jpeg", ".png", ".pdf"]
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
                files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
            return sorted(list(set(files)))
        elif os.path.exists(input_path):
            return [input_path]
        else:
            raise FileNotFoundError(f"Input path {input_path} does not exist")

    def _process_document(self, doc_path: str, save_dir: str) -> (str, List[Dict]):
        """Processes a single document, handling both PDF and image formats."""
        if doc_path.lower().endswith(".pdf"):
            self.timer.start_timer("convert_pdf_to_images")
            images = convert_pdf_to_images(doc_path)
            self.timer.stop_timer("convert_pdf_to_images")
            all_results = []
            for i, image in enumerate(images):
                page_name = f"{os.path.splitext(os.path.basename(doc_path))[0]}_page_{i+1:03d}"
                _, page_results = self._process_single_image(image, save_dir, page_name, save_individual=True)
                all_results.append({"page_number": i + 1, "elements": page_results})
            json_path = save_combined_pdf_results(all_results, doc_path, save_dir)
            return json_path, all_results
        else:
            image = Image.open(doc_path).convert("RGB")
            image_name = os.path.splitext(os.path.basename(doc_path))[0]
            return self._process_single_image(image, save_dir, image_name)

    def _process_single_image(
        self, image: Image.Image, save_dir: str, image_name: str, save_individual: bool = True
    ) -> (str, List[Dict]):
        """Processes a single image page."""
        self.timer.start_timer("layout_analysis")
        layout_output = self.model.inference(
            "Parse the reading order of this document.", image
        )
        self.timer.stop_timer("layout_analysis")
        padded_image, dims = prepare_image(image)

        recognition_results = self._process_elements(
            layout_output, padded_image, dims, save_dir, image_name
        )

        json_path = None
        if save_individual:
            dummy_path = f"{image_name}.jpg"
            json_path = save_outputs(recognition_results, dummy_path, save_dir, image)
        return json_path, recognition_results

    def _process_elements(self, layout_str: str, padded_image, dims, save_dir: str, image_name: str) -> List[Dict]:
        """Parses all document elements from a layout string."""
        layout_results = parse_layout_string(layout_str)
        text_elements, table_elements, figure_results = [], [], []
        previous_box = None
        reading_order = 0

        for bbox, label in layout_results:
            try:
                x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(bbox, padded_image, dims, previous_box)
                cropped = padded_image[y1:y2, x1:x2]

                if cropped.size > 3 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {"crop": pil_crop, "label": label, "bbox": [orig_x1, orig_y1, orig_x2, orig_y2], "reading_order": reading_order}
                    
                    if label == "fig":
                        fig_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                        del element_info["crop"]
                        figure_results.append({**element_info, "text": f"![Figure](figures/{fig_filename})", "figure_path": f"figures/{fig_filename}"})
                    elif label == "tab":
                        table_elements.append(element_info)
                    else:
                        text_elements.append(element_info)
                reading_order += 1
            except Exception as e:
                logger.error(f"Error processing bbox with label {label}: {e}")
                continue

        recognition_results = figure_results
        if text_elements:
            recognition_results.extend(self._process_element_batch(text_elements, "Read text in the image."))
        if table_elements:
            recognition_results.extend(self._process_element_batch(table_elements, "Parse the table in the image."))
        
        recognition_results.sort(key=lambda x: x.get("reading_order", 0))
        return recognition_results

    def _process_element_batch(self, elements: List[Dict], prompt: str) -> List[Dict]:
        """Processes a batch of elements with the same prompt."""
        results = []
        for i in range(0, len(elements), self.max_batch_size):
            batch = elements[i : i + self.max_batch_size]
            crops = [elem["crop"] for elem in batch]
            prompts = [prompt] * len(crops)

            self.timer.start_timer("batch_inference")
            batch_results = self.model.batch_inference(prompts, crops)
            self.timer.stop_timer("batch_inference")

            for j, result_text in enumerate(batch_results):
                elem = batch[j]
                del elem["crop"]
                results.append({**elem, "text": result_text.strip()})
        return results