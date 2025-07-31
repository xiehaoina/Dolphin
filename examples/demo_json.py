""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import os
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.pipelines.dolphin_pipeline import DolphinPipeline
from src.utils.image_process.load_image import load_image
from src.utils.utils import save_outputs


def main():
    parser = argparse.ArgumentParser(description="Refine JSON data using a document image with DOLPHIN.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file to be refined.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input image directory)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/Dolphin.yaml",
        help="Path to the pipeline configuration file.",
    )
    args = parser.parse_args()

    # Load pipeline configuration
    if not os.path.exists(args.config_path):
        logger.error(f"Configuration file not found at: {args.config_path}")
        sys.exit(1)
    config = OmegaConf.load(args.config_path)

    # Initialize the pipeline
    pipeline = DolphinPipeline(config)
    logger.info("Dolphin pipeline initialized successfully.")

    # Determine the output directory
    save_dir = args.save_dir or os.path.dirname(args.input_image)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load image and JSON data
    try:
        image = load_image(args.input_image)
        with open(args.input_json, 'r', encoding='utf-8') as f:
            input_json_data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error loading files: {e}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {args.input_json}. Please ensure it is a valid JSON file.")
        sys.exit(1)

    logger.info(f"Processing {Path(args.input_image).name} with {Path(args.input_json).name}...")

    # Run the refinement process
    # Note: Assuming a method `refine_json` exists in the pipeline.
    # This might need to be adjusted based on the actual pipeline implementation.
    if hasattr(pipeline, 'refine_json'):
        refined_json = pipeline.refine_json(input_json_data, image)
    else:
        logger.error("The current pipeline does not have a 'refine_json' method. Please check the implementation.")
        sys.exit(1)

    # Save the outputs
    image_name = Path(args.input_image).stem
    output_path_prefix = os.path.join(save_dir, image_name)
    
    # save_outputs expects a list of recognition results, so we wrap the dict
    # and create a dummy structure if needed.
    # We'll save the refined JSON directly.
    output_json_path = f"{output_path_prefix}_refined.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(refined_json, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Refined JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()
