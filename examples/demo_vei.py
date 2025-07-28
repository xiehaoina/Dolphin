""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import os
import sys
from omegaconf import OmegaConf

#from transformers import AutoProcessor, VisionEncoderDecoderModel
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.pipelines.dolphin_pipe import DolphinPipeline



def main():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image/PDF or directory of files")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum number of document elements to parse in a single batch (default: 16)",
    )
    args = parser.parse_args()
    
    config = OmegaConf.create({"model_id_or_path": args.model_path})
    pipeline = DolphinPipeline(config)
    
    # Determine the output directory
    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    
    # Run the pipeline
    pipeline.run(debug=True, input_path=args.input_path, save_dir=save_dir)


if __name__ == "__main__":
    main()
