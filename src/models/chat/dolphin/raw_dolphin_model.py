""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import os
import warnings
from collections import OrderedDict
from typing import Any, List

from omegaconf import ListConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

from src.models.chat.chat_model import ChatModel
from src.models.chat.dolphin.model import DonutConfig, DonutModel, SwinEncoder
from src.utils.processor import DolphinProcessor


def try_rename_lagacy_weights(ckpt, output_path=""):
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    if "module" in ckpt.keys():
        ckpt = ckpt["module"]
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("encoder"):
            new_ckpt["vpm" + k[len("encoder") :]] = v
        elif k.startswith("decoder"):
            new_ckpt["llm" + k[len("encoder") :]] = v
        else:
            new_ckpt[k] = v
    if output_path:
        torch.save(new_ckpt, output_path)
    return new_ckpt


def convert_listconfig_to_list(config):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, ListConfig):
            new_config[k] = list(v)
        else:
            new_config[k] = v
    return new_config


class RawDolphinModel(ChatModel):
    """
    A ChatModel implementation for the DOLPHIN model, which runs locally.
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the DolphinModel with a given configuration.

        Args:
            config: A configuration object containing model-specific settings.
        """
        self.model_args = config.model
        self.swin_args = config.model.pop("swin_args")
        self.swin_args = convert_listconfig_to_list(self.swin_args)

        vision_tower = SwinEncoder(
            input_size=self.swin_args["img_size"],
            patch_size=self.swin_args["patch_size"],
            embed_dim=self.swin_args["embed_dim"],
            window_size=self.swin_args["window_size"],
            encoder_layer=self.swin_args["encoder_layer"],
            num_heads=self.swin_args["num_heads"],
            align_long_axis=self.swin_args["align_long_axis"],
        )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.model_args.tokenizer_path)
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        if self.model_args.get("extra_answer_tokens", False):
            prompt_end_token = " <Answer/>"
            self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set([prompt_end_token]))})
            self.tokenizer._prompt_end_token = prompt_end_token
            self.tokenizer._prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(prompt_end_token)

        donut_config = DonutConfig(
            decoder_layer=self.model_args.decoder_layer,
            max_length=self.model_args.max_length,
            max_position_embeddings=self.model_args.max_position_embeddings,
            hidden_dimension=self.model_args.hidden_dimension,
        )

        self.model = DonutModel(config=donut_config, vision_tower=vision_tower, tokenizer=self.tokenizer)
        if self.model_args.model_name_or_path:
            ckpt = torch.load(self.model_args.model_name_or_path, map_location="cpu")
            ckpt = try_rename_lagacy_weights(ckpt)
            self.model.load_state_dict(ckpt, strict=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.model.to(torch.float32)
        self.model.to(self.device)
        self.model.eval()
        transform_args = {
            "input_size": self.swin_args["img_size"],
            "max_length": self.model_args.max_length,
        }
        self.processor = DolphinProcessor({}, self.tokenizer, transform_args=transform_args)
        self.validate()

    def validate(self) -> None:
        """
        Validates that the model is loaded.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")

    def _preprocess_image(self, image: Image.Image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_tensor = self.processor.process_image_for_inference(image, return_img_size=False)
        return image_tensor

    def _preprocess_prompt(self, prompt: str):
        if self.model_args.get("extra_answer_tokens", False):
            if self.tokenizer._prompt_end_token not in prompt:
                prompt = prompt + self.tokenizer._prompt_end_token
        prompt_ids = self.processor.process_prompt_for_inference(prompt)
        return prompt_ids

    def _postprocess(self, output: str, prompt: str) -> str:
        output = output.replace("<s>", "").replace(prompt, "").replace("</s>", "").replace("<pad>", "")
        if self.model_args.get("extra_answer_tokens", False):
            output = output.split(self.tokenizer._prompt_end_token)[-1]
        return output

    def inference(self, prompt: str, image: Image.Image) -> str:
        """
        Performs inference on a single prompt and image.
        """
        image_tensor = self._preprocess_image(image).to(self.device)
        prompt_ids = self._preprocess_prompt(prompt).to(self.device)

        model_output = self.model.inference(image_tensors=image_tensor, prompt_ids=prompt_ids)

        output = self._postprocess(model_output["repetitions"][0], prompt)
        return output

    def batch_inference(self, prompts: List[str], images: List[Image.Image], max_batch_size: int = 16) -> List[str]:
        """
        Performs inference on a batch of prompts and images.
        """
        image_tensors = [self._preprocess_image(img) for img in images]
        image_tensor = torch.cat(image_tensors, dim=0)

        processed_prompts = []
        for p in prompts:
            if self.model_args.get("extra_answer_tokens", False):
                if self.tokenizer._prompt_end_token not in p:
                    p = p + self.tokenizer._prompt_end_token
            if not p.startswith("<s>"):
                p = "<s>" + p
            processed_prompts.append(p)

        self.processor.tokenizer.padding_side = "left"
        prompt_ids = self.processor.tokenizer(
            processed_prompts, add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids

        model_output_batch = []
        for i in range(0, image_tensor.shape[0], max_batch_size):
            image_tensor_batch = image_tensor[i : i + max_batch_size].to(self.device)
            prompt_ids_batch = prompt_ids[i : i + max_batch_size].to(self.device)
            model_output = self.model.inference(image_tensors=image_tensor_batch, prompt_ids=prompt_ids_batch)
            model_output_batch.append(model_output)

        if not model_output_batch:
            return []

        # Combine results from all batches
        combined_output = {}
        for key in model_output_batch[0].keys():
            combined_list = []
            for batch in model_output_batch:
                data = batch[key]
                if isinstance(data, torch.Tensor):
                    combined_list.extend(data.cpu().numpy().tolist())
                else:
                    combined_list.extend(data)
            combined_output[key] = combined_list

        outputs = [self._postprocess(combined_output["repetitions"][i], prompts[i]) for i in range(len(prompts))]
        return outputs