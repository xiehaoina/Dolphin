import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
from loguru import logger
from typing import Any, List
from src.models.chat.chat_model import ChatModel

class HFDolphinModel(ChatModel):
    """
    A ChatModel implementation for the Dolphin model using the Hugging Face
    transformers library.
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the HFDolphinModel from a Hugging Face model ID or local path.

        Args:
            config: A configuration object with the attribute `model_id_or_path`.
        """
        model_id_or_path = config.model_id_or_path
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.tokenizer = self.processor.tokenizer
        self.validate()

    def validate(self) -> None:
        """
        Validates that the model and processor are loaded correctly.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model or processor not loaded properly.")
        logger.info("Huggingface dolphin load validation successful")

    def inference(self, prompt: str, image: Image.Image) -> str:
        """
        Performs inference on a single prompt and image.
        """
        results = self.batch_inference(prompts=[prompt], images=[image])
        return results[0] if results else "Error: Inference failed"

    def batch_inference(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Performs inference on a batch of prompts and images.
        """
        # Prepare images
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.to(self.device)
        if self.device != "cpu":
            batch_pixel_values = batch_pixel_values.half()

        # Prepare prompts
        full_prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            full_prompts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)

        # Generate text
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0
        )

        # Post-process output
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(full_prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
            
        return results
    
    