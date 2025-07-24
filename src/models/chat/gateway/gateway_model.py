import httpx
from openai import OpenAI
import asyncio
import random
from PIL import Image
from typing import List,  Any
from loguru import logger
import time

from src.utils.load_image import load_image, encode_image_base64
from src.models.chat.chat_model import ChatModel


class GatewayModel(ChatModel):
    """
    A ChatModel implementation that connects to a remote VLM service
    through an OpenAI-compatible API gateway.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        """
        Initializes the GatewayModel with a configuration object.

        Args:
            config: A configuration object with the following attributes:
                    - url (str): The base URL for the API.
                    - model_name (str): The name of the model to use.
                    - api_key (str, optional): The API key.
                    - max_concurrency (int, optional): Max concurrent requests.
        """
        self.model_name = getattr(config, 'model_name', 'default-model')
        self.api_key = getattr(config, 'api_key', None)
        self.base_url = getattr(config, 'url', None)
        self.max_concurrency = getattr(config, 'max_concurrency', 10)

        if not self.base_url:
            raise ValueError("API 'url' must be provided in the config.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(timeout=120)
        )
        self.async_client = httpx.AsyncClient(timeout=120)
        self.validate()

    def validate(self) -> None:
        """
        Validates the API connection and configuration.
        Raises ValueError if the connection is invalid.
        """
        try:
            self.client.models.list()
            logger.info("API connection validation successful")
        except Exception as e:
            logger.error(f"API connection validation failed: {e}")
            raise ValueError(f"API connection validation failed: {e}")

    def img2base64(self, image: Image.Image) -> tuple[str, str]:
        img_format = image.format if hasattr(image, 'format') and image.format else "PNG"
        image_base64 = encode_image_base64(image)
        return image_base64, img_format.lower()

    async def _call_openai_api(self, image: Image.Image, prompt: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            retries = 5
            delay = 1
            for i in range(retries):
                try:
                    start_time = time.time()
                    loaded_image = load_image(image, max_size=1600)
                    img_base64, img_type = self.img2base64(loaded_image)

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/{img_type};base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ],
                    }]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        timeout=120
                    )
                    #response = await self.async_client.post(
                    #    f"{self.base_url}/chat/completions",
                    #    headers={"Authorization": f"Bearer {self.api_key}"},
                    #    json={"model": self.model_name, "messages": messages},
                    #    timeout=120
                    #)
                    duration = time.time() - start_time
                    logger.debug(f'VLM infer duration: {duration:.2f}s, image base64 size:{len(img_base64)}')
                    return response.choices[0].message.content
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and i < retries - 1:
                        wait_time =  (delay * (2 ** i)) + (random.uniform(0, 1) * delay)
                        logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        delay *= 2
                    else:
                        logger.error(f'VLM error: {e}')
                        return f"Error: {str(e)}"
                except Exception as e:
                    logger.error(f'VLM error: {e}')
                    return f"Error: {str(e)}"
            return "Error: Max retries exceeded"

    def inference(self, prompt: str, image: Image.Image) -> str:
        """
        Performs inference on a single prompt and image.
        """
        results = self.batch_inference(prompts=[prompt], images=[image])
        if len(results) == 1:
            return results[0]
        else:
            raise Exception("Error: Inference failed")

    def batch_inference(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Performs inference on a batch of prompts and images.
        """
        return asyncio.run(self.async_batch_inference(images, prompts))

    async def async_batch_inference(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._call_openai_api( image, prompt, semaphore) for image, prompt in zip(images, prompts)]
        results = await asyncio.gather(*tasks)
        logger.debug(f'VLM result size: {len(results)}')
        return results