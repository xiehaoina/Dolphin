import json
import logging
import os

from omegaconf import OmegaConf
from src.models.factory import ModelFactory
from src.pipelines.processor.layout.dolphin import DolphinLayoutProcessor
from src.utils.image_process.load_image import load_image


logging.basicConfig(level=logging.INFO)
MODEL_PATH = os.getenv('MODEL_PATH', '../../../model_weight/Dolphin/hf_model')
config = OmegaConf.create({"model_id_or_path": os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))})
factory = ModelFactory()
model = factory.create_model("hf_dolphin", config)
layout_processor = DolphinLayoutProcessor(model)

def handler(event, context):
    logging.debug(f"received new request, event content: {event}")
    context.perf_timer.start_timer("remote_call")
    request = json.loads(event['body'])
    elements = layout_processor.process(load_image(request['image_url']['url']), add_reading_order=False)
    context.perf_timer.stop_timer("remote_call")
    for j, _ in enumerate(elements):
        elem = elements[j]
        del elem["crop"]
    response = {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(elements)
    }
    return response