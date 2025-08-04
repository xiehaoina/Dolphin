import json
import logging
import os

from omegaconf import OmegaConf
from src.models.factory import ModelFactory
from src.pipelines.processor.layout.doc_layout_yolo import DocLayoutYOLOProcessor
from src.utils.image_process.load_image import load_image,encode_image_base64


logging.basicConfig(level=logging.INFO)
DEVICE = os.getenv('DEVICE', 'cpu')
WEIGHT_PATH = os.getenv('WEIGHT_PATH', '../../../model_weight/Structure/layout_zh.pt')
config = OmegaConf.create({"weight": os.path.abspath(os.path.join(os.path.dirname(__file__), WEIGHT_PATH)),
                           "device": DEVICE})
factory = ModelFactory()
model = factory.create_model("doclayout_yolo", config)
layout_processor = DocLayoutYOLOProcessor(model)

def handler(event, context):
    logging.info(f"received new request, event content: {event}")
    request = json.loads(event['body'])
    elements = layout_processor.process(load_image(request['image_url']['url']))
    for j, _ in enumerate(elements):
        elem = elements[j]
        del elem["crop"]
    response = {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'elements': elements
        })
    }
    return response