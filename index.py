import json
import logging

from omegaconf import OmegaConf
from src.models import factory
from src.models.factory import ChatModelFactory
from src.pipelines.processor.layout.dolphin import DolphinLayoutProcessor
from src.utils.load_image import load_image,encode_image_base64


logging.basicConfig(level=logging.INFO)
config = OmegaConf.create({"model_id_or_path": "./model_weight/Dolphin/hf_model"})
factory = ChatModelFactory()
model = factory.create_model("hf_dolphin", config)
layout_processor = DolphinLayoutProcessor(model)

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

if __name__ == '__main__':
    image_base64 = encode_image_base64('examples/data/volc/bill/hn1.png')
    ret = handler({
        'body': json.dumps({
            'image_url': {
                'url': f'data:image/PNG;base64,{image_base64}'
            }
        })
    }, None)
    print(ret)