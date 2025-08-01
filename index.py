import json
import logging
import os

from omegaconf import OmegaConf
from src.models.factory import ModelFactory
from src.pipelines.processor.layout.doc_layout_yolo import DocLayoutYOLOProcessor
from src.utils.image_process.load_image import load_image,encode_image_base64


logging.basicConfig(level=logging.INFO)
API_KEY = os.getenv('API_KEY', '123456')
def handler(event, context):
    logging.info(f"received new request, event content: {event}")
    request = json.loads(event['body'])
    if request['api_key'] != API_KEY:
        return {
            'statusCode': 401,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'API key is invalid'
            })
        }

    config = OmegaConf.create({"device": "cpu", 
                                 "weight": "./model_weight/Structure/layout_zh.pt"
                                 })
    factory = ModelFactory()
    model = factory.create_model("doclayout_yolo", config)
    layout_processor = DocLayoutYOLOProcessor(model)
 
    
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
            },
            'api_key': API_KEY
        })
    }, None)
    print(ret)