import os
import base64
import json
import sys
import requests  # 添加requests导入

# 移除TestClient导入
#from fastapi.testclient import TestClient
# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
#from src.service.server import app
from src.utils.perf_timer import PerfTimer
import pytest

class RealClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def post(self, url, headers=None, json=None):
        full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        return self.session.post(full_url, headers=headers, json=json)
    
client = RealClient(base_url="http://sd286m9ufnav2uhs5jgng.apigateway-cn-beijing.volceapi.com/")


#client = TestClient(app)
timer = PerfTimer()
timer.enable()


    
def prepare_test_image():
    """准备测试图片并转换为base64格式"""
    test_image_path = os.path.join(os.path.dirname(__file__), 'images', 'bill.png')
    assert os.path.exists(test_image_path), f"测试图片不存在: {test_image_path}"
    
    with open(test_image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    return {
        'image_url': {
            'url': f'data:image/png;base64,{image_base64}'
        }
    }


def validate_response(response):
    """验证API响应结构"""
    assert response.status_code == 200, f"API请求失败: {response.text}"
    response_json = response.json()
    
    assert 'elements' in response_json, "响应缺少elements字段"
    assert isinstance(response_json['elements'], list), "elements应该是数组类型"
    assert len(response_json['elements']) > 0, "未检测到任何文档元素"
    
    for element in response_json['elements']:
        assert 'bbox' in element, "元素缺少bbox字段"
        assert 'label' in element, "元素缺少label字段"
        assert 'score' in element, "元素缺少score字段"
    print(response_json)
    return response_json



    

@pytest.mark.parametrize("endpoint", [
    '/v1/pipeline/slow_layout',
    '/v1/pipeline/fast_layout'
])
def test_layout_api_endpoints(endpoint):
    """参数化测试两个布局API端点"""
    try:
        request_data = prepare_test_image()
        
        response = client.post(
            endpoint,
            headers={
                'Authorization': f'Bearer {os.getenv("API_KEY", "sk-ocr-7Q2pZtR9xK4mF8sD3gH6jL1nP2bV5cX7rB9tY0kU3iO8fS2dA5sF7gH9jK")}',
                'Content-Type': 'application/json'
            },
            json=request_data
        )
        
        validate_response(response)
    except Exception as e:
        pytest.fail(f"测试{endpoint}失败: {str(e)}")

if __name__ == '__main__':
    for endpoint in ['/v1/pipeline/fast_layout', '/v1/pipeline/slow_layout']:
        test_layout_api_endpoints(endpoint)
