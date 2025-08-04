import os
import base64
import json
import sys
from fastapi.testclient import TestClient
# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.service.server import app

client = TestClient(app)

def test_layout_api_with_bill_image():
    # 获取测试图片路径
    test_image_path = os.path.join(os.path.dirname(__file__), 'images', 'bill.png')
    assert os.path.exists(test_image_path), f"测试图片不存在: {test_image_path}"

    # 读取图片并转换为base64
    with open(test_image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # 准备请求数据
    request_data = {
        'image_url':{
             'url': f'data:image/png;base64,{image_base64}'
        }
    }

    # 发送请求（使用默认测试API密钥）
    response = client.post(
        '/v1/pipeline/layout',
        headers={
            'Authorization': 'Bearer 123456',
            'Content-Type': 'application/json'
        },
        json=request_data
    )

    # 验证响应状态码
    assert response.status_code == 200, f"API请求失败: {response.text}"

    # 解析响应内容
    response_json = response.json()

    # 验证响应结构
    assert 'elements' in response_json, "响应缺少elements字段"
    assert isinstance(response_json['elements'], list), "elements应该是数组类型"

    # 验证至少有一个元素被检测到
    assert len(response_json['elements']) > 0, "未检测到任何文档元素"

    # 验证元素包含必要的字段
    for element in response_json['elements']:
        assert 'bbox' in element, "元素缺少bbox字段"
        assert 'label' in element, "元素缺少label字段"
        assert 'score' in element, "元素缺少score字段"
        assert isinstance(element['score'], float), "score应该是浮点型"
        assert 0 <= element['score'] <= 1, "score值应该在0-1之间"
    print(response_json)
    

if __name__ == '__main__':
    test_layout_api_with_bill_image()