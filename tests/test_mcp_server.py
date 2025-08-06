import os
import base64
import unittest
import asyncio
from mcp.client.session_group import ClientSessionGroup
from requests import session


# 测试配置
MCP_SERVER_URL = "http://localhost:8001"
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'images', 'bill.png')
AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'default_test_token')

class TestMcpServerIntegration(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        """类级别的测试准备，读取测试图片"""
        # 读取测试图片并转换为base64
        with open(TEST_IMAGE_PATH, 'rb') as f:
            cls.TEST_IMAGE_BASE64 = base64.b64encode(f.read()).decode('utf-8')

    async def asyncSetUp(self):
        """异步测试准备，初始化ClientSessionGroup并连接服务器"""
        # 按照示例创建会话组（无base_url参数）
        self.session_group = ClientSessionGroup()

        # 连接到MCP服务器（修复参数传递方式）
        self.session = self.session_group.connect_to_server(
            url=MCP_SERVER_URL,
            headers={
                'Authorization': f'Bearer {AUTH_TOKEN}',
                'Content-Type': 'application/json'
            },
            timeout=30
        )
        print(self.session_group.resources)

    async def asyncTearDown(self):
        """异步测试清理，关闭会话组"""
        await self.session.close()

    async def test_valid_url_input(self):
        """测试URL模式的图像输入（异步测试）"""
        # 从会话组获取会话（示例中的异步上下文管理）
        
        endpoint = f"/v1/pipeline/fast_layout"
        payload = {
            "body": {
                "image_url": {"url": "https://example.com/test-document.jpg"}
            }
        }

        # 异步发送请求
        response = await self.session.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, f"MCP服务请求失败: {await response.text()}")
        result = await response.json()
        self._validate_response(result)

    async def test_valid_base64_input(self):
        """测试Base64模式的图像输入（异步测试）"""
        endpoint = f"/v1/pipeline/slow_layout"
        payload = {
            "body": {
                "image_url": {
                    "data": f"data:image/png;base64,{self.TEST_IMAGE_BASE64}"
                }
            }
        }

        response = await self.session.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, f"MCP服务请求失败: {await response.text()}")
        result = await response.json()
        self._validate_response(result)

    def test_invalid_image_url_format(self):
        """测试无效的image_url格式处理"""
        # 修复：使用完整URL而非base_url+endpoint
        endpoint = f"{MCP_SERVER_URL}/v1/pipeline/fast_layout"
        payload = {
            "body": {
                "image_url": {"invalid_key": "https://example.com/image.jpg"}
            }
        }

        response = self.session.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 400, f"期望400错误，实际状态码: {response.status_code}")
        error_info = response.json()
        self.assertIn("无效的image_url格式", error_info.get("error", ""))

    def test_invalid_mode_parameter(self):
        """测试无效的mode参数处理"""
        # 修复：使用完整URL而非base_url+endpoint
        endpoint = f"{MCP_SERVER_URL}/v1/pipeline/invalid_mode_layout"
        payload = {
            "body": {
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        }

        response = self.session.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 404, f"期望404错误，实际状态码: {response.status_code}")

    def test_empty_image_url(self):
        """测试空image_url参数处理"""
     
        # 修复：使用完整URL而非base_url+endpoint
        endpoint = f"{MCP_SERVER_URL}/v1/pipeline/fast_layout"
        payload = {"body": {"image_url": None}}

        response = self.session.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 400, f"期望400错误，实际状态码: {response.status_code}")
        error_info = response.json()
        self.assertIn("image_url参数不能为空", error_info.get("error", ""))

    def _validate_response(self, result):
        """验证响应结构和内容"""
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        for element in result:
            self.assertIsInstance(element, dict)
            self.assertIn("type", element)
            self.assertIn("bbox", element)
            self.assertIn("reading_order", element)
            self.assertIn("score", element)

if __name__ == '__main__':
    unittest.main(verbosity=2)