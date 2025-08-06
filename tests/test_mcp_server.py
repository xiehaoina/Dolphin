import os
import base64
import unittest
import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.session_group import ClientSessionGroup
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from requests import session
from typing import Optional

# 测试配置
MCP_SERVER_URL = "http://localhost:8001"
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'images', 'bill.png')
AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'default_test_token')


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None

    async def connect_to_streamable_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # 初始化AsyncExitStack用于管理异步上下文
        self._exit_stack = AsyncExitStack()

        # 获取流并保存上下文引用
        self._streams_context = streamablehttp_client(url=server_url)
        streams = await self._exit_stack.enter_async_context(self._streams_context)

        # 初始化客户端会话并设置30秒超时
        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._exit_stack.enter_async_context(self._session_context)

        # 初始化会话
        await self.session.initialize()

        # 列出可用工具验证连接
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if hasattr(self, '_exit_stack'):
            await self._exit_stack.aclose()

    async def call_tool(self, tool_name, tool_args):
        """调用指定工具并返回结果"""
        return await self.session.call_tool(tool_name, tool_args)

    

class TestMcpServerIntegration(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        """类级别的测试准备，读取测试图片"""
        # 读取测试图片并转换为base64
        with open(TEST_IMAGE_PATH, 'rb') as f:
            cls.TEST_IMAGE_BASE64 = base64.b64encode(f.read()).decode('utf-8')

    async def asyncSetUp(self):
        self._exit_stack = AsyncExitStack()
        self.mcp_client = MCPClient()
        await self.mcp_client.connect_to_streamable_server(MCP_SERVER_URL)

    async def asyncTearDown(self):
        await self._exit_stack.aclose()

    async def test_valid_url_input(self):
        """测试URL模式的图像输入"""
        # 使用MCPClient调用fast_layout工具
        result = await self.mcp_client.call_tool("fast_layout", {
            "image_url": {"url": "https://example.com/test-document.jpg"}
        })
        self._validate_response(result)

    async def test_valid_base64_input(self):
        """测试Base64模式的图像输入"""
        # 使用MCPClient调用slow_layout工具
        result = await self.mcp_client.call_tool("slow_layout", {
            "image_url": {
                "data": f"data:image/png;base64,{self.TEST_IMAGE_BASE64}"
            }
        })
        self._validate_response(result)

    async def test_invalid_image_url_format(self):
        """测试无效的image_url格式处理"""
        try:
            await self.mcp_client.call_tool("fast_layout", {
                "image_url": {"invalid_key": "https://example.com/image.jpg"}
            })
            self.fail("未捕获到无效image_url格式错误")
        except Exception as e:
            error_info = json.loads(str(e))
            self.assertIn("无效的image_url格式", error_info.get("error", ""))

    async def test_invalid_mode_parameter(self):
        """测试无效的mode参数处理"""
        # 使用MCPClient调用不存在的工具
        try:
            await self.mcp_client.call_tool("invalid_mode_layout", {
                "image_url": {"url": "https://example.com/image.jpg"}
            })
            self.fail("未捕获到无效mode错误")
        except Exception as e:
            error_info = json.loads(str(e))
            self.assertEqual(error_info.get("status_code"), 404)

    async def test_empty_image_url(self):
        """测试空image_url参数处理"""
        # 使用MCPClient调用工具传递空image_url
        try:
            await self.mcp_client.call_tool("fast_layout", {
                "image_url": None
            })
            self.fail("未捕获到空image_url错误")
        except Exception as e:
            error_info = json.loads(str(e))
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