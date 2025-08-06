import os
import argparse
import logging
from mcp.server.fastmcp import FastMCP
import requests


# Create MCP server
mcp = FastMCP(
    "volcano ocr layout analysis server", host="0.0.0.0", port=int(os.getenv("PORT", "8001"))
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



API_KEY = os.getenv('API_KEY','sk-ocr-7Q2pZtR9xK4mF8sD3gH6jL1nP2bV5cX7rB9tY0kU3iO8fS2dA5sF7gH9jK')
BASE_URL = os.getenv('BASE_URL',"http://sd286m9ufnav2uhs5jgng.apigateway-cn-beijing.volceapi.com/")
class HttpClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def post(self, url, headers=None, json=None):
        full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        return self.session.post(full_url, headers=headers, json=json)


@mcp.tool()
def layout_analysis(image_url: object, mode: str):
    """文档布局分析工具，支持快速和慢速两种模式

    Parameters:
        image_url: 图像资源对象，符合OpenAI API标准格式，支持两种模式：
                   1. URL模式: {"url": "https://example.com/document.jpg"} (HTTP/HTTPS可访问URL)
                   2. Base64模式: {"data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."} (data URI嵌入base64编码图像)
        mode: 分析模式，可选值：
            - "fast": 使用低延时、低精度模型进行快速布局分析
            - "slow": 使用高延时、高精度模型进行高精度布局分析

    Returns:
        list: 布局分析结果，包含文档元素坐标、类型和置信度等信息
            - type (str): 元素类型，可能值包括'table'、'image'、'title'、'text'等
            - bbox (list): 边界框坐标，格式为[x1, y1, x2, y2]，代表元素在图像中的像素位置
            - reading_order (int): 阅读顺序编号，按文档逻辑流排序
            - score (float): 元素类型预测置信度，范围0-1

            示例:
            [
                {
                    "type": "table",
                    "bbox": [100, 200, 500, 400],
                    "reading_order": 0,
                    "score": 0.98
                },
                {
                    "type": "text",
                    "bbox": [100, 450, 500, 600],
                    "reading_order": 1,
                    "score": 0.95
                }

            ]
    Examples:
        >>> layout_analysis( {"url": "https://example.com/document.jpg"}, "fast")
        [
            {
                "type": "table",
                "bbox": [100, 200, 500, 400],
                "reading_order": 0,
                "score": 0.98
            },
            {
                "type": "text",
                "bbox": [100, 450, 500, 600],
                "reading_order": 1,
                "score": 0.95
            }
        ]
        >>> layout_analysis( {"data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."}, "slow")
        [
            {
                "type": "table",
                "bbox": [100, 200, 500, 400],
                "reading_order": 0,
                "score": 0.98
            },
            {
                "type": "text",
                "bbox": [100, 450, 500, 600],
                "reading_order": 1,
                "score": 0.95
            }
        ]
    """
    event = {
        'body': {
            'image_url': image_url,
            },
    }
    headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
    
    client = HttpClient(base_url=BASE_URL)
    resp = None
    if mode == "slow":
        resp = client.post("/v1/pipeline/slow_layout", headers=headers, json=event)
    else:
        resp = client.post("/v1/pipeline/fast_layout", headers=headers, json=event)
        
    
    return resp



if __name__ == '__main__':
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Run the Dolphin MCP Server")
    parser.add_argument(
        "--transport",
        "-t",
        choices=["sse", "stdio", "streamable-http"],
        default="streamable-http",
        help="Transport protocol to use (sse or stdio)",
    )

    args = parser.parse_args()

    # Run the MCP server
    logger.info(f"Starting volcano ocr layout analysis server with {args.transport} transport")
    

    if args.transport == "streamable-http":
        mcp.settings.stateless_http = True
        mcp.settings.json_response = True

    mcp.run(transport=args.transport)