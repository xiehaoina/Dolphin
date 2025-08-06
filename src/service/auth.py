from typing import Tuple, Dict
from fastapi import Request
import json
import os
from loguru import logger


API_KEY = os.getenv('API_KEY','sk-ocr-7Q2pZtR9xK4mF8sD3gH6jL1nP2bV5cX7rB9tY0kU3iO8fS2dA5sF7gH9jK')
logger.info(f"API_KEY: {API_KEY}")
def check_api_key(request: Request) -> Tuple[bool, Dict]:
    """验证API密钥，返回(是否有效, 错误响应字典)"""
    auth_header = request.headers.get("Authorization")
    
    # 1. 检查Authorization头格式
    if not auth_header or not auth_header.startswith("Bearer "):
        return False, {
            'statusCode': 401,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Invalid Authorization header format. Use: Bearer <api_key>'
            })
        }
    
    # 2. 提取并验证API密钥
    api_key = auth_header.split(" ")[1]
    if api_key != API_KEY:
        return False, {
            'statusCode': 401,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'API key is invalid'})
        }
    
    return True, {}