import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request, Depends
import json
import sys
import os

# 修正路径：从service目录上两级即可到达项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.service.faas.layout import handler
from src.service.auth import check_api_key

app = FastAPI(title="document extraction Service")


# 添加依赖项封装，便于复用
async def verify_api_key(request: Request) -> bool:
    is_valid, error_response = check_api_key(request)
    if not is_valid:
        raise HTTPException(
            status_code=error_response["statusCode"],
            detail=json.loads(error_response["body"]),
            headers=error_response["headers"]
        )
    return True

@app.post("/v1/pipeline/layout", response_class=JSONResponse)
async def layout(
    request: Request,
    api_key_valid: bool = Depends(verify_api_key)
):
    try:
        # 移除重复的API验证代码，使用依赖项验证
        request_body = await request.json()
        event = {
            'body': json.dumps(request_body),
            'headers': dict(request.headers)
        }
        
        handler_response = handler(event, None)
        return JSONResponse(
            status_code=handler_response['statusCode'],
            headers=handler_response['headers'],
            content=json.loads(handler_response['body'])
        )
    except Exception as e:
        # 使用HTTPException统一错误处理
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # 启动FastAPI服务
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")