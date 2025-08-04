# 使用官方Python镜像作为基础镜像
FROM hub.byted.org/base/security.debian12-slim.python310:latest
ARG HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118/
ARG HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118/
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
# 安装系统依赖（解决OpenCV的libGL错误）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目依赖文件
COPY requirements.txt .
COPY src ./src

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

# 设置启动命令
CMD ["python", "/app/src/service/server.py"]