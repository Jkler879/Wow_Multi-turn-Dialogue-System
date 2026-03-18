# 使用 Python 3.11  slim 镜像，与你本地环境一致
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（如需编译某些 Python 包，如 sentence-transformers 可能依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（注意 .dockerignore 已排除无用文件）
COPY . .

# 确保 scripts 目录下的脚本有执行权限（如果有）
RUN chmod +x scripts/*.sh || true

# 暴露端口
EXPOSE 8000

# 启动命令（使用 uvicorn 运行 FastAPI 应用）
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]