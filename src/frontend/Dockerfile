# 多阶段构建 - 分子毒性预测平台
FROM node:18-alpine AS frontend-builder

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建前端
RUN npm run build

# Python后端阶段
FROM python:3.9-slim AS backend

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制后端代码
COPY backend/ ./backend/
COPY ../backupunimolpy/ ./backupunimolpy/

# 安装Python依赖
RUN pip install fastapi uvicorn python-multipart numpy pandas matplotlib seaborn pillow

# 最终阶段
FROM node:18-alpine AS production

# 安装Python
RUN apk add --no-cache python3 py3-pip

# 设置工作目录
WORKDIR /app

# 从构建阶段复制文件
COPY --from=frontend-builder /app/.next ./.next
COPY --from=frontend-builder /app/public ./public
COPY --from=frontend-builder /app/package*.json ./
COPY --from=backend /app ./

# 安装生产依赖
RUN npm ci --only=production

# 安装Python依赖
RUN pip3 install fastapi uvicorn python-multipart numpy pandas matplotlib seaborn pillow

# 暴露端口
EXPOSE 3000 8000

# 创建启动脚本
RUN echo '#!/bin/sh\n\
# 启动后端\n\
cd /app/backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
# 启动前端\n\
cd /app && npm start\n\
' > /app/start.sh && chmod +x /app/start.sh

# 启动命令
CMD ["/app/start.sh"]