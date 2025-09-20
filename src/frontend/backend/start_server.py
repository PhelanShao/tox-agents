#!/usr/bin/env python3
"""
简化的FastAPI启动脚本
避免在启动时加载重型模块
"""

import os
import sys
import logging
from pathlib import Path

# 设置工作目录到集成包根目录
BASE_DIR = Path(__file__).resolve().parents[2]
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="分子毒性预测API",
    description="基于原有Gradio项目的API封装",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "message": "API服务正常运行",
        "working_directory": os.getcwd(),
        "python_path": sys.path[:3]
    }

@app.get("/test-import")
async def test_import():
    """测试模块导入"""
    try:
        # 延迟导入，避免启动时阻塞
        from interface import process_binary_prediction
        from predictor import BinaryPredictor
        from MoleculePredictor import MoleculePredictor
        from visualizer import display_molecule_pymol
        
        return {
            "success": True,
            "message": "所有模块导入成功",
            "modules": [
                "interface.process_binary_prediction",
                "predictor.BinaryPredictor", 
                "MoleculePredictor.MoleculePredictor",
                "visualizer.display_molecule_pymol"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"模块导入失败: {str(e)}"
        }

if __name__ == "__main__":
    print("🚀 启动简化版分子毒性预测API服务器...")
    print(f"📁 工作目录: {os.getcwd()}")
    print("📱 前端地址: http://localhost:3000")
    print("🔧 API地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
