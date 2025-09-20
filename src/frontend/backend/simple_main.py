#!/usr/bin/env python3
"""
简化版FastAPI后端 - 用于测试和调试
"""

import os
import sys
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="分子毒性预测API (简化版)",
    description="用于测试的简化版API",
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
    return {"status": "healthy", "message": "简化版API服务正常运行"}

@app.post("/api/convert/xyz-to-npz")
async def convert_xyz_to_npz_api(file: UploadFile = File(...)):
    """XYZ转NPZ格式 (模拟)"""
    logger.info(f"收到XYZ转换请求: {file.filename}")
    
    if not file.filename.endswith('.xyz'):
        raise HTTPException(status_code=400, detail="只支持XYZ格式文件")
    
    try:
        # 读取文件内容
        content = await file.read()
        logger.info(f"文件大小: {len(content)} bytes")
        
        # 模拟转换过程
        return JSONResponse({
            "success": True,
            "message": f"成功转换 {file.filename} (模拟)",
            "data": {
                "file_id": "test_converted",
                "output_path": "/tmp/test_converted.npz"
            }
        })
        
    except Exception as e:
        logger.error(f"XYZ转NPZ错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"转换失败: {str(e)}"
        })

@app.post("/api/predict/binary")
async def predict_binary_api(
    file: UploadFile = File(...),
    model_path: str = Form("models/ToxPred_modelmini")
):
    """二元分类预测 (模拟)"""
    logger.info(f"收到二元分类预测请求: {file.filename}")
    logger.info(f"模型路径: {model_path}")
    
    try:
        # 读取文件内容
        content = await file.read()
        logger.info(f"文件大小: {len(content)} bytes")
        
        # 模拟预测过程
        import time
        time.sleep(2)  # 模拟计算时间
        
        # 模拟预测结果
        result_data = {
            "prediction": 0,
            "probability": 0.234,
            "confidence": "medium",
            "interpretation": "模型预测该分子无毒，置信度为0.234",
            "csv_path": "/tmp/test_predictions.csv",
            "total_predictions": 1,
            "positive_predictions": 0,
            "negative_predictions": 1
        }
        
        logger.info(f"预测完成 (模拟): {result_data}")
        
        return JSONResponse({
            "success": True,
            "message": "预测完成 (模拟)",
            "data": result_data
        })
        
    except Exception as e:
        logger.error(f"二元分类预测错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })

@app.post("/api/predict/property")
async def predict_property_api(
    file: UploadFile = File(...),
    model_path: str = Form("models/MD_model"),
    reference_path: str = Form("models/refscale.npz")
):
    """属性预测 (模拟)"""
    logger.info(f"收到属性预测请求: {file.filename}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"参考路径: {reference_path}")
    
    try:
        # 读取文件内容
        content = await file.read()
        logger.info(f"文件大小: {len(content)} bytes")
        
        # 模拟预测过程
        import time
        time.sleep(3)  # 模拟计算时间
        
        # 模拟属性预测结果
        properties = [
            {"name": "AtomNum", "value": 12.5, "unit": "", "description": "原子数量"},
            {"name": "Weight", "value": 180.16, "unit": "g/mol", "description": "分子量"},
            {"name": "HOMO", "value": -5.2, "unit": "eV", "description": "最高占据分子轨道"},
            {"name": "LUMO", "value": -2.1, "unit": "eV", "description": "最低未占据分子轨道"},
            {"name": "Dipole_Moment", "value": 2.3, "unit": "D", "description": "偶极矩"}
        ]
        
        result_data = {
            "properties": properties,
            "summary": {
                "toxicity_score": 0.35,
                "risk_level": "low",
                "recommendations": [
                    "建议进一步进行实验验证",
                    "关注分子的ADMET性质",
                    "考虑结构优化以降低毒性"
                ]
            },
            "csv_path": "/tmp/test_property_predictions.csv"
        }
        
        logger.info(f"属性预测完成 (模拟): {len(properties)}个属性")
        
        return JSONResponse({
            "success": True,
            "message": "属性预测完成 (模拟)",
            "data": result_data
        })
        
    except Exception as e:
        logger.error(f"属性预测错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })

@app.post("/api/visualize/molecule")
async def visualize_molecule_api(
    file: UploadFile = File(...),
    frame_index: int = Form(0),
    representation: str = Form("sticks"),
    rotation_x: float = Form(0),
    rotation_y: float = Form(0),
    rotation_z: float = Form(0),
    zoom: float = Form(1.0)
):
    """分子可视化 (模拟)"""
    logger.info(f"收到分子可视化请求: {file.filename}")
    logger.info(f"参数: frame={frame_index}, repr={representation}, rot=({rotation_x},{rotation_y},{rotation_z}), zoom={zoom}")
    
    try:
        # 读取文件内容
        content = await file.read()
        logger.info(f"文件大小: {len(content)} bytes")
        
        # 模拟可视化过程
        import time
        time.sleep(1)  # 模拟渲染时间
        
        # 创建一个简单的占位图像
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((150, 140), "分子结构占位图", fill='black')
        
        # 保存图像
        image_path = f"/tmp/molecule_viz_{frame_index}.png"
        img.save(image_path)
        
        result_data = {
            "image_path": image_path,
            "legend": f"分子: {file.filename}, 帧: {frame_index}",
            "total_frames": 5,  # 模拟多帧
            "current_frame": frame_index
        }
        
        logger.info(f"可视化完成 (模拟): {result_data}")
        
        return JSONResponse({
            "success": True,
            "message": "可视化完成 (模拟)",
            "data": result_data
        })
        
    except Exception as e:
        logger.error(f"分子可视化错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"可视化失败: {str(e)}"
        })

@app.post("/api/chat/configure")
async def configure_chat_api(
    base_url: str = Form(...),
    api_key: str = Form(...)
):
    """配置聊天API (模拟)"""
    logger.info(f"收到聊天配置请求: {base_url}")
    
    return JSONResponse({
        "success": True,
        "message": "聊天API配置成功 (模拟)"
    })

@app.post("/api/chat/message")
async def send_chat_message_api(
    message: str = Form(...),
    model_name: str = Form("google/gemini-2.0-flash-thinking-exp:free"),
    image_path: Optional[str] = Form(None)
):
    """发送聊天消息 (模拟)"""
    logger.info(f"收到聊天消息: {message[:50]}...")
    
    # 模拟AI回复
    response = f"这是对您消息的模拟回复: {message[:30]}... (使用模型: {model_name})"
    
    return JSONResponse({
        "success": True,
        "message": "消息发送成功 (模拟)",
        "data": {
            "response": response,
            "history": [["用户", message], ["AI", response]]
        }
    })

@app.get("/api/download/{file_path:path}")
async def download_file_api(file_path: str):
    """下载文件 (模拟)"""
    logger.info(f"收到下载请求: {file_path}")
    
    # 创建一个简单的文本文件作为示例
    temp_file = f"/tmp/{os.path.basename(file_path)}"
    with open(temp_file, 'w') as f:
        f.write(f"这是模拟的下载文件: {file_path}\n")
        f.write("实际使用时会返回真实的文件内容。\n")
    
    return FileResponse(
        temp_file,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    print("🚀 启动简化版分子毒性预测API服务器...")
    print("📱 前端地址: http://localhost:3000")
    print("🔧 API地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("⚠️  注意: 这是简化版，所有功能都是模拟的")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
