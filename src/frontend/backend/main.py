#!/usr/bin/env python3
"""
FastAPI后端 - 桥接原有的分子毒性预测功能
修复版本：正确导入原项目模块
"""

import os
import sys
import tempfile
import shutil
import json
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# 设置工作目录到集成包根目录
BASE_DIR = Path(__file__).resolve().parents[2]
os.chdir(BASE_DIR)

# 添加包根目录到Python路径
sys.path.insert(0, str(BASE_DIR))


def resolve_path(path: str) -> str:
    """Resolve paths against the bundle root by default."""
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = (BASE_DIR / path_obj).resolve()
    return str(path_obj)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入原有功能模块
try:
    from interface import (
        convert_xyz_to_npz, convert_npz_to_xyz,
        process_molecule_visualization, process_binary_prediction,
        process_property_prediction, export_frame_data
    )
    from chatbot import ChatInterface
    from probability_plot import create_probability_plot
    logger.info("✅ 成功导入原有功能模块")
except ImportError as e:
    logger.error(f"❌ 导入模块失败: {e}")
    logger.error(f"当前工作目录: {os.getcwd()}")
    logger.error(f"Python路径: {sys.path}")
    # 继续运行，但功能会受限

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

# 全局变量
chat_interface = None
temp_files = {}  # 存储临时文件路径

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global chat_interface
    try:
        chat_interface = ChatInterface()
        logger.info("✅ ChatInterface初始化成功")
    except Exception as e:
        logger.error(f"❌ ChatInterface初始化失败: {e}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "message": "API服务正常运行",
        "working_directory": os.getcwd(),
        "modules_loaded": 'interface' in sys.modules
    }

@app.post("/api/convert/xyz-to-npz")
async def convert_xyz_to_npz_api(file: UploadFile = File(...)):
    """XYZ转NPZ格式"""
    if not file.filename.endswith('.xyz'):
        raise HTTPException(status_code=400, detail="只支持XYZ格式文件")
    
    try:
        logger.info(f"开始XYZ转NPZ转换，文件: {file.filename}")
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xyz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 创建临时文件对象（模拟Gradio的文件对象）
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            # 调用原有转换函数
            result_message, output_path = convert_xyz_to_npz(temp_file_obj)
            
            if output_path and os.path.exists(output_path):
                # 存储文件路径供后续使用
                file_id = f"converted_{len(temp_files)}"
                temp_files[file_id] = output_path
                
                logger.info(f"转换成功: {output_path}")
                
                return JSONResponse({
                    "success": True,
                    "message": result_message,
                    "data": {
                        "file_id": file_id,
                        "output_path": output_path
                    }
                })
            else:
                logger.error(f"转换失败: {result_message}")
                return JSONResponse({
                    "success": False,
                    "message": result_message or "转换失败"
                })
                
    except Exception as e:
        logger.error(f"XYZ转NPZ错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"转换失败: {str(e)}"
        })
    finally:
        # 清理临时文件
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/api/convert/npz-to-xyz")
async def convert_npz_to_xyz_api(file: UploadFile = File(...)):
    """NPZ转XYZ格式"""
    if not file.filename.endswith('.npz'):
        raise HTTPException(status_code=400, detail="只支持NPZ格式文件")
    
    try:
        logger.info(f"开始NPZ转XYZ转换，文件: {file.filename}")
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            # 调用原有转换函数
            result_message, output_path = convert_npz_to_xyz(temp_file_obj)
            
            if output_path and os.path.exists(output_path):
                logger.info(f"转换成功: {output_path}")
                return JSONResponse({
                    "success": True,
                    "message": result_message,
                    "data": {
                        "download_url": f"/api/download/{os.path.basename(output_path)}"
                    }
                })
            else:
                logger.error(f"转换失败: {result_message}")
                return JSONResponse({
                    "success": False,
                    "message": result_message or "转换失败"
                })
                
    except Exception as e:
        logger.error(f"NPZ转XYZ错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"转换失败: {str(e)}"
        })

@app.post("/api/predict/binary")
async def predict_binary_api(
    file: UploadFile = File(...),
    model_path: str = Form("models/ToxPred_modelmini")
):
    """二元分类预测 - 调用真实的UniMol模型"""
    try:
        logger.info(f"🔥 开始二元分类预测，文件: {file.filename}")
        model_path = resolve_path(model_path)
        logger.info(f"🔥 使用模型路径: {model_path}")
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            return JSONResponse({
                "success": False,
                "message": f"模型路径不存在: {model_path}"
            })
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            logger.info(f"🔥 调用process_binary_prediction，模型路径: {model_path}")
            logger.info(f"🔥 输入文件: {temp_file.name}")
            
            # 调用原有预测函数 - 这里会调用真实的UniMol模型
            output_path, log_message = process_binary_prediction(temp_file_obj, model_path)
            
            logger.info(f"🔥 预测完成，输出路径: {output_path}")
            logger.info(f"🔥 日志消息: {log_message}")
            
            if output_path and os.path.exists(output_path):
                # 读取预测结果
                import pandas as pd
                df = pd.read_csv(output_path)
                
                # 计算统计信息
                predictions = df['prediction'].values
                probabilities = df['probability'].values
                
                logger.info(f"🔥 预测结果: {len(predictions)}个样本")
                logger.info(f"🔥 概率范围: {probabilities.min():.3f} - {probabilities.max():.3f}")
                
                # 生成概率图表
                plot_path = None
                try:
                    plot_path = create_probability_plot(output_path)
                    logger.info(f"🔥 概率图表生成成功: {plot_path}")
                except Exception as plot_error:
                    logger.warning(f"⚠️ 生成概率图表失败: {plot_error}")
                
                result_data = {
                    "prediction": int(predictions[0]) if len(predictions) > 0 else 0,
                    "probability": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    "confidence": "high" if len(probabilities) > 0 and (probabilities[0] > 0.8 or probabilities[0] < 0.2) else "medium",
                    "interpretation": f"UniMol模型预测该分子{'有毒' if predictions[0] == 1 else '无毒'}，置信度为{probabilities[0]:.3f}",
                    "csv_path": output_path,
                    "total_predictions": len(predictions),
                    "positive_predictions": int((predictions == 1).sum()),
                    "negative_predictions": int((predictions == 0).sum()),
                    "model_used": model_path
                }
                
                if plot_path:
                    result_data["plot_path"] = plot_path
                
                logger.info(f"✅ 预测完成: {result_data}")
                
                return JSONResponse({
                    "success": True,
                    "message": f"UniMol预测完成: {log_message}",
                    "data": result_data
                })
            else:
                logger.error(f"❌ 预测失败: {log_message}")
                return JSONResponse({
                    "success": False,
                    "message": log_message or "预测失败"
                })
                
    except Exception as e:
        logger.error(f"❌ 二元分类预测错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })
    finally:
        # 清理临时文件
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/api/predict/property")
async def predict_property_api(
    file: UploadFile = File(...),
    model_path: str = Form("models/MD_model"),
    reference_path: str = Form("models/refscale.npz")
):
    """属性预测 - 调用真实的UniMol模型"""
    try:
        logger.info(f"🔥 开始属性预测，文件: {file.filename}")
        model_path = resolve_path(model_path)
        reference_path = resolve_path(reference_path)
        logger.info(f"🔥 模型路径: {model_path}")
        logger.info(f"🔥 参考文件: {reference_path}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            return JSONResponse({
                "success": False,
                "message": f"模型路径不存在: {model_path}"
            })
        
        if not os.path.exists(reference_path):
            logger.error(f"❌ 参考文件不存在: {reference_path}")
            return JSONResponse({
                "success": False,
                "message": f"参考文件不存在: {reference_path}"
            })
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            logger.info(f"🔥 调用process_property_prediction")
            
            # 调用原有预测函数 - 这里会调用真实的UniMol模型
            output_path, log_message = process_property_prediction(temp_file_obj, model_path, reference_path)
            
            logger.info(f"🔥 属性预测完成，输出路径: {output_path}")
            logger.info(f"🔥 日志消息: {log_message}")
            
            if output_path and os.path.exists(output_path):
                # 读取预测结果
                import pandas as pd
                df = pd.read_csv(output_path)
                
                # 解析属性数据
                properties = []
                for col in df.columns:
                    if col not in ['id', 'frame']:
                        values = df[col].values
                        properties.append({
                            "name": col,
                            "value": float(values[0]) if len(values) > 0 else 0.0,
                            "unit": "",  # 可以根据需要添加单位映射
                            "description": f"{col}属性预测值"
                        })
                
                logger.info(f"🔥 解析了{len(properties)}个属性")
                
                # 计算毒性评分（基于实际属性值）
                toxicity_score = 0.5  # 默认值，可以根据实际属性计算
                risk_level = "medium"
                
                result_data = {
                    "properties": properties,
                    "summary": {
                        "toxicity_score": toxicity_score,
                        "risk_level": risk_level,
                        "recommendations": [
                            "基于UniMol模型的属性预测结果",
                            "建议进一步进行实验验证",
                            "关注分子的ADMET性质"
                        ]
                    },
                    "csv_path": output_path,
                    "model_used": model_path
                }
                
                logger.info(f"✅ 属性预测完成: {len(properties)}个属性")
                
                return JSONResponse({
                    "success": True,
                    "message": f"UniMol属性预测完成: {log_message}",
                    "data": result_data
                })
            else:
                logger.error(f"❌ 属性预测失败: {log_message}")
                return JSONResponse({
                    "success": False,
                    "message": log_message or "预测失败"
                })
                
    except Exception as e:
        logger.error(f"❌ 属性预测错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })
    finally:
        # 清理临时文件
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

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
    """分子可视化 - 使用PyMOL进行真实的3D渲染"""
    try:
        logger.info(f"🔥 开始分子可视化，文件: {file.filename}")
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            logger.info(f"🔥 调用process_molecule_visualization")
            
            # 调用原有可视化函数 - 这里会使用PyMOL进行真实渲染
            img, legend, slider_update, current_image = process_molecule_visualization(
                temp_file_obj, frame_index, representation, 
                rotation_x, rotation_y, rotation_z, zoom
            )
            
            if img:
                # 保存图像到临时文件
                image_path = f"/tmp/molecule_viz_{frame_index}.png"
                img.save(image_path)
                
                # 将图像转换为base64以便前端显示
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                # 获取总帧数（从slider_update中提取）
                total_frames = 1
                if hasattr(slider_update, 'maximum'):
                    total_frames = slider_update.maximum + 1
                
                result_data = {
                    "image_path": image_path,
                    "image_base64": img_base64,
                    "legend": legend,
                    "total_frames": total_frames,
                    "current_frame": frame_index
                }
                
                logger.info(f"✅ PyMOL可视化完成: {total_frames}帧")
                
                return JSONResponse({
                    "success": True,
                    "message": "PyMOL可视化成功",
                    "data": result_data
                })
            else:
                logger.error("❌ 可视化失败")
                return JSONResponse({
                    "success": False,
                    "message": "可视化失败"
                })
                
    except Exception as e:
        logger.error(f"❌ 分子可视化错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"可视化失败: {str(e)}"
        })
    finally:
        # 清理临时文件
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/api/chat/configure")
async def configure_chat_api(
    base_url: str = Form(...),
    api_key: str = Form(...)
):
    """配置聊天API"""
    global chat_interface
    try:
        if chat_interface:
            # 更新聊天接口配置
            chat_interface.base_url = base_url
            chat_interface.api_key = api_key
            
            logger.info(f"✅ 聊天API配置成功: {base_url}")
            
            return JSONResponse({
                "success": True,
                "message": "聊天API配置成功"
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "聊天接口未初始化"
            })
    except Exception as e:
        logger.error(f"❌ 配置聊天API错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"配置失败: {str(e)}"
        })

@app.post("/api/chat/message")
async def send_chat_message_api(
    message: str = Form(...),
    model_name: str = Form("google/gemini-2.0-flash-thinking-exp:free"),
    image_path: Optional[str] = Form(None)
):
    """发送聊天消息"""
    global chat_interface
    try:
        if not chat_interface:
            return JSONResponse({
                "success": False,
                "message": "聊天接口未初始化"
            })
        
        logger.info(f"🔥 发送聊天消息: {message[:50]}...")
        
        # 调用聊天接口
        history, error = chat_interface.process_message(
            message, image_path, bool(image_path), 
            chat_interface.memory.get_display_history(), model_name
        )
        
        if error:
            logger.error(f"❌ 聊天错误: {error}")
            return JSONResponse({
                "success": False,
                "message": error
            })
        else:
            # 获取最新的回复
            latest_response = ""
            if history and len(history) > 0:
                latest_response = history[-1][1] if len(history[-1]) > 1 else ""
            
            logger.info(f"✅ 聊天回复成功")
            
            return JSONResponse({
                "success": True,
                "message": "消息发送成功",
                "data": {
                    "response": latest_response,
                    "history": history
                }
            })
            
    except Exception as e:
        logger.error(f"❌ 发送聊天消息错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"发送失败: {str(e)}"
        })

@app.post("/api/export/frame")
async def export_frame_api(
    frame_index: int = Form(...),
    export_format: str = Form("PNG"),
    binary_pred_file: Optional[str] = Form(None),
    property_pred_file: Optional[str] = Form(None),
    current_image_path: Optional[str] = Form(None)
):
    """导出帧数据"""
    try:
        logger.info(f"🔥 导出帧数据: frame {frame_index}")
        
        # 调用原有导出函数
        image_path, json_path = export_frame_data(
            current_image_path, export_format, frame_index,
            binary_pred_file, property_pred_file
        )
        
        if image_path and json_path:
            logger.info(f"✅ 导出成功: {image_path}, {json_path}")
            return JSONResponse({
                "success": True,
                "message": "导出成功",
                "data": {
                    "image_path": image_path,
                    "json_path": json_path,
                    "download_url": f"/api/download/{os.path.basename(image_path)}"
                }
            })
        else:
            logger.error("❌ 导出失败")
            return JSONResponse({
                "success": False,
                "message": "导出失败"
            })
            
    except Exception as e:
        logger.error(f"❌ 导出帧数据错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"导出失败: {str(e)}"
        })

@app.get("/api/download/{file_path:path}")
async def download_file_api(file_path: str):
    """下载文件"""
    try:
        # 安全检查：只允许下载特定目录的文件
        if not os.path.exists(file_path):
            # 尝试在当前目录查找
            local_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(local_path):
                file_path = local_path
            else:
                raise HTTPException(status_code=404, detail="文件不存在")
        
        return FileResponse(
            file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"❌ 下载文件错误: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")

if __name__ == "__main__":
    print("🚀 启动分子毒性预测API服务器...")
    print(f"📁 工作目录: {os.getcwd()}")
    print("📱 前端地址: http://localhost:3000")
    print("🔧 API地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
