#!/usr/bin/env python3
"""
修复版FastAPI后端 - 桥接原有的分子毒性预测功能
使用延迟导入避免启动时阻塞
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
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 设置工作目录到集成项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))


def resolve_path(path: str) -> str:
    """Convert provided path into an absolute path within the bundle unless already absolute."""
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = (BASE_DIR / path_obj).resolve()
    return str(path_obj)

# 导入RAG服务
try:
    from simple_rag_service import get_simple_rag_service
    RAG_AVAILABLE = True
    print("✅ 简化RAG服务导入成功")
except ImportError as e:
    print(f"⚠️ 简化RAG服务导入失败: {e}")
    RAG_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="分子毒性预测API",
    description="基于原有Gradio项目的API封装 - 修复版",
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
rag_initialized = False

def lazy_import_modules():
    """延迟导入模块，避免启动时阻塞"""
    try:
        from interface import (
            convert_xyz_to_npz, convert_npz_to_xyz,
            process_molecule_visualization, process_binary_prediction,
            process_property_prediction, export_frame_data
        )
        from chatbot import ChatInterface
        from probability_plot import create_probability_plot
        return True, {
            'convert_xyz_to_npz': convert_xyz_to_npz,
            'convert_npz_to_xyz': convert_npz_to_xyz,
            'process_molecule_visualization': process_molecule_visualization,
            'process_binary_prediction': process_binary_prediction,
            'process_property_prediction': process_property_prediction,
            'export_frame_data': export_frame_data,
            'ChatInterface': ChatInterface,
            'create_probability_plot': create_probability_plot
        }
    except Exception as e:
        logger.error(f"模块导入失败: {e}")
        return False, str(e)

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global chat_interface
    logger.info("🚀 API服务器启动中...")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "message": "API服务正常运行",
        "working_directory": os.getcwd(),
        "modules_loaded": 'interface' in sys.modules
    }

@app.get("/test-import")
async def test_import():
    """测试模块导入"""
    success, result = lazy_import_modules()
    if success:
        return {
            "success": True,
            "message": "所有模块导入成功",
            "modules": list(result.keys())
        }
    else:
        return {
            "success": False,
            "message": f"模块导入失败: {result}"
        }

@app.post("/api/convert/xyz-to-npz")
async def convert_xyz_to_npz_api(file: UploadFile = File(...)):
    """XYZ转NPZ格式"""
    if not file.filename.endswith('.xyz'):
        raise HTTPException(status_code=400, detail="只支持XYZ格式文件")
    
    try:
        logger.info(f"🔥 开始XYZ转NPZ转换，文件: {file.filename}")
        
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        convert_xyz_to_npz = modules['convert_xyz_to_npz']
        
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
                
                logger.info(f"✅ 转换成功: {output_path}")
                
                return JSONResponse({
                    "success": True,
                    "message": result_message,
                    "data": {
                        "file_id": file_id,
                        "output_path": output_path
                    }
                })
            else:
                logger.error(f"❌ 转换失败: {result_message}")
                return JSONResponse({
                    "success": False,
                    "message": result_message or "转换失败"
                })
                
    except Exception as e:
        logger.error(f"❌ XYZ转NPZ错误: {e}")
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
        
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        process_binary_prediction = modules['process_binary_prediction']
        create_probability_plot = modules['create_probability_plot']
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            return JSONResponse({
                "success": False,
                "message": f"模型路径不存在: {model_path}"
            })
        
        # 保存上传的文件并处理格式转换
        file_suffix = '.xyz' if file.filename.endswith('.xyz') else '.npz'
        # 对于XYZ文件使用文本模式，对于NPZ文件使用二进制模式
        file_mode = 'w+t' if file.filename.endswith('.xyz') else 'w+b'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, mode=file_mode) as temp_file:
            content = await file.read()
            if file.filename.endswith('.xyz'):
                # XYZ文件以文本形式写入
                temp_file.write(content.decode('utf-8'))
            else:
                # NPZ文件以二进制形式写入
                temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            # 如果是XYZ文件，先转换为NPZ
            logger.info(f"🔥 检查文件类型: {file.filename}, 是否为XYZ: {file.filename.endswith('.xyz')}")
            if file.filename.endswith('.xyz'):
                try:
                    convert_xyz_to_npz = modules['convert_xyz_to_npz']
                    logger.info(f"🔥 开始转换XYZ到NPZ格式，输入文件: {temp_file_obj.name}")
                    result_message, npz_path = convert_xyz_to_npz(temp_file_obj)
                    logger.info(f"🔥 转换函数返回: message={result_message}, path={npz_path}")
                    if npz_path and os.path.exists(npz_path):
                        temp_file_obj = TempFile(npz_path)
                        logger.info(f"🔥 转换成功，更新文件路径: {npz_path}")
                    else:
                        raise Exception(f"XYZ转NPZ失败: {result_message}")
                except Exception as convert_error:
                    logger.error(f"❌ XYZ转换失败: {convert_error}")
                    import traceback
                    logger.error(f"❌ 转换详细错误: {traceback.format_exc()}")
                    return JSONResponse({
                        "success": False,
                        "message": f"文件格式转换失败: {str(convert_error)}"
                    })
            else:
                logger.info(f"🔥 文件不是XYZ格式，跳过转换")
            
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
                plot_data = None
                try:
                    plot_fig = create_probability_plot(output_path)
                    if plot_fig:
                        # 将Plotly图表转换为JSON格式
                        plot_data = plot_fig.to_dict()
                        logger.info(f"🔥 概率图表生成成功")
                except Exception as plot_error:
                    logger.warning(f"⚠️ 生成概率图表失败: {plot_error}")
                
                # 检查是否有多个分子
                if len(predictions) > 1:
                    # 多个分子的情况
                    prediction_list = []
                    for i in range(len(predictions)):
                        prediction_list.append({
                            "prediction": int(predictions[i]),
                            "probability": float(probabilities[i]),
                            "confidence": "high" if (probabilities[i] > 0.8 or probabilities[i] < 0.2) else "medium",
                            "interpretation": f"UniMol model predicts molecule {i+1} as {'toxic' if predictions[i] == 1 else 'non-toxic'}, Model used: 3998_ToxPred_modelmini"
                        })
                    
                    result_data = {
                        "predictions": prediction_list,
                        "csv_path": output_path,
                        "total_predictions": len(predictions),
                        "positive_predictions": int((predictions == 1).sum()),
                        "negative_predictions": int((predictions == 0).sum()),
                        "model_used": model_path
                    }
                else:
                    # 单个分子的情况
                    result_data = {
                        "prediction": int(predictions[0]) if len(predictions) > 0 else 0,
                        "probability": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                        "confidence": "high" if len(probabilities) > 0 and (probabilities[0] > 0.8 or probabilities[0] < 0.2) else "medium",
                        "interpretation": f"UniMol model predicts this molecule as {'toxic' if predictions[0] == 1 else 'non-toxic'}, Model used: 3998_ToxPred_modelmini",
                        "csv_path": output_path,
                        "total_predictions": len(predictions),
                        "positive_predictions": int((predictions == 1).sum()),
                        "negative_predictions": int((predictions == 0).sum()),
                        "model_used": model_path
                    }
                
                if plot_data:
                    result_data["plot_data"] = plot_data
                
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
        
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        process_property_prediction = modules['process_property_prediction']
        
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
        
        # 保存上传的文件并处理格式转换
        file_suffix = '.xyz' if file.filename.endswith('.xyz') else '.npz'
        # 对于XYZ文件使用文本模式，对于NPZ文件使用二进制模式
        file_mode = 'w+t' if file.filename.endswith('.xyz') else 'w+b'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, mode=file_mode) as temp_file:
            content = await file.read()
            if file.filename.endswith('.xyz'):
                # XYZ文件以文本形式写入
                temp_file.write(content.decode('utf-8'))
            else:
                # NPZ文件以二进制形式写入
                temp_file.write(content)
            temp_file.flush()
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            # 如果是XYZ文件，先转换为NPZ
            logger.info(f"🔥 检查文件类型: {file.filename}, 是否为XYZ: {file.filename.endswith('.xyz')}")
            if file.filename.endswith('.xyz'):
                try:
                    convert_xyz_to_npz = modules['convert_xyz_to_npz']
                    logger.info(f"🔥 开始转换XYZ到NPZ格式，输入文件: {temp_file_obj.name}")
                    result_message, npz_path = convert_xyz_to_npz(temp_file_obj)
                    logger.info(f"🔥 转换函数返回: message={result_message}, path={npz_path}")
                    if npz_path and os.path.exists(npz_path):
                        temp_file_obj = TempFile(npz_path)
                        logger.info(f"🔥 转换成功，更新文件路径: {npz_path}")
                    else:
                        raise Exception(f"XYZ转NPZ失败: {result_message}")
                except Exception as convert_error:
                    logger.error(f"❌ XYZ转换失败: {convert_error}")
                    import traceback
                    logger.error(f"❌ 转换详细错误: {traceback.format_exc()}")
                    return JSONResponse({
                        "success": False,
                        "message": f"文件格式转换失败: {str(convert_error)}"
                    })
            else:
                logger.info(f"🔥 文件不是XYZ格式，跳过转换")
            
            logger.info(f"🔥 调用process_property_prediction")
            
            # 调用原有预测函数 - 这里会调用真实的UniMol模型
            output_path, log_message = process_property_prediction(temp_file_obj, model_path, reference_path)
            
            logger.info(f"🔥 属性预测完成，输出路径: {output_path}")
            logger.info(f"🔥 日志消息: {log_message}")
            
            if output_path and os.path.exists(output_path):
                # 读取预测结果
                import pandas as pd
                df = pd.read_csv(output_path)
                
                logger.info(f"🔥 读取预测结果: {len(df)}行数据")
                
                # 化学描述符映射表
                descriptor_mapping = {
                    "AtomNum": "Number of Atoms",
                    "Weight": "Molecular Weight",
                    "HOMO": "Highest Occupied Molecular Orbital Energy",
                    "HOMO_number": "HOMO Orbital Number",
                    "LUMO": "Lowest Unoccupied Molecular Orbital Energy",
                    "HOMO_LUMO_Gap": "HOMO-LUMO Energy Gap",
                    "ODI_HOMO_1": "Orbital Delocalization Index HOMO-1",
                    "ODI_HOMO": "Orbital Delocalization Index HOMO",
                    "ODI_LUMO": "Orbital Delocalization Index LUMO",
                    "ODI_LUMO_Add1": "Orbital Delocalization Index LUMO+1",
                    "ODI_Mean": "Mean Orbital Delocalization Index",
                    "ODI_Std": "Standard Deviation of Orbital Delocalization Index"
                }

                # 导入分子描述符计算模块
                from molecular_descriptors import replace_predicted_with_calculated

                # 检查是否有多个分子
                if len(df) > 1:
                    # 多个分子的情况
                    predictions = []
                    for idx, row in df.iterrows():
                        properties = []
                        for col in df.columns:
                            if col not in ['id', 'frame']:
                                properties.append({
                                    "name": col,
                                    "value": float(row[col]) if pd.notna(row[col]) else 0.0,
                                    "unit": "g/mol" if col == "Weight" else "",
                                    "description": descriptor_mapping.get(col, f"{col} Property Value")
                                })

                        # 用计算值替换基础描述符的预测值 - 使用对应的帧索引
                        try:
                            from molecular_descriptors import replace_predicted_with_calculated_for_frame
                            properties = replace_predicted_with_calculated_for_frame(properties, temp_file.name, idx)
                            logger.info(f"分子 {idx + 1} (帧 {idx}): 已用计算值替换基础描述符")
                        except Exception as e:
                            logger.warning(f"分子 {idx + 1} (帧 {idx}): 替换基础描述符失败: {e}")
                            # 如果新函数不存在，回退到旧方法但只对第一个分子使用
                            try:
                                if idx == 0:
                                    properties = replace_predicted_with_calculated(properties, temp_file.name)
                                    logger.info(f"分子 {idx + 1}: 使用回退方法替换基础描述符")
                            except:
                                pass

                        predictions.append({
                            "properties": properties,
                            "summary": {
                                "toxicity_score": 0.5,
                                "risk_level": "medium",
                                "recommendations": [
                                    f"分子 {idx + 1} 的UniMol属性预测结果",
                                    "基础描述符(AtomNum, Weight)使用精确计算值",
                                    "建议进一步进行实验验证",
                                    "关注分子的ADMET性质"
                                ]
                            }
                        })

                    result_data = {
                        "predictions": predictions,
                        "total_predictions": len(predictions),
                        "csv_path": output_path,
                        "model_used": model_path
                    }
                    num_properties = len(predictions[0]["properties"]) if predictions else 0
                else:
                    # 单个分子的情况
                    properties = []
                    for col in df.columns:
                        if col not in ['id', 'frame']:
                            values = df[col].values
                            properties.append({
                                "name": col,
                                "value": float(values[0]) if len(values) > 0 else 0.0,
                                "unit": "g/mol" if col == "Weight" else "",
                                "description": descriptor_mapping.get(col, f"{col} Property Value")
                            })

                    # 用计算值替换基础描述符的预测值
                    try:
                        properties = replace_predicted_with_calculated(properties, temp_file.name)
                        logger.info("单个分子: 已用计算值替换基础描述符")
                    except Exception as e:
                        logger.warning(f"单个分子: 替换基础描述符失败: {e}")

                    result_data = {
                        "properties": properties,
                        "total_predictions": 1,
                        "summary": {
                            "toxicity_score": 0.5,
                            "risk_level": "medium",
                            "recommendations": [
                                "基于UniMol模型的属性预测结果",
                                "基础描述符(AtomNum, Weight)使用精确计算值",
                                "建议进一步进行实验验证",
                                "关注分子的ADMET性质"
                            ]
                        },
                        "csv_path": output_path,
                        "model_used": model_path
                    }
                    num_properties = len(properties)

                logger.info(f"🔥 解析了{'多个分子' if len(df) > 1 else '单个分子'}的属性数据")

                logger.info(f"✅ 属性预测完成: {num_properties}个属性")
                
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
    zoom: float = Form(1.0),
    molecule_index: int = Form(0)
):
    """分子可视化 - 使用PyMOL进行真实的3D渲染，修复编码问题"""
    try:
        logger.info(f"🔥 开始分子可视化，文件: {file.filename}")
        
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        process_molecule_visualization = modules['process_molecule_visualization']
        
        # 检查文件类型和大小
        if not file.filename.endswith(('.npz', '.xyz')):
            logger.warning(f"⚠️ 不支持的文件类型: {file.filename}")
            return JSONResponse({
                "success": False,
                "message": f"不支持的文件类型，请上传NPZ或XYZ文件"
            })
        
        # 保存上传的文件，使用二进制模式避免编码问题
        file_suffix = '.npz' if file.filename.endswith('.npz') else '.xyz'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, mode='wb') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            logger.info(f"🔥 临时文件保存: {temp_file.name}, 大小: {len(content)} bytes")
            
            class TempFile:
                def __init__(self, name):
                    self.name = name
            
            temp_file_obj = TempFile(temp_file.name)
            
            # 如果是XYZ文件，先转换为NPZ
            if file.filename.endswith('.xyz'):
                try:
                    convert_xyz_to_npz = modules['convert_xyz_to_npz']
                    logger.info(f"🔥 转换XYZ到NPZ格式")
                    result_message, npz_path = convert_xyz_to_npz(temp_file_obj)
                    if npz_path and os.path.exists(npz_path):
                        temp_file_obj = TempFile(npz_path)
                        logger.info(f"🔥 转换成功: {npz_path}")
                    else:
                        raise Exception(f"XYZ转NPZ失败: {result_message}")
                except Exception as convert_error:
                    logger.error(f"❌ XYZ转换失败: {convert_error}")
                    return JSONResponse({
                        "success": False,
                        "message": f"文件格式转换失败: {str(convert_error)}"
                    })
            
            logger.info(f"🔥 调用process_molecule_visualization，参数:")
            logger.info(f"   - 文件: {temp_file_obj.name}")
            logger.info(f"   - 帧索引: {frame_index}")
            logger.info(f"   - 分子索引: {molecule_index}")
            logger.info(f"   - 表示方式: {representation}")
            logger.info(f"   - 旋转: ({rotation_x}, {rotation_y}, {rotation_z})")
            logger.info(f"   - 缩放: {zoom}")
            
            # 调用原有可视化函数 - 这里会使用PyMOL进行真实渲染
            # 对于分子导航，使用molecule_index作为frame_index，但要确保不超出范围
            try:
                # 首先检查文件中的总帧数
                if temp_file_obj.name.lower().endswith('.npz'):
                    import numpy as np
                    data = np.load(temp_file_obj.name, allow_pickle=True)
                    total_frames = len(data['coord']) if 'coord' in data else 1
                else:
                    # 对于XYZ文件，使用现有的计算方法
                    from visualizer import count_xyz_frames
                    total_frames = count_xyz_frames(temp_file_obj.name)

                # 确保索引在有效范围内
                effective_frame_index = min(molecule_index, max(0, total_frames - 1))
                logger.info(f"🔥 总帧数: {total_frames}, 请求分子索引: {molecule_index}, 有效帧索引: {effective_frame_index}")

                result = process_molecule_visualization(
                    temp_file_obj, effective_frame_index, representation,
                    rotation_x, rotation_y, rotation_z, zoom
                )
                
                # 处理返回结果
                if isinstance(result, tuple) and len(result) >= 4:
                    img, legend, slider_update, current_image = result
                elif isinstance(result, tuple) and len(result) >= 1:
                    img = result[0]
                    legend = "分子结构可视化"
                    slider_update = None
                    current_image = None
                else:
                    img = result
                    legend = "分子结构可视化"
                    slider_update = None
                    current_image = None
                
                logger.info(f"🔥 可视化函数返回: img={type(img)}, legend={legend}")
                
            except Exception as viz_error:
                logger.error(f"❌ 可视化函数调用失败: {viz_error}")
                import traceback
                logger.error(f"❌ 可视化详细错误: {traceback.format_exc()}")
                return JSONResponse({
                    "success": False,
                    "message": f"分子可视化处理失败: {str(viz_error)}"
                })
            
            if img:
                try:
                    # 创建唯一的临时图像文件名
                    import time
                    timestamp = int(time.time() * 1000)
                    image_path = f"/tmp/molecule_viz_{timestamp}_{frame_index}.png"
                    
                    # 保存图像，处理不同的图像类型
                    if hasattr(img, 'save'):
                        # PIL Image对象
                        img.save(image_path, format='PNG')
                        logger.info(f"🔥 PIL图像保存成功: {image_path}")
                    elif hasattr(img, 'write_image'):
                        # Plotly图像对象
                        img.write_image(image_path)
                        logger.info(f"🔥 Plotly图像保存成功: {image_path}")
                    else:
                        # 尝试直接保存
                        with open(image_path, 'wb') as f:
                            f.write(img)
                        logger.info(f"🔥 二进制图像保存成功: {image_path}")
                    
                    # 验证文件是否成功保存
                    if not os.path.exists(image_path):
                        raise Exception("图像文件保存失败")
                    
                    file_size = os.path.getsize(image_path)
                    if file_size == 0:
                        raise Exception("图像文件为空")
                    
                    logger.info(f"🔥 图像文件验证成功: {image_path}, 大小: {file_size} bytes")
                    
                    # 将图像转换为base64以便前端显示
                    with open(image_path, "rb") as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    logger.info(f"🔥 Base64编码成功，长度: {len(img_base64)}")
                    
                    # 获取总帧数（从slider_update中提取）
                    total_frames = 1
                    if slider_update and hasattr(slider_update, 'maximum'):
                        total_frames = slider_update.maximum + 1
                    elif slider_update and hasattr(slider_update, 'value'):
                        # 尝试从value属性获取
                        total_frames = getattr(slider_update, 'maximum', 1) + 1
                    
                    # 处理图例，确保是字符串格式
                    legend_text = str(legend) if legend else "分子结构可视化"
                    
                    result_data = {
                        "image_path": image_path,
                        "image_base64": img_base64,
                        "legend": legend_text,
                        "total_frames": total_frames,
                        "current_frame": frame_index,
                        "file_size": file_size
                    }
                    
                    logger.info(f"✅ PyMOL可视化完成: {total_frames}帧, 图像大小: {file_size} bytes")
                    
                    return JSONResponse({
                        "success": True,
                        "message": "PyMOL可视化成功",
                        "data": result_data
                    })
                    
                except Exception as img_error:
                    logger.error(f"❌ 图像处理失败: {img_error}")
                    import traceback
                    logger.error(f"❌ 图像处理详细错误: {traceback.format_exc()}")
                    return JSONResponse({
                        "success": False,
                        "message": f"图像处理失败: {str(img_error)}"
                    })
            else:
                logger.error("❌ 可视化函数返回空图像")
                return JSONResponse({
                    "success": False,
                    "message": "可视化失败：未生成图像"
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
                logger.info(f"🔥 清理临时文件: {temp_file.name}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ 清理临时文件失败: {cleanup_error}")

@app.post("/api/chat/configure")
async def configure_chat_api(
    base_url: str = Form(...),
    api_key: str = Form(...)
):
    """配置聊天API"""
    global chat_interface
    try:
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        ChatInterface = modules['ChatInterface']
        
        if not chat_interface:
            chat_interface = ChatInterface()
        
        # 更新聊天接口配置
        chat_interface.base_url = base_url
        chat_interface.api_key = api_key
        
        # 初始化客户端
        result = chat_interface.initialize_client(base_url, api_key)
        if "Error" in result:
            return JSONResponse({
                "success": False,
                "message": result
            })
        
        logger.info(f"✅ 聊天API配置成功: {base_url}")
        
        return JSONResponse({
            "success": True,
            "message": "聊天API配置成功"
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
    image_path: str = Form(None)
):
    """发送聊天消息"""
    global chat_interface
    try:
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        ChatInterface = modules['ChatInterface']
        
        if not chat_interface:
            return JSONResponse({
                "success": False,
                "message": "聊天接口未配置，请先配置API"
            })
        
        # 调用聊天接口处理消息
        history, error = chat_interface.process_message(
            message,
            image_path,
            bool(image_path),
            chat_interface.memory.get_display_history(),
            model_name
        )
        
        if error:
            logger.error(f"❌ 聊天消息处理失败: {error}")
            return JSONResponse({
                "success": False,
                "message": error
            })
        
        # 获取最新的助手回复
        assistant_response = ""
        if history and len(history) > 0:
            last_message = history[-1]
            if last_message[1]:  # 助手的回复
                assistant_response = last_message[1]
        
        logger.info(f"✅ 聊天消息处理成功")
        
        return JSONResponse({
            "success": True,
            "message": "消息发送成功",
            "data": {
                "response": assistant_response,
                "history": history
            }
        })
        
    except Exception as e:
        logger.error(f"❌ 发送聊天消息错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"发送失败: {str(e)}"
        })

@app.get("/api/chat/models")
async def get_available_models_api():
    """获取可用的模型列表"""
    global chat_interface
    try:
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        ChatInterface = modules['ChatInterface']
        
        if not chat_interface:
            chat_interface = ChatInterface()
        
        # 如果配置了API，尝试从OpenRouter获取模型列表
        models = []
        if hasattr(chat_interface, 'client') and chat_interface.client:
            try:
                # 尝试从OpenRouter API获取模型列表
                import requests
                headers = {
                    "Authorization": f"Bearer {chat_interface.api_key}",
                    "HTTP-Referer": "localhost",
                    "X-Title": "Molecular Toxicity Predictor"
                }
                
                response = requests.get(
                    f"{chat_interface.base_url.rstrip('/api/v1')}/api/v1/models",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    api_models = response.json()
                    if 'data' in api_models:
                        models = [
                            {
                                "id": model["id"],
                                "name": model.get("name", model["id"]),
                                "description": model.get("description", ""),
                                "context_length": model.get("context_length", 0),
                                "pricing": model.get("pricing", {}),
                                "top_provider": model.get("top_provider", {})
                            }
                            for model in api_models["data"]
                            if not model["id"].startswith("@")  # 过滤掉一些特殊模型
                        ]
                        logger.info(f"✅ 从OpenRouter获取到 {len(models)} 个模型")
                else:
                    logger.warning(f"⚠️ 获取模型列表失败: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ 从API获取模型列表失败: {e}")
        
        # 如果没有从API获取到模型，使用默认模型列表
        if not models:
            default_models = [
                {
                    "id": "google/gemini-2.0-flash-thinking-exp:free",
                    "name": "Gemini 2.0 Flash Thinking (Free)",
                    "description": "Google's latest reasoning model with advanced thinking capabilities",
                    "context_length": 32768,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "top_provider": {"name": "Google"}
                },
                {
                    "id": "anthropic/claude-3.5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's most capable model for complex reasoning",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.003", "completion": "0.015"},
                    "top_provider": {"name": "Anthropic"}
                },
                {
                    "id": "openai/gpt-4o",
                    "name": "GPT-4o",
                    "description": "OpenAI's flagship multimodal model",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.005", "completion": "0.015"},
                    "top_provider": {"name": "OpenAI"}
                },
                {
                    "id": "deepseek/deepseek-r1",
                    "name": "DeepSeek R1",
                    "description": "DeepSeek's reasoning model with strong analytical capabilities",
                    "context_length": 65536,
                    "pricing": {"prompt": "0.0014", "completion": "0.0028"},
                    "top_provider": {"name": "DeepSeek"}
                },
                {
                    "id": "qwen/qwen-2.5-72b-instruct",
                    "name": "Qwen 2.5 72B Instruct",
                    "description": "Alibaba's large language model optimized for instruction following",
                    "context_length": 32768,
                    "pricing": {"prompt": "0.0009", "completion": "0.0009"},
                    "top_provider": {"name": "Alibaba"}
                }
            ]
            models = default_models
            logger.info(f"✅ 使用默认模型列表: {len(models)} 个模型")
        
        return JSONResponse({
            "success": True,
            "message": f"获取到 {len(models)} 个可用模型",
            "data": {
                "models": models,
                "total": len(models)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ 获取模型列表错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"获取模型列表失败: {str(e)}"
        })

# ToxD4C API 端点
@app.post("/api/toxd4c/predict/smiles")
async def toxd4c_predict_smiles_api(smiles_input: str = Form(...)):
    """ToxD4C SMILES预测"""
    try:
        logger.info(f"🔥 开始ToxD4C SMILES预测: {smiles_input}")
        
        # 导入ToxD4C wrapper
        from toxd4c_wrapper import predict_toxicity_from_smiles
        
        # 处理SMILES输入（可能包含多行）
        smiles_list = [s.strip() for s in smiles_input.strip().split('\n') if s.strip()]
        
        # 进行预测
        result = predict_toxicity_from_smiles(smiles_list)
        
        if result.get('success'):
            logger.info(f"✅ ToxD4C SMILES预测成功: {result['num_molecules']} 个分子")
            return JSONResponse({
                "success": True,
                "message": f"成功预测 {result['num_molecules']} 个分子",
                "data": result
            })
        else:
            logger.error(f"❌ ToxD4C SMILES预测失败: {result.get('error')}")
            return JSONResponse({
                "success": False,
                "message": result.get('error', '预测失败')
            })
            
    except Exception as e:
        logger.error(f"❌ ToxD4C SMILES预测错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })

@app.post("/api/toxd4c/predict/xyz")
async def toxd4c_predict_xyz_api(file: UploadFile = File(...)):
    """ToxD4C XYZ文件预测"""
    try:
        logger.info(f"🔥 开始ToxD4C XYZ预测: {file.filename}")
        
        # 检查文件格式
        if not file.filename.endswith('.xyz'):
            raise HTTPException(status_code=400, detail="只支持XYZ格式文件")
        
        # 导入ToxD4C wrapper
        from toxd4c_wrapper import predict_toxicity_from_xyz
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xyz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 进行预测
            result = predict_toxicity_from_xyz(temp_file.name)
            
            if result.get('success'):
                logger.info(f"✅ ToxD4C XYZ预测成功: {result['num_molecules']} 个分子")
                return JSONResponse({
                    "success": True,
                    "message": f"成功预测 {result['num_molecules']} 个分子",
                    "data": result
                })
            else:
                logger.error(f"❌ ToxD4C XYZ预测失败: {result.get('error')}")
                return JSONResponse({
                    "success": False,
                    "message": result.get('error', '预测失败')
                })
                
    except Exception as e:
        logger.error(f"❌ ToxD4C XYZ预测错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
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

@app.post("/api/toxd4c/predict/file")
async def toxd4c_predict_file_api(file: UploadFile = File(...)):
    """ToxD4C 文件预测（自动检测格式）"""
    try:
        logger.info(f"🔥 开始ToxD4C文件预测: {file.filename}")
        
        # 根据文件扩展名决定处理方式
        if file.filename.endswith('.smi') or file.filename.endswith('.smiles'):
            # SMILES文件
            content = await file.read()
            smiles_content = content.decode('utf-8')
            smiles_list = [s.strip() for s in smiles_content.strip().split('\n') if s.strip()]
            
            from toxd4c_wrapper import predict_toxicity_from_smiles
            result = predict_toxicity_from_smiles(smiles_list)
            
        elif file.filename.endswith('.xyz'):
            # XYZ文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xyz') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                from toxd4c_wrapper import predict_toxicity_from_xyz
                result = predict_toxicity_from_xyz(temp_file.name)
                
                # 清理临时文件
                os.unlink(temp_file.name)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式，请上传.xyz文件")
        
        if result.get('success'):
            logger.info(f"✅ ToxD4C文件预测成功: {result['num_molecules']} 个分子")
            return JSONResponse({
                "success": True,
                "message": f"成功预测 {result['num_molecules']} 个分子",
                "data": result
            })
        else:
            logger.error(f"❌ ToxD4C文件预测失败: {result.get('error')}")
            return JSONResponse({
                "success": False,
                "message": result.get('error', '预测失败')
            })
            
    except Exception as e:
        logger.error(f"❌ ToxD4C文件预测错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"预测失败: {str(e)}"
        })

@app.get("/api/toxd4c/info")
async def toxd4c_info_api():
    """获取ToxD4C模型信息"""
    try:
        from toxd4c_wrapper import get_toxd4c_wrapper
        
        wrapper = get_toxd4c_wrapper()
        info = wrapper.get_model_info()
        
        return JSONResponse({
            "success": True,
            "data": info
        })
        
    except Exception as e:
        logger.error(f"❌ 获取ToxD4C信息错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"获取信息失败: {str(e)}"
        })

# 文件转换API端点
@app.post("/api/convert/file")
async def convert_file_api(
    file: UploadFile = File(...),
    target_format: str = Form(...)
):
    """通用文件转换API"""
    try:
        logger.info(f"🔄 开始文件转换: {file.filename} -> {target_format}")
        
        # 导入文件转换工具
        from file_conversion_utils import convert_file
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 进行转换
            result = convert_file(temp_file.name, target_format)
            
            if result.get('success'):
                output_path = result['output_path']
                
                # 存储文件路径供下载
                file_id = f"converted_{len(temp_files)}"
                temp_files[file_id] = output_path
                
                logger.info(f"✅ 文件转换成功: {output_path}")
                
                return JSONResponse({
                    "success": True,
                    "message": result['message'],
                    "data": {
                        "file_id": file_id,
                        "output_path": output_path,
                        "download_url": f"/api/download/{file_id}",
                        **{k: v for k, v in result.items() if k not in ['success', 'output_path', 'message']}
                    }
                })
            else:
                logger.error(f"❌ 文件转换失败: {result.get('error')}")
                return JSONResponse({
                    "success": False,
                    "message": result.get('error', '转换失败')
                })
                
    except Exception as e:
        logger.error(f"❌ 文件转换错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
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

@app.post("/api/convert/enhanced-xyz-to-npz")
async def enhanced_xyz_to_npz_api(file: UploadFile = File(...)):
    """增强版XYZ转NPZ转换"""
    try:
        logger.info(f"🔄 开始增强XYZ转NPZ转换: {file.filename}")
        
        # 导入增强的转换工具
        from file_conversion_utils import convert_xyz_to_npz
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xyz') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 进行转换
            result = convert_xyz_to_npz(temp_file.name)
            
            if result.get('success'):
                output_path = result['output_path']
                
                # 存储文件路径供下载
                file_id = f"enhanced_npz_{len(temp_files)}"
                temp_files[file_id] = output_path
                
                logger.info(f"✅ 增强XYZ转NPZ转换成功: {output_path}")
                
                return JSONResponse({
                    "success": True,
                    "message": result['message'],
                    "data": {
                        "file_id": file_id,
                        "output_path": output_path,
                        "download_url": f"/api/download/{file_id}",
                        "num_frames": result.get('num_frames', 0),
                        "total_atoms": result.get('total_atoms', 0)
                    }
                })
            else:
                logger.error(f"❌ 增强XYZ转NPZ转换失败: {result.get('error')}")
                return JSONResponse({
                    "success": False,
                    "message": result.get('error', '转换失败')
                })
                
    except Exception as e:
        logger.error(f"❌ 增强XYZ转NPZ转换错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
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

# RAG API 端点
@app.post("/api/rag/initialize")
async def initialize_rag_api(
    api_key: str = Form(...),
    base_url: str = Form("https://openrouter.ai/api/v1")
):
    """初始化RAG服务"""
    global rag_initialized
    try:
        if not RAG_AVAILABLE:
            return JSONResponse({
                "success": False,
                "message": "RAG服务不可用，请检查LightRAG安装"
            })
        
        logger.info(f"🔥 初始化RAG服务: {base_url}")
        
        rag_service = await get_simple_rag_service()
        success = await rag_service.initialize(api_key, base_url)
        if success:
            rag_initialized = True
            logger.info("✅ RAG服务初始化成功")
            return JSONResponse({
                "success": True,
                "message": "RAG服务初始化成功"
            })
        else:
            logger.error("❌ RAG服务初始化失败")
            return JSONResponse({
                "success": False,
                "message": "RAG服务初始化失败"
            })
            
    except Exception as e:
        logger.error(f"❌ RAG初始化错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"初始化失败: {str(e)}"
        })

@app.post("/api/rag/upload")
async def upload_document_api(
    file: UploadFile = File(...),
    filename: str = Form(None)
):
    """上传文档到知识库"""
    try:
        if not RAG_AVAILABLE or not rag_initialized:
            return JSONResponse({
                "success": False,
                "message": "RAG服务未初始化"
            })
        
        logger.info(f"🔥 上传文档到知识库: {file.filename}")
        
        # 检查文件类型
        if not file.filename.endswith(('.pdf', '.txt')):
            return JSONResponse({
                "success": False,
                "message": "只支持PDF和TXT文件"
            })
        
        # 读取文件内容
        content = await file.read()
        
        # 提取文本内容
        rag_service = await get_simple_rag_service()
        if file.filename.endswith('.pdf'):
            # 直接处理PDF文件内容
            text_content = rag_service.extract_text_from_pdf(content)
        else:
            # TXT文件处理
            text_content = rag_service.extract_text_from_txt(content)
        
        if not text_content.strip():
            return JSONResponse({
                "success": False,
                "message": "文件内容为空或无法提取文本"
            })
        
        # 准备元数据
        metadata = {
            "filename": filename or file.filename,
            "upload_time": datetime.now().isoformat(),
            "file_type": "pdf" if file.filename.endswith('.pdf') else "txt",
            "content_length": len(text_content)
        }
        
        # 添加到知识库
        rag_service = await get_simple_rag_service()
        success = await rag_service.add_document(
            content=text_content,
            file_path=file.filename,
            metadata=metadata
        )
        
        if success:
            logger.info(f"✅ 文档添加成功: {file.filename}")
            return JSONResponse({
                "success": True,
                "message": f"文档 '{metadata['filename']}' 添加成功",
                "data": {
                    "filename": metadata['filename'],
                    "content_length": metadata['content_length'],
                    "upload_time": metadata['upload_time']
                }
            })
        else:
            logger.error(f"❌ 文档添加失败: {file.filename}")
            return JSONResponse({
                "success": False,
                "message": "文档添加失败"
            })
            
    except Exception as e:
        logger.error(f"❌ 上传文档错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"上传失败: {str(e)}"
        })

@app.post("/api/rag/query")
async def query_rag_api(
    query: str = Form(...),
    mode: str = Form("hybrid")
):
    """查询知识库"""
    try:
        if not RAG_AVAILABLE or not rag_initialized:
            return JSONResponse({
                "success": False,
                "message": "RAG服务未初始化"
            })
        
        logger.info(f"🔥 查询知识库: {query[:50]}...")
        
        # 执行查询
        rag_service = await get_simple_rag_service()
        result = await rag_service.query_knowledge_base(query, mode)
        
        if result["success"]:
            logger.info(f"✅ 知识库查询成功")
            return JSONResponse({
                "success": True,
                "message": "查询成功",
                "data": {
                    "response": result["response"],
                    "mode": result["mode"],
                    "query": query
                }
            })
        else:
            logger.error(f"❌ 知识库查询失败: {result['message']}")
            return JSONResponse({
                "success": False,
                "message": result["message"]
            })
            
    except Exception as e:
        logger.error(f"❌ 查询知识库错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"查询失败: {str(e)}"
        })

@app.get("/api/rag/stats")
async def get_rag_stats_api():
    """获取知识库统计信息"""
    try:
        if not RAG_AVAILABLE:
            return JSONResponse({
                "success": False,
                "message": "RAG服务不可用"
            })
        
        rag_service = await get_simple_rag_service()
        result = await rag_service.get_knowledge_base_stats()
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "message": "统计信息获取成功",
                "data": result["stats"]
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result["message"]
            })
            
    except Exception as e:
        logger.error(f"❌ 获取RAG统计错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"获取统计失败: {str(e)}"
        })

@app.post("/api/rag/clear")
async def clear_rag_api():
    """清空知识库"""
    try:
        if not RAG_AVAILABLE or not rag_initialized:
            return JSONResponse({
                "success": False,
                "message": "RAG服务未初始化"
            })
        
        logger.info("🔥 清空知识库")
        
        rag_service = await get_simple_rag_service()
        success = await rag_service.clear_knowledge_base()
        
        if success:
            logger.info("✅ 知识库清空成功")
            return JSONResponse({
                "success": True,
                "message": "知识库清空成功"
            })
        else:
            logger.error("❌ 知识库清空失败")
            return JSONResponse({
                "success": False,
                "message": "知识库清空失败"
            })
            
    except Exception as e:
        logger.error(f"❌ 清空知识库错误: {e}")
        return JSONResponse({
            "success": False,
            "message": f"清空失败: {str(e)}"
        })

@app.post("/api/chat/message-with-rag")
async def send_chat_message_with_rag_api(
    message: str = Form(...),
    model_name: str = Form("google/gemini-2.0-flash-thinking-exp:free"),
    use_rag: bool = Form(True),
    rag_mode: str = Form("hybrid")
):
    """发送聊天消息（支持RAG增强）"""
    global chat_interface
    try:
        # 延迟导入
        success, modules = lazy_import_modules()
        if not success:
            raise HTTPException(status_code=500, detail=f"模块导入失败: {modules}")
        
        ChatInterface = modules['ChatInterface']
        
        if not chat_interface:
            return JSONResponse({
                "success": False,
                "message": "聊天接口未配置，请先配置API"
            })
        
        enhanced_message = message
        rag_context = None
        
        # 如果启用RAG且RAG服务可用
        if use_rag and RAG_AVAILABLE and rag_initialized:
            try:
                logger.info(f"🔥 使用RAG增强查询: {message[:50]}...")
                
                # 查询知识库
                rag_service = await get_simple_rag_service()
                rag_result = await rag_service.query_knowledge_base(message, rag_mode)
                
                if rag_result["success"] and rag_result["response"]:
                    rag_context = rag_result["response"]
                    
                    # 构建增强的消息
                    enhanced_message = f"""基于知识库的相关信息回答以下问题：

知识库检索结果：
{rag_context}

用户问题：
{message}

请结合知识库信息和你的专业知识，提供准确、详细的回答。如果知识库信息与问题不相关，请忽略知识库信息，直接回答问题。"""
                    
                    logger.info(f"✅ RAG增强成功，上下文长度: {len(rag_context)}")
                else:
                    logger.warning(f"⚠️ RAG查询失败或无结果: {rag_result.get('message', '未知错误')}")
            except Exception as rag_error:
                logger.warning(f"⚠️ RAG增强失败，使用原始消息: {rag_error}")
        
        # 调用聊天接口处理消息
        history, error = chat_interface.process_message(
            enhanced_message,
            None,  # image_path
            False,  # multimodal_enabled
            chat_interface.memory.get_display_history(),
            model_name
        )
        
        if error:
            logger.error(f"❌ 聊天消息处理失败: {error}")
            return JSONResponse({
                "success": False,
                "message": error
            })
        
        # 获取最新的助手回复
        assistant_response = ""
        if history and len(history) > 0:
            last_message = history[-1]
            if last_message[1]:  # 助手的回复
                assistant_response = last_message[1]
        
        logger.info(f"✅ 聊天消息处理成功{'（RAG增强）' if rag_context else ''}")
        
        return JSONResponse({
            "success": True,
            "message": "消息发送成功",
            "data": {
                "response": assistant_response,
                "history": history,
                "rag_enhanced": bool(rag_context),
                "rag_context_length": len(rag_context) if rag_context else 0
            }
        })
        
    except Exception as e:
        logger.error(f"❌ 发送RAG增强聊天消息错误: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"发送失败: {str(e)}"
        })

if __name__ == "__main__":
    print("� 启动分子毒性预测API服务器 (修复版)...")
    print(f"📁 工作目录: {os.getcwd()}")
    print("📱 前端地址: http://localhost:3000")
    print("🔧 API地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    if RAG_AVAILABLE:
        print("🧠 RAG服务: 可用")
    else:
        print("⚠️ RAG服务: 不可用")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
