#!/usr/bin/env python3
"""
真实UniMol预测功能测试
使用现有的正确格式NPZ文件
"""

import requests
import os
import sys
from pathlib import Path

# 设置代理绕过
os.environ['no_proxy'] = 'localhost,127.0.0.1'

def test_real_prediction():
    """测试真实的UniMol预测功能"""
    
    print("🧪 真实UniMol预测功能测试")
    print("=" * 60)
    
    # 测试API健康状态
    print("🔍 测试API健康状态...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API服务正常运行: {result['message']}")
            print(f"   工作目录: {result['working_directory']}")
            print(f"   模块已加载: {result['modules_loaded']}")
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        return
    
    # 测试模块导入
    print("\n🔍 测试模块导入...")
    try:
        response = requests.get('http://localhost:8000/test-import', timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 模块导入成功: {len(result['modules'])}个模块")
                for module in result['modules']:
                    print(f"   - {module}")
            else:
                print(f"❌ 模块导入失败: {result['message']}")
                return
        else:
            print(f"❌ 模块导入测试失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 模块导入测试异常: {e}")
        return
    
    # 使用bundle内的模型目录（需用户自行放置模型/NPZ文件）
    bundle_root = Path(__file__).resolve().parents[1]
    models_root = Path(os.environ.get('INTEGRATED_MODELS_DIR', bundle_root / 'models')).resolve()

    # 使用现有的NPZ文件进行预测测试（如果存在）
    npz_files = [
        models_root / 'ToxPred_modelmini' / '3998merged_structures_merged.npz',
        models_root / 'converted_sequential.npz'
    ]
    
    for npz_file in npz_files:
        if not os.path.exists(npz_file):
            print(f"⚠️ 跳过不存在的文件: {npz_file}")
            continue
            
        print(f"\n🔍 测试二元预测功能 - {os.path.basename(npz_file)}...")
        
        try:
            # 读取NPZ文件
            with open(npz_file, 'rb') as f:
                files = {'file': (os.path.basename(npz_file), f, 'application/octet-stream')}
                data = {
                    'model_path': str(models_root / 'ToxPred_modelmini')
                }
                
                response = requests.post(
                    'http://localhost:8000/api/predict/binary',
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"✅ 二元预测成功!")
                    data = result['data']
                    print(f"   预测结果: {'有毒' if data['prediction'] == 1 else '无毒'}")
                    print(f"   概率: {data['probability']:.3f}")
                    print(f"   置信度: {data['confidence']}")
                    print(f"   解释: {data['interpretation']}")
                    print(f"   总样本数: {data['total_predictions']}")
                    print(f"   阳性样本: {data['positive_predictions']}")
                    print(f"   阴性样本: {data['negative_predictions']}")
                    print(f"   使用模型: {data['model_used']}")
                    
                    # 如果有概率图表
                    if 'plot_path' in data:
                        print(f"   概率图表: {data['plot_path']}")
                    
                    return True  # 成功了就返回
                else:
                    print(f"❌ 二元预测失败: {result['message']}")
            else:
                print(f"❌ 二元预测请求失败: {response.status_code}")
                print(f"   响应: {response.text}")
                
        except Exception as e:
            print(f"❌ 二元预测异常: {e}")
    
    print("\n❌ 所有NPZ文件测试都失败了")
    return False

if __name__ == "__main__":
    success = test_real_prediction()
    if success:
        print("\n🎉 真实UniMol预测功能测试成功!")
    else:
        print("\n💥 真实UniMol预测功能测试失败!")
