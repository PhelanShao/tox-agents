#!/usr/bin/env python3
"""
集成测试脚本 - 验证前后端集成是否正常工作
"""

import requests
import json
import time
import sys
from pathlib import Path

# API基础URL
API_BASE = "http://localhost:8000"

def test_api_health():
    """测试API健康状态"""
    print("🔍 测试API健康状态...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API服务正常运行")
            return True
        else:
            print(f"❌ API服务异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到API服务: {e}")
        print("请确保后端服务已启动: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        return False

def test_file_conversion():
    """测试文件转换功能"""
    print("\n🔍 测试文件转换功能...")
    
    # 创建测试XYZ文件
    test_xyz_content = """2
Test molecule
C 0.0 0.0 0.0
H 1.0 0.0 0.0
"""
    
    test_file_path = Path("test_molecule.xyz")
    test_file_path.write_text(test_xyz_content)
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_molecule.xyz', f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE}/api/convert/xyz-to-npz", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 文件转换功能正常")
                return True
            else:
                print(f"❌ 文件转换失败: {result.get('message')}")
                return False
        else:
            print(f"❌ 文件转换请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 文件转换测试异常: {e}")
        return False
    finally:
        # 清理测试文件
        if test_file_path.exists():
            test_file_path.unlink()

def test_prediction():
    """测试预测功能"""
    print("\n🔍 测试预测功能...")
    
    # 创建测试NPZ文件（简化版）
    test_xyz_content = """2
Test molecule
C 0.0 0.0 0.0
H 1.0 0.0 0.0
"""
    
    test_file_path = Path("test_molecule.xyz")
    test_file_path.write_text(test_xyz_content)
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_molecule.xyz', f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE}/api/predict/binary", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 预测功能正常")
                print(f"   预测结果: {result.get('data', {})}")
                return True
            else:
                print(f"❌ 预测失败: {result.get('message')}")
                return False
        else:
            print(f"❌ 预测请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 预测测试异常: {e}")
        return False
    finally:
        # 清理测试文件
        if test_file_path.exists():
            test_file_path.unlink()

def test_visualization():
    """测试可视化功能"""
    print("\n🔍 测试可视化功能...")
    
    # 创建测试文件
    test_xyz_content = """2
Test molecule
C 0.0 0.0 0.0
H 1.0 0.0 0.0
"""
    
    test_file_path = Path("test_molecule.xyz")
    test_file_path.write_text(test_xyz_content)
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_molecule.xyz', f, 'application/octet-stream')}
            data = {
                'frame_index': '0',
                'representation': 'sticks',
                'rotation_x': '0',
                'rotation_y': '0',
                'rotation_z': '0',
                'zoom': '1.0'
            }
            response = requests.post(f"{API_BASE}/api/visualize/molecule", files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 可视化功能正常")
                return True
            else:
                print(f"❌ 可视化失败: {result.get('message')}")
                return False
        else:
            print(f"❌ 可视化请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 可视化测试异常: {e}")
        return False
    finally:
        # 清理测试文件
        if test_file_path.exists():
            test_file_path.unlink()

def test_chat_configuration():
    """测试聊天配置功能"""
    print("\n🔍 测试聊天配置功能...")
    
    try:
        data = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': 'test_key_12345'
        }
        response = requests.post(f"{API_BASE}/api/chat/configure", data=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 聊天配置功能正常")
                return True
            else:
                print(f"❌ 聊天配置失败: {result.get('message')}")
                return False
        else:
            print(f"❌ 聊天配置请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 聊天配置测试异常: {e}")
        return False

def check_frontend():
    """检查前端是否运行"""
    print("\n🔍 检查前端服务...")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ 前端服务正常运行")
            return True
        else:
            print(f"❌ 前端服务异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到前端服务: {e}")
        print("请确保前端服务已启动: npm run dev")
        return False

def main():
    """主测试函数"""
    print("🧪 分子毒性预测平台集成测试")
    print("=" * 50)
    
    tests = [
        ("API健康检查", test_api_health),
        ("文件转换功能", test_file_conversion),
        ("预测功能", test_prediction),
        ("可视化功能", test_visualization),
        ("聊天配置功能", test_chat_configuration),
        ("前端服务检查", check_frontend),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            time.sleep(1)  # 避免请求过快
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！集成成功！")
        print("\n🚀 你可以开始使用以下服务:")
        print("   📱 前端界面: http://localhost:3000")
        print("   🔧 后端API: http://localhost:8000")
        print("   📚 API文档: http://localhost:8000/docs")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个测试失败")
        print("请检查服务是否正常启动，并查看错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)