#!/usr/bin/env python3
"""
测试简化版后端
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def start_simple_backend():
    """启动简化版后端"""
    print("🚀 启动简化版后端...")
    
    backend_dir = Path(__file__).parent / "backend"
    
    try:
        # 启动简化版后端
        process = subprocess.Popen(
            [sys.executable, "simple_main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ 简化版后端已启动")
        print("等待服务启动...")
        time.sleep(3)
        
        # 测试健康检查
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ 后端健康检查通过")
                print(f"响应: {response.json()}")
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 无法连接到后端: {e}")
        
        return process
        
    except Exception as e:
        print(f"❌ 启动后端失败: {e}")
        return None

def test_api_endpoints():
    """测试API端点"""
    print("\n🧪 测试API端点...")
    
    # 测试健康检查
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ 健康检查: {response.json()}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return
    
    # 创建测试文件
    test_content = """2
Test molecule
C 0.0 0.0 0.0
H 1.0 0.0 0.0
"""
    
    # 测试文件转换
    try:
        files = {'file': ('test.xyz', test_content, 'text/plain')}
        response = requests.post("http://localhost:8000/api/convert/xyz-to-npz", files=files)
        print(f"✅ 文件转换: {response.json()}")
    except Exception as e:
        print(f"❌ 文件转换失败: {e}")
    
    # 测试二元预测
    try:
        files = {'file': ('test.npz', b'fake npz content', 'application/octet-stream')}
        response = requests.post("http://localhost:8000/api/predict/binary", files=files)
        print(f"✅ 二元预测: {response.json()}")
    except Exception as e:
        print(f"❌ 二元预测失败: {e}")
    
    # 测试聊天配置
    try:
        data = {'base_url': 'https://test.com', 'api_key': 'test_key'}
        response = requests.post("http://localhost:8000/api/chat/configure", data=data)
        print(f"✅ 聊天配置: {response.json()}")
    except Exception as e:
        print(f"❌ 聊天配置失败: {e}")

def main():
    """主函数"""
    print("🧪 简化版后端测试")
    print("=" * 50)
    
    # 启动后端
    backend_process = start_simple_backend()
    
    if not backend_process:
        print("❌ 无法启动后端")
        return
    
    try:
        # 测试API
        test_api_endpoints()
        
        print("\n" + "=" * 50)
        print("✅ 简化版后端测试完成")
        print("🌐 API文档: http://localhost:8000/docs")
        print("🔧 健康检查: http://localhost:8000/health")
        print("\n按 Ctrl+C 停止服务")
        
        # 保持运行
        backend_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在停止服务...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ 服务已停止")

if __name__ == "__main__":
    main()