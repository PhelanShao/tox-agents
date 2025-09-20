#!/usr/bin/env python3
"""
启动脚本 - 同时运行FastAPI后端和Next.js前端
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖...")
    
    # 检查Python依赖
    try:
        import fastapi
        import uvicorn
        print("✅ Python依赖已安装")
    except ImportError as e:
        print(f"❌ Python依赖缺失: {e}")
        print("请运行: pip install fastapi uvicorn python-multipart")
        return False
    
    # 检查Node.js依赖
    if not (Path.cwd() / "node_modules").exists():
        print("❌ Node.js依赖缺失")
        print("请运行: npm install")
        return False
    else:
        print("✅ Node.js依赖已安装")
    
    return True

def start_backend():
    """启动FastAPI后端"""
    print("🚀 启动FastAPI后端...")
    backend_dir = Path.cwd() / "backend"
    
    # 确保后端目录存在
    if not backend_dir.exists():
        print(f"❌ 后端目录不存在: {backend_dir}")
        return None
    
    # 启动FastAPI服务器
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ FastAPI后端已启动 (http://localhost:8000)")
        return process
    except Exception as e:
        print(f"❌ 启动FastAPI后端失败: {e}")
        return None

def start_frontend():
    """启动Next.js前端"""
    print("🚀 启动Next.js前端...")
    
    # 启动Next.js开发服务器
    cmd = ["npm", "run", "dev"]
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ Next.js前端已启动 (http://localhost:3000)")
        return process
    except Exception as e:
        print(f"❌ 启动Next.js前端失败: {e}")
        return None

def main():
    """主函数"""
    print("🔥 分子毒性预测平台启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    processes = []
    
    try:
        # 启动后端
        backend_process = start_backend()
        if backend_process:
            processes.append(backend_process)
            time.sleep(2)  # 等待后端启动
        
        # 启动前端
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
        
        if not processes:
            print("❌ 没有成功启动任何服务")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎉 服务启动成功!")
        print("📱 前端地址: http://localhost:3000")
        print("🔧 后端API: http://localhost:8000")
        print("📚 API文档: http://localhost:8000/docs")
        print("=" * 50)
        print("按 Ctrl+C 停止所有服务")
        
        # 等待用户中断
        while True:
            time.sleep(1)
            
            # 检查进程是否还在运行
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"⚠️  进程 {i+1} 已退出")
                    processes.remove(process)
            
            if not processes:
                print("❌ 所有进程已退出")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在停止服务...")
        
        # 终止所有进程
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ 所有服务已停止")
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        
        # 清理进程
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()