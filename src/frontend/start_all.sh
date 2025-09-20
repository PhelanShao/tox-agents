#!/bin/bash
# 分子毒性预测平台 - 完整版本启动脚本

echo "🚀 启动分子毒性预测平台 (完整版本)..."
echo "============================================================"

# 检查并清理现有进程
echo "🔍 检查现有进程..."
pkill -f "python.*main" 2>/dev/null
pkill -f "next" 2>/dev/null
sleep 2

# 启动后端 (真实UniMol功能)
echo "🔧 启动后端服务器 (真实UniMol推理)..."
cd /mnt/backup2/ai4s/frontend/backend
python main_fixed.py &
BACKEND_PID=$!
echo "   后端PID: $BACKEND_PID"

# 等待后端启动
echo "⏳ 等待后端启动..."
sleep 8

# 检查后端状态
echo "🔍 检查后端状态..."
if curl --noproxy localhost -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 后端启动成功: http://localhost:8000"
else
    echo "❌ 后端启动失败"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# 启动前端
echo "📱 启动前端服务器..."
cd /mnt/backup2/ai4s/frontend
npm run dev &
FRONTEND_PID=$!
echo "   前端PID: $FRONTEND_PID"

# 等待前端启动
echo "⏳ 等待前端启动..."
sleep 5

# 检查前端状态
echo "🔍 检查前端状态..."
if curl --noproxy localhost -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ 前端启动成功: http://localhost:3000"
else
    echo "❌ 前端启动失败"
fi

echo ""
echo "🎉 系统启动完成!"
echo "============================================================"
echo "📱 前端界面: http://localhost:3000"
echo "🔧 后端API:  http://localhost:8000"
echo "📚 API文档:  http://localhost:8000/docs"
echo "🧪 健康检查: http://localhost:8000/health"
echo ""
echo "📁 测试文件: /mnt/backup2/ai4s/paa_TS.xyz"
echo "🔬 功能测试: python test_real_prediction.py"
echo ""
echo "⚠️  注意: 这是完整版本，包含真实的UniMol推理功能"
echo "============================================================"

# 等待用户输入停止
echo ""
read -p "按Enter键停止所有服务..."

# 停止服务
echo "🛑 正在停止所有服务..."
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null
pkill -f "python.*main" 2>/dev/null
pkill -f "next" 2>/dev/null

echo "✅ 所有服务已停止"