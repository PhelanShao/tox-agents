# 分子毒性预测平台启动指南

## 🚀 完整版本启动步骤

### 1. 启动后端服务器 (真实UniMol功能)

```bash
# 进入后端目录
cd /mnt/backup2/ai4s/frontend/backend

# 启动修复版后端 (包含真实UniMol推理)
python main_fixed.py
```

**后端启动成功标志:**
- 看到: `🚀 启动分子毒性预测API服务器 (修复版)...`
- 看到: `📁 工作目录: /mnt/backup2/ai4s/backupunimolpy`
- 看到: `🔧 API地址: http://localhost:8000`
- 看到: `INFO:__main__:✅ 成功导入原有功能模块`

### 2. 启动前端服务器

**在新的终端窗口中:**

```bash
# 进入前端目录
cd /mnt/backup2/ai4s/frontend

# 启动Next.js前端
npm run dev
```

**前端启动成功标志:**
- 看到: `▲ Next.js 14.2.29`
- 看到: `- Local: http://localhost:3000`
- 看到: `✓ Ready in X.Xs`

### 3. 验证系统运行

**测试API连接:**
```bash
# 在第三个终端窗口中
cd /mnt/backup2/ai4s/frontend

# 运行真实功能测试
python test_real_prediction.py
```

**访问前端界面:**
- 打开浏览器访问: http://localhost:3000
- 应该看到现代化的分子毒性预测界面

### 4. 使用真实分子数据测试

**上传测试文件:**
- 使用提供的测试文件: `/mnt/backup2/ai4s/paa_TS.xyz`
- 或者任何其他XYZ格式的分子文件

**完整测试流程:**
1. 上传XYZ文件 → 自动转换为NPZ格式
2. 进行二元毒性预测 → 调用真实UniMol模型
3. 查看3D分子可视化 → PyMOL渲染
4. 进行属性预测 → 多维分子属性分析
5. AI聊天分析 → 智能分子分析报告

## 🔧 故障排除

### 后端启动问题

**如果端口被占用:**
```bash
# 杀死占用端口的进程
pkill -f "python.*main"
# 等待2秒后重新启动
sleep 2
python main_fixed.py
```

**如果模块导入失败:**
```bash
# 确保在正确目录
cd /mnt/backup2/ai4s/frontend/backend
# 检查Python路径
python -c "import sys; print(sys.path)"
```

### 前端启动问题

**如果npm依赖问题:**
```bash
cd /mnt/backup2/ai4s/frontend
npm install
npm run dev
```

**如果端口3000被占用:**
```bash
# 杀死占用端口的进程
pkill -f "next"
# 或者使用不同端口
npm run dev -- -p 3001
```

### 代理问题解决

**如果API请求失败:**
```bash
# 设置环境变量绕过代理
export no_proxy="localhost,127.0.0.1"
# 或者在测试时使用--noproxy参数
curl --noproxy localhost http://localhost:8000/health
```

## 📊 系统状态检查

### 检查后端状态
```bash
# 健康检查
curl --noproxy localhost http://localhost:8000/health

# 模块导入检查
curl --noproxy localhost http://localhost:8000/test-import
```

### 检查前端状态
```bash
# 访问前端
curl --noproxy localhost http://localhost:3000
```

### 检查进程状态
```bash
# 查看运行的Python进程
ps aux | grep python

# 查看端口占用
netstat -tlnp | grep -E "(3000|8000)"
```

## 🎯 功能验证清单

- [ ] 后端API健康检查通过
- [ ] 前端界面正常加载
- [ ] XYZ文件上传和转换成功
- [ ] 真实UniMol二元预测运行
- [ ] PyMOL 3D分子可视化显示
- [ ] 属性预测功能工作
- [ ] AI聊天分析配置成功

## 🔥 真实UniMol验证

**确认真实推理运行的标志:**
- 后端日志显示: `Loading pretrained weights from mol_pre_all_h_220816.pt`
- 看到: `start predict NNModel:unimolv1`
- 看到: `load model success!`
- 看到进度条: `val: X%|████| X/1000`

这些日志确认系统正在运行真实的UniMol神经网络推理，而不是模拟版本。

## 📱 访问地址

- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## ⚡ 快速启动脚本

创建一个快速启动脚本:

```bash
#!/bin/bash
# 保存为 start_all.sh

echo "🚀 启动分子毒性预测平台..."

# 启动后端
cd /mnt/backup2/ai4s/frontend/backend
python main_fixed.py &
BACKEND_PID=$!

# 等待后端启动
sleep 5

# 启动前端
cd /mnt/backup2/ai4s/frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ 系统启动完成!"
echo "📱 前端: http://localhost:3000"
echo "🔧 后端: http://localhost:8000"
echo "📚 API文档: http://localhost:8000/docs"

# 等待用户输入停止
read -p "按Enter键停止所有服务..."

# 停止服务
kill $BACKEND_PID $FRONTEND_PID
echo "🛑 所有服务已停止"
```

使用方法:
```bash
chmod +x start_all.sh
./start_all.sh