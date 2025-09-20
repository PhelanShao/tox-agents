# 🔧 故障排除指南

## 当前问题诊断

根据测试结果，后端服务无法启动。这通常是由于以下原因：

### 1. 依赖导入问题

主要问题是 `frontend/backend/main.py` 试图导入原项目模块失败。

**解决方案A: 使用简化版后端（推荐用于测试）**

```bash
cd frontend
python test_simple_backend.py
```

这会启动一个简化版后端，所有功能都是模拟的，但可以验证前后端通信。

**解决方案B: 修复原有模块导入**

```bash
# 1. 检查原项目依赖
cd /mnt/backup2/ai4s/backupunimolpy
python -c "from interface import convert_xyz_to_npz; print('导入成功')"

# 2. 如果导入失败，安装缺失依赖
pip install gradio pandas numpy matplotlib seaborn pillow

# 3. 检查具体缺失的模块
cd frontend/backend
python -c "import sys; sys.path.insert(0, '/mnt/backup2/ai4s/backupunimolpy'); from interface import convert_xyz_to_npz"
```

### 2. 端口占用问题

```bash
# 检查端口占用
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# 如果端口被占用，杀死进程
sudo kill -9 <PID>
```

### 3. 权限问题

```bash
# 确保有写入权限
chmod +x frontend/backend/main.py
chmod +x frontend/backend/simple_main.py
```

## 🚀 快速启动方案

### 方案1: 简化版测试（立即可用）

```bash
cd frontend
python test_simple_backend.py
```

这会启动：
- 简化版后端 (端口8000)
- 模拟所有API功能
- 可以测试前端界面

### 方案2: 手动启动简化版

```bash
# 终端1: 启动简化版后端
cd frontend/backend
python simple_main.py

# 终端2: 启动前端
cd frontend
npm run dev
```

### 方案3: 修复完整版后端

1. **检查原项目依赖**:
```bash
cd /mnt/backup2/ai4s/backupunimolpy
python toxpre.py  # 确保原项目可以运行
```

2. **修复导入路径**:
```bash
cd frontend/backend
export PYTHONPATH="/mnt/backup2/ai4s/backupunimolpy:$PYTHONPATH"
python main.py
```

3. **如果仍有问题，创建符号链接**:
```bash
cd frontend/backend
ln -s /mnt/backup2/ai4s/backupunimolpy/* .
python main.py
```

## 🧪 测试验证

### 1. 后端API测试

```bash
# 健康检查
curl http://localhost:8000/health

# 文件上传测试
echo -e "2\nTest molecule\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0" > test.xyz
curl -X POST -F "file=@test.xyz" http://localhost:8000/api/convert/xyz-to-npz
```

### 2. 前端测试

访问 http://localhost:3000 并：
- 上传测试文件
- 选择预测类型
- 查看是否有错误信息

### 3. 集成测试

```bash
cd frontend
python test_integration.py
```

## 📋 常见错误及解决方案

### 错误1: "Connection refused"
**原因**: 后端服务未启动
**解决**: 使用简化版后端或修复导入问题

### 错误2: "ModuleNotFoundError"
**原因**: Python模块导入失败
**解决**: 
```bash
pip install fastapi uvicorn python-multipart
cd /mnt/backup2/ai4s/backupunimolpy
pip install -r requirements.txt  # 如果存在
```

### 错误3: "Permission denied"
**原因**: 文件权限问题
**解决**:
```bash
chmod +x frontend/backend/*.py
chmod 755 frontend/backend/
```

### 错误4: "Port already in use"
**原因**: 端口被占用
**解决**:
```bash
# 查找占用进程
lsof -i :8000
# 杀死进程
kill -9 <PID>
```

## 🔄 开发模式

### 使用简化版进行前端开发

1. 启动简化版后端（模拟所有API）
2. 开发和测试前端功能
3. 前端完成后再集成真实后端

### 逐步集成真实功能

1. 先让简化版运行
2. 逐个替换模拟功能为真实功能
3. 测试每个功能模块

## 📞 获取帮助

### 查看详细日志

```bash
# 后端日志
cd frontend/backend
python simple_main.py 2>&1 | tee backend.log

# 前端日志
cd frontend
npm run dev 2>&1 | tee frontend.log
```

### 诊断信息收集

```bash
# 系统信息
python --version
node --version
npm --version

# 依赖检查
pip list | grep -E "(fastapi|uvicorn|gradio)"
npm list | grep -E "(next|react)"

# 端口检查
netstat -tulpn | grep -E ":(3000|8000)"
```

## 🎯 推荐解决流程

1. **立即测试**: 使用 `python test_simple_backend.py`
2. **验证前端**: 访问 http://localhost:3000
3. **测试功能**: 上传文件，测试各个功能模块
4. **逐步修复**: 如果简化版工作正常，再修复真实后端
5. **完整集成**: 最后集成所有真实功能

这样可以确保至少有一个可工作的版本，然后逐步改进。