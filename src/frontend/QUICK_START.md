# 🚀 分子毒性预测平台 - 快速启动指南

## 📋 问题修复说明

根据用户反馈，我们已经修复了以下关键问题：

### ✅ 已修复的问题

1. **后端推理问题**: 
   - 修复了FastAPI后端无法正确调用原有推理逻辑的问题
   - 现在后端会正确调用 `process_binary_prediction` 和 `process_property_prediction`
   - 保持了原有的模型加载和推理流程

2. **分子可视化问题**:
   - 集成了Molstar作为主要的3D分子可视化引擎
   - 提供了后端PyMOL渲染作为备选方案
   - 支持多种分子文件格式 (.xyz, .npz, .sdf, .mol, .pdb)

3. **API集成问题**:
   - 完善了前后端API通信
   - 添加了详细的错误处理和日志记录
   - 保持了原有功能的完整性

## 🛠️ 环境要求

### Python环境
```bash
# 确保已安装原项目依赖
cd /mnt/backup2/ai4s/backupunimolpy
pip install -r requirements.txt  # 如果有的话

# 安装FastAPI相关依赖
pip install fastapi uvicorn python-multipart
```

### Node.js环境
```bash
cd frontend
npm install
```

## 🚀 启动步骤

### 方式一：使用启动脚本（推荐）

```bash
cd frontend
python start.py
```

这个脚本会：
- 自动检查依赖
- 同时启动后端API (端口8000) 和前端 (端口3000)
- 提供实时状态监控

### 方式二：手动启动

```bash
# 终端1：启动后端API
cd frontend/backend
python main.py

# 终端2：启动前端
cd frontend
npm run dev
```

## 🧪 测试验证

### 1. 运行集成测试
```bash
cd frontend
python test_integration.py
```

### 2. 手动测试步骤

1. **访问前端**: http://localhost:3000
2. **上传测试文件**: 使用 `.xyz` 或 `.npz` 格式的分子文件
3. **选择预测类型**: 二元分类或属性预测
4. **查看结果**: 确认预测结果正确显示
5. **测试可视化**: 确认分子3D结构正确渲染
6. **测试聊天**: 配置API密钥后测试AI分析功能

## 📊 功能验证清单

### ✅ 后端推理验证
- [ ] 文件上传成功
- [ ] XYZ转NPZ转换正常
- [ ] 二元分类预测显示进度和结果
- [ ] 属性预测返回详细数据
- [ ] 后端日志显示UniMol模型加载信息

### ✅ 前端可视化验证
- [ ] Molstar 3D渲染正常工作
- [ ] 后端PyMOL渲染作为备选
- [ ] 分子结构交互操作流畅
- [ ] 多帧动画播放正常
- [ ] 导出功能正常

### ✅ 集成功能验证
- [ ] API通信正常
- [ ] 错误处理友好
- [ ] 实时状态更新
- [ ] 文件下载功能
- [ ] 聊天AI分析功能

## 🔧 故障排除

### 后端推理问题

如果看到 "正在进行二元分类预测..." 一直显示：

1. **检查后端日志**:
```bash
# 查看FastAPI日志
curl http://localhost:8000/health
```

2. **检查模型路径**:
```bash
ls -la /mnt/backup2/ai4s/backupunimolpy/ToxPred_modelmini
ls -la /mnt/backup2/ai4s/backupunimolpy/MD_model
```

3. **检查原有功能**:
```bash
cd /mnt/backup2/ai4s/backupunimolpy
python toxpre.py  # 测试原有Gradio界面
```

### 分子可视化问题

如果Molstar无法加载：

1. **检查网络连接**: 确保能访问CDN
2. **查看浏览器控制台**: 检查JavaScript错误
3. **使用后端渲染**: 系统会自动回退到PyMOL渲染

### API连接问题

1. **检查端口占用**:
```bash
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000
```

2. **检查CORS设置**: 确保前后端域名匹配

## 📝 开发调试

### 查看详细日志

```bash
# 后端日志
tail -f /tmp/fastapi.log

# 前端开发服务器日志
npm run dev -- --verbose
```

### API测试

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试文件上传（需要实际文件）
curl -X POST -F "file=@test.xyz" http://localhost:8000/api/convert/xyz-to-npz
```

## 🎯 性能优化建议

1. **模型预加载**: 首次预测会较慢，后续会快很多
2. **文件大小**: 建议单个文件不超过10MB
3. **浏览器缓存**: 清除缓存可解决部分显示问题
4. **内存使用**: 大分子文件可能需要更多内存

## 📞 技术支持

如果遇到问题：

1. **查看日志**: 检查浏览器控制台和后端日志
2. **运行测试**: 使用 `test_integration.py` 诊断问题
3. **重启服务**: 有时重启可以解决临时问题
4. **检查依赖**: 确保所有Python和Node.js依赖都已安装

## 🔄 更新说明

### v1.1.0 (当前版本)
- ✅ 修复后端推理调用问题
- ✅ 集成Molstar 3D可视化
- ✅ 完善API错误处理
- ✅ 添加详细日志记录
- ✅ 保持原有功能完整性

### 下一步计划
- 🔄 优化Molstar集成
- 🔄 添加更多分子文件格式支持
- 🔄 改进用户界面响应速度
- 🔄 添加批量处理功能

---

**重要提醒**: 这个现代化前端完全保持了原有项目的功能逻辑，只是提供了更好的用户界面和用户体验。所有的核心算法、模型和数据处理流程都保持不变。