# 分子毒性预测平台 - 现代化前端集成

## 概述

这是一个现代化的前端界面，完全集成了原有Gradio项目的核心功能。使用Next.js 14、React 18、TypeScript和Tailwind CSS构建，提供了美观的用户界面和流畅的用户体验。

## 🎯 功能特性

### ✅ 已实现的核心功能
- **文件上传与转换**: 支持 `.xyz` 格式
- **3D分子可视化**: 
  - 实时渲染分子结构
  - 多种表示方式（棒状、球状、卡通、表面、线框）
  - 交互式旋转和缩放控制
  - 多帧动画支持
- **毒性预测**:
  - 二元分类预测（有毒/无毒）
  - 属性预测（详细毒性属性分析）
  - 概率分布图表
  - 结果导出功能
- **AI聊天分析**:
  - 配置API密钥支持
  - 上下文感知对话
  - 预测结果解释
  - 分子结构分析

### 🔧 技术架构
- **前端**: Next.js 14 + React 18 + TypeScript
- **样式**: Tailwind CSS 4 + 自定义设计系统
- **动画**: Framer Motion
- **状态管理**: React Hooks + TanStack Query
- **后端桥接**: FastAPI (Python)
- **原有功能**: 完全保留 `backupunimolpy/` 的所有逻辑

## 🚀 快速开始

### 1. 安装依赖

```bash
# 进入前端目录
cd frontend

# 安装Node.js依赖
npm install

# 安装Python依赖（用于后端桥接）
pip install fastapi uvicorn python-multipart
```

### 2. 启动服务

#### 方式一：使用启动脚本（推荐）
```bash
python start.py
```

#### 方式二：手动启动
```bash
# 终端1：启动FastAPI后端
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 终端2：启动Next.js前端
npm run dev
```

### 3. 访问应用
- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx         # 根布局
│   │   ├── page.tsx           # 主页面
│   │   └── globals.css        # 全局样式
│   ├── components/            # React组件
│   │   ├── ui/               # 基础UI组件
│   │   │   └── file-upload.tsx
│   │   ├── layout/           # 布局组件
│   │   │   └── header.tsx
│   │   ├── visualization/    # 可视化组件
│   │   │   └── molecular-visualization.tsx
│   │   ├── prediction/       # 预测组件
│   │   │   └── prediction-results.tsx
│   │   ├── chat/            # 聊天组件
│   │   │   └── chat-interface.tsx
│   │   └── providers/       # Context提供者
│   └── lib/                 # 工具库
│       ├── api.ts          # API客户端
│       └── utils.ts        # 工具函数
├── backend/                # FastAPI后端桥接
│   └── main.py            # 主要API路由
├── public/                # 静态资源
├── start.py              # 启动脚本
└── package.json          # 依赖配置
```

## 🔗 API集成

### FastAPI后端桥接
`backend/main.py` 文件作为桥接层，将原有的Python功能暴露为REST API：

- **文件转换**: `/api/convert/xyz-to-npz`, `/api/convert/npz-to-xyz`
- **预测功能**: `/api/predict/binary`, `/api/predict/property`
- **可视化**: `/api/visualize/molecule`
- **聊天功能**: `/api/chat/configure`, `/api/chat/message`
- **导出功能**: `/api/export/frame`

### 原有功能保留
所有原有的 `backupunimolpy/` 功能都通过API调用保持不变：
- `interface.py` 中的核心处理逻辑
- `toxpre.py` 中的集成接口
- 模型文件和配置保持原样

## 🎨 设计特性

### 现代化UI设计
- **Glassmorphism**: 毛玻璃效果设计
- **渐变色彩**: 科学主题配色方案
- **微交互**: 流畅的动画效果
- **响应式**: 适配各种屏幕尺寸
- **暗色模式**: 完整的暗色主题支持

### 用户体验优化
- **实时反馈**: 加载状态和进度指示
- **错误处理**: 友好的错误提示
- **快捷操作**: 键盘快捷键支持
- **智能建议**: 上下文相关的操作建议

## 🔧 配置说明

### 环境变量
创建 `.env.local` 文件：
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### API配置
在聊天界面中配置：
- **Base URL**: API服务地址
- **API Key**: 你的API密钥
- **Model**: 使用的AI模型

## 🚀 部署

### 开发环境
```bash
npm run dev
```

### 生产构建
```bash
npm run build
npm start
```

### Docker部署
```bash
# 构建镜像
docker build -t molecular-toxicity-frontend .

# 运行容器
docker run -p 3000:3000 molecular-toxicity-frontend
```

## 🤝 与原项目的关系

### 功能传承
- ✅ 保持所有原有的预测逻辑
- ✅ 保持模型文件和配置不变
- ✅ 保持API接口兼容性
- ✅ 保持数据处理流程

### 创新改进
- 🎨 现代化的用户界面设计
- ⚡ 更快的响应速度和性能
- 📱 更好的移动端适配
- 🔧 更灵活的配置选项
- 📊 更丰富的数据可视化

## 📝 使用指南

### 1. 上传分子文件
- 支持拖拽上传
- 自动格式验证
- 实时上传进度

### 2. 选择预测类型
- **二元分类**: 快速判断有毒/无毒
- **属性预测**: 详细的毒性属性分析

### 3. 查看可视化
- 3D分子结构渲染
- 多种表示方式切换
- 交互式控制面板

### 4. AI分析对话
- 配置API密钥
- 智能结果解释
- 上下文感知问答

## 🐛 故障排除

### 常见问题
1. **端口冲突**: 确保3000和8000端口未被占用
2. **依赖问题**: 运行 `npm install` 重新安装依赖
3. **API连接**: 检查后端服务是否正常启动
4. **文件上传**: 确保文件格式和大小符合要求

### 日志查看
- 前端日志: 浏览器开发者工具
- 后端日志: 终端输出
- API日志: http://localhost:8000/docs

## 📞 支持

如有问题或建议，请：
1. 检查本文档的故障排除部分
2. 查看浏览器控制台错误信息
3. 检查API服务状态
4. 提交Issue或联系开发团队

---

**注意**: 这个现代化前端完全保持了原有项目的功能逻辑，只是在用户界面和用户体验方面进行了创新和改进。所有的核心算法、模型和数据处理流程都保持不变。