# 🧬 Molecular Toxicity Predictor - Modern Frontend

一个使用最前沿技术构建的分子毒性预测平台前端界面，具有现代化的设计和强大的功能。

## ✨ 技术特色

### 🚀 前沿技术栈
- **Next.js 15** - 最新的React框架，支持App Router
- **React 19** - 最新的React版本
- **TypeScript** - 类型安全的开发体验
- **Tailwind CSS 4** - 现代化的CSS框架
- **Framer Motion** - 流畅的动画效果
- **TanStack Query** - 强大的数据获取和缓存

### 🎨 现代化设计
- **玻璃态效果 (Glassmorphism)** - 半透明背景和毛玻璃效果
- **渐变色彩系统** - 科技感的配色方案
- **微交互动画** - 流畅的用户体验
- **响应式设计** - 完美适配各种设备
- **深色模式** - 支持系统主题自动切换

### 🔧 核心功能
- **文件上传** - 支持拖拽上传分子数据文件
- **3D分子可视化** - 交互式分子结构展示
- **AI毒性预测** - 实时毒性评估和分析
- **智能聊天** - AI驱动的分析对话
- **数据导出** - 多格式结果导出

## 📁 项目结构

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── globals.css        # 全局样式
│   │   ├── layout.tsx         # 根布局
│   │   └── page.tsx           # 主页面
│   ├── components/            # React组件
│   │   ├── chat/              # 聊天界面组件
│   │   ├── layout/            # 布局组件
│   │   ├── prediction/        # 预测结果组件
│   │   ├── providers/         # Context提供者
│   │   ├── ui/                # 通用UI组件
│   │   └── visualization/     # 可视化组件
│   ├── lib/                   # 工具函数
│   │   └── utils.ts           # 通用工具
│   └── types/                 # TypeScript类型定义
├── public/                    # 静态资源
├── package.json              # 项目依赖
├── tailwind.config.js        # Tailwind配置
├── tsconfig.json             # TypeScript配置
└── next.config.js            # Next.js配置
```

## 🎯 设计亮点

### 1. 玻璃态设计系统
```css
.glass-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
```

### 2. 科技感配色
- **主色调**: 蓝色渐变 (#0ea5e9 → #0284c7)
- **强调色**: 紫色渐变 (#d946ef → #c026d3)
- **成功色**: 绿色系 (#22c55e)
- **警告色**: 橙色系 (#f59e0b)
- **危险色**: 红色系 (#ef4444)

### 3. 动画效果
- **页面加载动画**: 淡入和滑动效果
- **悬停动画**: 卡片提升和阴影变化
- **加载状态**: 旋转和脉冲动画
- **渐变动画**: 背景色彩流动

### 4. 响应式布局
- **移动优先**: 从小屏幕开始设计
- **断点系统**: xs, sm, md, lg, xl, 2xl
- **弹性网格**: CSS Grid + Flexbox
- **自适应组件**: 根据屏幕尺寸调整

## 🚀 快速开始

### 安装依赖
```bash
npm install
```

### 开发模式
```bash
npm run dev
```

### 构建生产版本
```bash
npm run build
```

### 启动生产服务器
```bash
npm start
```

## 🔧 配置说明

### Tailwind CSS 配置
- 自定义颜色系统
- 玻璃态效果工具类
- 动画关键帧
- 响应式断点

### TypeScript 配置
- 严格模式启用
- 路径别名配置
- Next.js 插件集成

### Next.js 配置
- App Router 启用
- 类型化路由
- API 代理配置
- 性能优化

## 🎨 组件库

### UI 组件
- **FileUpload**: 文件上传组件，支持拖拽
- **Button**: 多种样式的按钮组件
- **Card**: 玻璃态卡片组件
- **Modal**: 模态对话框组件

### 功能组件
- **Header**: 导航头部组件
- **PredictionResults**: 预测结果展示
- **MolecularVisualization**: 3D分子可视化
- **ChatInterface**: AI聊天界面

### 提供者组件
- **ThemeProvider**: 主题切换提供者
- **QueryProvider**: 数据查询提供者

## 🌟 特色功能

### 1. 智能文件上传
- 支持 XYZ 分子文件格式 (.xyz)
- 拖拽上传体验
- 实时上传进度
- 文件验证和错误处理

### 2. 3D分子可视化
- 交互式3D渲染
- 多种显示模式
- 属性映射可视化
- 导出高质量图像

### 3. AI驱动分析
- 实时毒性预测
- 置信度评估
- 分子属性分析
- 智能对话解释

### 4. 现代化界面
- 流畅的动画效果
- 直观的用户体验
- 无障碍访问支持
- 多设备适配

## 🔮 未来规划

- [ ] 集成更多3D可视化库
- [ ] 添加批量处理功能
- [ ] 实现实时协作功能
- [ ] 增加更多AI模型
- [ ] 优化文件格式处理
- [ ] 添加数据分析仪表板

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Built with ❤️ using cutting-edge web technologies**