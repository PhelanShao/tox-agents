# 🚀 安装指南

## 快速安装

### 1. 清理并重新安装依赖

```bash
# 删除现有的 node_modules 和 package-lock.json
rm -rf node_modules package-lock.json

# 使用 legacy peer deps 安装
npm install --legacy-peer-deps
```

### 2. 启动开发服务器

```bash
npm run dev
```

### 3. 如果仍有问题，使用强制安装

```bash
npm install --force
```

## 故障排除

### 依赖冲突问题
如果遇到 ERESOLVE 错误，请使用以下命令：

```bash
npm install --legacy-peer-deps --force
```

### TypeScript 错误
TypeScript 错误是正常的，因为依赖还未安装。安装完成后错误会消失。

### 端口冲突
如果端口 3000 被占用，可以指定其他端口：

```bash
npm run dev -- -p 3001
```

## 验证安装

安装成功后，你应该能看到：
- 现代化的玻璃态设计界面
- 响应式布局
- 深色/浅色主题切换
- 文件上传功能
- 动画效果

## 浏览器支持

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 下一步

1. 配置后端 API 连接
2. 集成 3D 分子可视化库
3. 连接 AI 模型接口
4. 自定义主题和样式