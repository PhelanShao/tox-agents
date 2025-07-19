# ToxD4C 深度学习项目打包清单

## 项目概述
ToxD4C是一个先进的深度学习框架，用于分子毒性预测。该项目采用GNN-Transformer混合架构，结合多尺度特征融合和对比学习。

## 必要文件和目录清单

### 1. 核心源代码 (models/)
```
models/
├── __init__.py                                    # 模块初始化文件
├── toxd4c.py                                     # 主模型类
├── architectures/
│   ├── __init__.py
│   └── gnn_transformer_hybrid.py                # GNN-Transformer混合架构
├── encoders/
│   ├── __init__.py
│   ├── geometric_encoder.py                     # 几何编码器
│   ├── geometric_topological_encoder.py         # 几何拓扑编码器
│   └── hierarchical_encoder.py                  # 层次化编码器
├── fingerprints/
│   ├── __init__.py
│   └── molecular_fingerprint_enhanced.py        # 增强分子指纹模块
├── heads/
│   ├── __init__.py
│   └── multi_scale_prediction_head.py           # 多尺度预测头
└── losses/
    ├── __init__.py
    └── contrastive_loss.py                       # 对比学习损失函数
```

### 2. 配置文件 (configs/)
```
configs/
├── __init__.py
└── toxd4c_config.py                              # 主配置文件
```

### 3. 数据处理模块 (data/)
```
data/
├── __init__.py
├── lmdb_dataset.py                               # LMDB数据集处理
└── dataset/                                      # 数据集目录（保留结构，不包含实际数据）
    ├── README.md                                 # 数据集说明
    └── .gitkeep                                  # 保持目录结构
```

### 4. 训练和推理脚本
```
train.py                                          # 主训练脚本
inference_toxd4c.py                               # 推理脚本
preprocess_data.py                                # 数据预处理脚本
test_demo.py                                      # 演示测试脚本
```

### 5. 依赖和环境配置
```
requirements.txt                                  # Python依赖包列表
install_dependencies.sh                           # 依赖安装脚本
```

### 6. 文档和说明
```
README.md                                         # 项目说明文档
```

### 7. 示例和工具文件
```
molecules_to_predict.smi                          # 示例分子SMILES文件
input.xyz                                         # 示例输入文件
```

### 8. 目录结构（保留但清空内容）
```
checkpoints_real/                                 # 模型检查点目录（仅保留结构）
├── README.md                                     # 检查点说明
└── .gitkeep

training/                                         # 训练相关工具
├── __init__.py
└── rl_optimizer.py                               # 强化学习优化器

results/                                          # 结果输出目录（仅保留结构）
├── README.md
└── .gitkeep

logs/                                             # 日志目录（仅保留结构）
├── README.md
└── .gitkeep
```

## 排除的文件和目录

### 1. 缓存和临时文件
- `__pycache__/` - Python字节码缓存
- `*.pyc` - 编译的Python文件
- `*.pyo` - 优化的Python文件

### 2. 训练权重和模型文件
- `checkpoints_real/*.pth` - 训练好的模型权重
- `checkpoints_real/*.pt` - PyTorch模型文件
- `checkpoints_real/*/` - 具体实验的检查点目录内容

### 3. 数据文件
- `data/dataset/*.lmdb` - LMDB数据库文件
- `data/processed/*.lmdb` - 处理后的数据文件

### 4. 日志和结果文件
- `*.log` - 日志文件
- `training_real_data.log` - 训练日志
- `*.csv` - 结果CSV文件（除了示例文件）
- `*.json` - 结果JSON文件
- `*.png` - 图表文件
- `*.pdf` - PDF报告文件

### 5. 分析和可视化文件
- `ablation_*.py` - 消融研究脚本
- `final_visualization.py` - 最终可视化脚本
- `detailed_ablation_*.py` - 详细消融分析
- `*_analysis.py` - 各种分析脚本

## 文件大小估算
- 核心代码: ~2MB
- 配置文件: ~50KB
- 文档: ~100KB
- 脚本文件: ~500KB
- 总计: ~3MB（不包含数据和权重）

## 打包说明
1. 保持原有目录结构
2. 包含所有必要的源代码和配置
3. 提供完整的依赖安装脚本
4. 包含详细的使用说明
5. 排除所有训练权重和数据文件
6. 清理所有缓存和临时文件

这个打包清单确保了项目的完整性和可重现性，同时保持了合理的文件大小。