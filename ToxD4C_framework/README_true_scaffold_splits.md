# 真正的Scaffold切分实现指南

## 🎯 目标

生成真正的scaffold切分数据，确保随机vs scaffold性能对比的准确性。

## 📋 问题分析

之前发现的问题：
- ❌ `--split_strategy "scaffold"`参数被忽略
- ❌ 训练脚本直接使用预先切分的LMDB文件
- ❌ 无论设置什么切分策略，都使用相同的数据

## ✅ 解决方案

### 方案1: 生成真正的切分数据（推荐）

## 🚀 快速开始

### 步骤1: 查找原始数据

首先确认你的原始数据位置：

```bash
# 查找可能的数据位置
find . -name "*.lmdb" -type d | head -10

# 常见位置：
# - data/dataset/train.lmdb (包含所有数据)
# - data/dataset/all.lmdb
# - data/original/all.lmdb
```

### 步骤2: 生成随机切分数据

```bash
cd /mnt/backup2/ai4s/backupunimolpy/responds-work/ToxD4C

# 自动查找数据并生成随机切分
python generate_random_splits.py --output_dir "data/random_split"

# 或指定具体数据文件
python generate_random_splits.py \
    --input_data "data/dataset/train.lmdb" \
    --output_dir "data/random_split" \
    --train_size 0.8 \
    --val_size 0.1 \
    --random_state 42
```

### 步骤3: 生成Scaffold切分数据

```bash
# 自动查找数据并生成scaffold切分
python generate_scaffold_splits.py --output_dir "data/scaffold_split"

# 或指定具体数据文件
python generate_scaffold_splits.py \
    --input_data "data/dataset/train.lmdb" \
    --output_dir "data/scaffold_split" \
    --train_size 0.8 \
    --val_size 0.1 \
    --random_state 42
```

### 步骤4: 训练随机切分模型

```bash
python train.py \
    --experiment_name "toxd4c_baseline_random" \
    --data_dir "data/random_split" \
    --deterministic \
    --seed 42 \
    --batch_size 16 \
    --num_epochs 50
```

### 步骤5: 训练Scaffold切分模型

```bash
python train.py \
    --experiment_name "toxd4c_baseline_scaffold" \
    --data_dir "data/scaffold_split" \
    --deterministic \
    --seed 42 \
    --batch_size 16 \
    --num_epochs 50
```

### 步骤6: 生成性能对比图

```bash
python compare_split_methods.py \
    --random_results "checkpoints/toxd4c_baseline_random" \
    --scaffold_results "checkpoints/toxd4c_baseline_scaffold" \
    --output_dir "split_comparison_results"
```

## 📊 生成的文件结构

### 随机切分数据 (`data/random_split/`)
```
random_split/
├── train.lmdb          # 训练集
├── valid.lmdb          # 验证集
├── test.lmdb           # 测试集
└── split_info.json     # 切分信息
```

### Scaffold切分数据 (`data/scaffold_split/`)
```
scaffold_split/
├── train.lmdb              # 训练集
├── valid.lmdb              # 验证集
├── test.lmdb               # 测试集
├── split_info.json         # 切分信息
└── scaffold_analysis.json  # Scaffold分析
```

## 🔍 验证切分质量

### 检查切分信息

```bash
# 查看随机切分信息
cat data/random_split/split_info.json

# 查看scaffold切分信息
cat data/scaffold_split/split_info.json

# 查看scaffold分析
cat data/scaffold_split/scaffold_analysis.json
```

### 关键验证指标

1. **无重叠**: 确保train/val/test之间没有样本重叠
2. **Scaffold分离**: Scaffold切分中不同集合间无scaffold重叠
3. **比例正确**: 实际切分比例接近设定值

## 🎯 预期结果

### Scaffold切分特点
- ✅ **更严格**: 相同scaffold的分子不会出现在不同集合中
- ✅ **更现实**: 更接近真实药物发现场景
- ✅ **性能更低**: 通常比随机切分性能低5-15%

### 随机切分特点
- ✅ **基准性能**: 提供模型的最佳性能基准
- ✅ **完全随机**: 分子结构相似性不影响切分
- ✅ **性能更高**: 通常是最乐观的性能估计

## 🚨 常见问题

### Q: 找不到原始数据文件？
A: 检查以下位置：
```bash
ls -la data/dataset/
ls -la data/original/
ls -la data/raw/
```

### Q: 内存不足？
A: 对于大数据集，可以分批处理：
```bash
# 使用较小的map_size
export LMDB_MAP_SIZE=1073741824  # 1GB
python generate_scaffold_splits.py
```

### Q: Scaffold切分后性能差异很小？
A: 可能原因：
- 数据集中scaffold多样性不足
- 分子结构相似性较低
- 需要检查scaffold分析结果

### Q: 想要不同的切分比例？
A: 修改参数：
```bash
python generate_scaffold_splits.py \
    --train_size 0.7 \
    --val_size 0.15 \
    --output_dir "data/scaffold_split_70_15_15"
```

## 📈 性能对比分析

生成切分数据后，你将能够：

1. **量化切分影响**: 精确测量scaffold切分对性能的影响
2. **验证模型泛化**: 评估模型在结构新颖分子上的表现
3. **满足审稿要求**: 提供严格的验证切分对比

## 🔄 与其他任务的关系

这个真正的scaffold切分为以下任务提供基础：

- **A1**: 验证切分完整性 ✅
- **A5**: 详细指标分析的切分对比
- **B4**: 不同切分下的解释性分析
- **C1/C2**: 外部验证的内部基准

## 🎉 完成标志

当你看到以下输出时，说明成功：

```
🎉 Success! Scaffold split data generated.
📁 Output directory: data/scaffold_split
📋 Generated files:
  - train.lmdb
  - valid.lmdb
  - test.lmdb
  - split_info.json
  - scaffold_analysis.json

✅ Perfect scaffold separation achieved!
```

现在你就有了真正的scaffold切分数据，可以进行准确的性能对比了！🚀
