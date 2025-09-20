# 随机 vs Scaffold 切分性能对比图生成指南

## 📋 概述

这个工具包帮助你生成A1任务所需的"随机 vs Scaffold 切分性能对比图"，包括：

1. **性能对比柱状图** - 显示每个终点在两种切分下的性能
2. **梯形图 (Ladder Plot)** - 显示从随机到Scaffold的性能变化趋势
3. **统计摘要表** - 量化性能差异

## 🚀 快速开始

### 方法1: 使用模拟数据演示（推荐先试用）

```bash
cd /mnt/backup2/ai4s/backupunimolpy/responds-work/ToxD4C
python quick_split_comparison.py
```

这会生成演示图表，帮你了解最终效果。

### 方法2: 使用真实实验结果

如果你有两个实验的结果文件：

```bash
python compare_split_methods.py \
    --random_results "checkpoints/toxd4c_baseline_random" \
    --scaffold_results "checkpoints/toxd4c_baseline_scaffold" \
    --output_dir "split_comparison_results"
```

## 📁 实验结果文件结构

脚本期望的实验结果目录结构：

```
experiment_directory/
├── test_metrics.json      # 测试指标
├── config.json           # 实验配置
└── test_predictions.json # 测试预测结果（可选）
```

### test_metrics.json 格式示例：

```json
{
  "classification": {
    "auc": {
      "0": 0.75,
      "1": 0.68,
      ...
    },
    "pr_auc": {
      "0": 0.72,
      "1": 0.65,
      ...
    },
    "f1": {
      "0": 0.67,
      "1": 0.61,
      ...
    }
  },
  "regression": {
    "r2": {
      "0": 0.45,
      "1": 0.52,
      ...
    },
    "rmse": {
      "0": 0.34,
      "1": 0.28,
      ...
    }
  }
}
```

## 🎯 如何获取真实数据

### 1. 训练随机切分模型

```bash
python train.py \
    --experiment_name "toxd4c_baseline_random" \
    --data_dir "data/dataset" \
    --split_strategy "random" \
    --deterministic \
    --seed 42 \
    --batch_size 16 \
    --num_epochs 20
```

### 2. 训练Scaffold切分模型

```bash
python train.py \
    --experiment_name "toxd4c_baseline_scaffold" \
    --data_dir "data/processed_scaffold" \
    --split_strategy "scaffold" \
    --deterministic \
    --seed 42 \
    --batch_size 16 \
    --num_epochs 20
```

### 3. 生成对比图

```bash
python compare_split_methods.py \
    --random_results "checkpoints/toxd4c_baseline_random" \
    --scaffold_results "checkpoints/toxd4c_baseline_scaffold"
```

## 📊 生成的图表说明

### 1. 性能对比柱状图 (performance_comparison.png)

- **左上**: 分类任务AUC对比
- **右上**: 分类任务PR-AUC对比  
- **左下**: 回归任务R²对比
- **右下**: 性能相关性散点图

### 2. 梯形图 (ladder_plot.png)

- **左**: 分类任务性能变化
- **右**: 回归任务性能变化
- 绿线表示Scaffold性能更好，红线表示随机性能更好

### 3. 数据文件

- `split_comparison_data.csv`: 完整对比数据
- `split_comparison_summary.csv`: 统计摘要

## 🔧 自定义选项

### 修改终点列表

编辑脚本中的终点定义：

```python
classification_endpoints = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

regression_endpoints = [
    'IGC50', 'LC50', 'LC50DM', 'LLNA', 'LOAEL'
]
```

### 修改图表样式

在脚本中调整matplotlib参数：

```python
plt.style.use('seaborn-v0_8')  # 或其他样式
sns.set_palette("husl")        # 或其他调色板
```

## 📈 预期结果解读

### 典型观察：

1. **Scaffold切分性能通常更低** - 这是正常的，因为Scaffold切分更严格
2. **性能差异范围** - 通常AUC差异在0.05-0.15之间
3. **终点间差异** - 不同终点对切分方法的敏感性不同

### 统计意义：

- **平均性能差异** - 量化整体影响
- **标准差** - 评估稳定性
- **终点数量** - 确保统计可靠性

## 🚨 常见问题

### Q: 找不到实验结果文件？
A: 检查实验目录结构，确保有`test_metrics.json`文件

### Q: 图表显示异常？
A: 检查数据格式，确保指标值在合理范围内(0-1)

### Q: 想要更多指标？
A: 修改脚本中的`extract_metrics_by_endpoint`函数

## 📝 用于论文的建议

1. **图表标题**: "Performance Comparison: Random vs Scaffold Split"
2. **图例说明**: 详细解释每个子图的含义
3. **统计检验**: 考虑添加配对t检验结果
4. **讨论要点**: 
   - Scaffold切分的必要性
   - 性能下降的合理性
   - 模型泛化能力评估

## 🔄 与其他任务的关系

这个对比图是A1任务的核心输出，也为以下任务提供基础：

- **A2**: 不确定性量化在不同切分下的表现
- **A5**: 详细指标分析的切分对比部分
- **C1/C2**: 外部验证的内部基准

生成这些图表后，你就完成了A1任务的重要组成部分！🎉
