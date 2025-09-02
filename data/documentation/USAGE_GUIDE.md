# EPA ToxCast/tcpl合规毒性标签系统 - 使用指南

## 📁 文件结构

### `/code/` - 核心代码
- `tcpl_system_final.py` - 完整的tcpl合规系统（主要代码）
- `data_verification.py` - 数据验证系统
- `mapping_system.py` - 化学品映射系统

### `/data/` - 数据文件
- `original/` - 原始数据
  - `processed_final8k213_original.csv` - 原始21st数据集（7,330个化学品）
- `processed/` - 处理后的中间数据
  - `tcpl_chemical_scores_final.csv` - 化学品tcpl分数详表
  - `cas_pubchem_mapping.json` - CAS到PUBCHEM_CID映射
  - `chemical_bridge_table.csv` - chid到CAS映射桥表
- `final/` - 最终结果
  - `processed_final8k213_tcpl_labeled_final.csv` - **主要输出**：带tcpl标签的最终数据集
  - `dataset_summary.json` - 数据集统计摘要

### `/validation/` - 验证文档
- `external_validation_results.csv` - 外部验证结果（ToxRefDB）
- `data_verification.json` - 数据来源验证报告
- `data_integrity_sha256.json` - 文件完整性校验
- `reproducibility_report.json` - 可追溯性报告（种子、参数等）

### `/reports/` - 分析报告
- `compliance_assessment.md` - tcpl合规性评估报告
- `system_analysis.md` - 系统分析报告

## 🎯 主要成果

### 数据覆盖率
- **总记录数**: 7,330个化学品
- **tcpl标签覆盖**: 6,216个（84.8%）
- **外部验证**: 1,369个化学品与ToxRefDB对齐

### 标签质量
- **数据源**: EPA invitrodb v4.2 Summary Files
- **命中判定**: SC2的hitc∈{-1,0,1}，排除hitc=-1
- **质量控制**: mc6_flags伪影过滤 + cytotox burst过滤（21,258条）
- **统计方法**: Jeffreys Beta-Binomial收缩 + 机制等权聚合

### 外部验证结果
- **Spearman相关性**: r=0.072, p=0.008（显著）
- **ROC-AUC**: 0.544 [95%CI: 0.513-0.571] (τ=10 mg/kg/day)
- **阈值选择**: 5×3嵌套交叉验证

## 🔧 使用方法

### 1. 直接使用最终数据
```python
import pandas as pd

# 加载带tcpl标签的最终数据集
data = pd.read_csv('data/final/processed_final8k213_tcpl_labeled_final.csv')

# 查看tcpl标签列
tcpl_columns = [col for col in data.columns if 'tcpl' in col.lower()]
print("可用的tcpl标签列:", tcpl_columns)

# 筛选有tcpl标签的化学品
labeled_data = data[data['tcpl_binary_compliant'] != -1]
print(f"有tcpl标签的化学品: {len(labeled_data)}个")
```

### 2. 重新运行完整系统
```bash
cd code/
python tcpl_system_final.py
```

### 3. 验证数据完整性
```bash
cd code/
python data_verification.py
```

## 📊 标签说明

### 二分类标签 (`tcpl_binary_compliant`)
- `0`: 低毒性
- `1`: 高毒性
- `-1`: 无标签

### 三分类标签 (`tcpl_ternary_compliant`)
- `0`: 低毒性
- `1`: 中等毒性
- `2`: 高毒性
- `-1`: 无标签

### 分数列
- `S_c_tcpl_compliant`: tcpl分数（0-1范围）
- `S_c_ci_lower_compliant`, `S_c_ci_upper_compliant`: 95%置信区间
- `tcpl_n_tested_compliant`: 测试的端点数
- `tcpl_n_positive_compliant`: 阳性端点数

## ✅ 质量保证

1. **完全tcpl合规**: 严格按照EPA ToxCast/tcpl标准实现
2. **数据完整性**: 所有源文件SHA256校验
3. **可重现性**: 固定随机种子，完整参数记录
4. **外部验证**: ToxRefDB独立验证，统计显著
5. **方法透明**: 完整的代码和文档

## 📞 技术支持

如有问题，请参考：
- `/validation/reproducibility_report.json` - 完整的技术参数
- `/reports/compliance_assessment.md` - 合规性详细评估
- 代码注释和文档字符串

---
生成时间: 2025-09-02 09:10:27
