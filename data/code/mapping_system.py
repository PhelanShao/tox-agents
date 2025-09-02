#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chemical Bridge Table System
化学品桥表系统

构建完整的 chid → (casn, dsstox_substance_id) → PUBCHEM_CID 映射系统
并基于MC5-6数据重新计算tcpl分数
"""

import pandas as pd
import numpy as np
import re
from scipy import stats
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

class ChemicalBridgeSystem:
    """化学品桥表系统"""
    
    def __init__(self):
        self.bridge_table = None
        self.cas_pubchem_mapping = None
        self.mc56_data = None
        self.chemical_scores = None
        self.pubchem_scores = None
        
    def normalize_cas(self, cas_series):
        """标准化CAS号格式"""
        print("🔧 标准化CAS号格式")
        
        def clean_cas(cas):
            if pd.isna(cas):
                return None
            # 转换为字符串并清理
            cas_str = str(cas).strip().upper()
            # 移除空格和特殊字符，保留数字和连字符
            cas_clean = re.sub(r'[^\d\-]', '', cas_str)
            # 确保格式为 XXXXX-XX-X
            if re.match(r'^\d{2,7}-\d{2}-\d$', cas_clean):
                return cas_clean
            return None
        
        normalized = cas_series.apply(clean_cas)
        valid_count = normalized.notna().sum()
        print(f"  CAS标准化: {valid_count:,}/{len(cas_series):,} 有效")
        
        return normalized
    
    def load_sc2_data(self):
        """加载SC1-SC2数据构建基础桥表"""
        print("\n📊 加载SC1-SC2数据构建基础桥表")
        print("-" * 50)
        
        # 加载SC1-SC2数据
        sc2_file = 'data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/sc1_sc2_invitrodb_v4_2_SEPT2024.xlsx'
        
        print("加载SC1-SC2数据...")
        sc2_data = pd.read_excel(sc2_file)
        print(f"SC2原始数据: {len(sc2_data):,} 记录")
        
        # 提取化学品标识符
        chem_ids = sc2_data[['chid', 'casn', 'dsstox_substance_id']].drop_duplicates()
        
        # 过滤有效数据
        valid_chem_ids = chem_ids.dropna(subset=['chid'])
        print(f"有效化学品记录: {len(valid_chem_ids):,}")
        
        # 标准化CAS号
        valid_chem_ids['casn_normalized'] = self.normalize_cas(valid_chem_ids['casn'])
        
        # 统计
        print(f"唯一chid: {valid_chem_ids['chid'].nunique():,}")
        print(f"有效CAS号: {valid_chem_ids['casn_normalized'].notna().sum():,}")
        print(f"有效DTXSID: {valid_chem_ids['dsstox_substance_id'].notna().sum():,}")
        
        self.bridge_table = valid_chem_ids
        return valid_chem_ids
    
    def load_cas_pubchem_mappings(self):
        """加载和合并所有CAS→PUBCHEM_CID映射"""
        print("\n🔗 加载和合并CAS→PUBCHEM_CID映射")
        print("-" * 50)
        
        mappings = []
        
        # 1. 从tox21_toxrefdb_matched_via_cas.csv加载
        print("1. 加载tox21_toxrefdb数据...")
        tox21_data = pd.read_csv('tox21_toxrefdb_matched_via_cas.csv')
        
        tox21_mapping = tox21_data[['CAS_NORM', 'PUBCHEM_CID']].copy()
        tox21_mapping['casn_normalized'] = self.normalize_cas(tox21_mapping['CAS_NORM'])
        tox21_mapping = tox21_mapping.dropna(subset=['casn_normalized', 'PUBCHEM_CID'])
        tox21_mapping['source'] = 'tox21_toxrefdb'
        
        print(f"  Tox21映射: {len(tox21_mapping):,} 记录, {tox21_mapping['PUBCHEM_CID'].nunique():,} 唯一CID")
        mappings.append(tox21_mapping[['casn_normalized', 'PUBCHEM_CID', 'source']])
        
        # 2. 可以在这里添加其他CAS→PUBCHEM_CID映射源
        # 例如：ChEMBL, DrugBank等数据库的映射
        
        # 合并所有映射
        if mappings:
            all_mappings = pd.concat(mappings, ignore_index=True)
            
            # 去重处理：同一CAS对应多个PUBCHEM_CID时的处理规则
            print("\n处理CAS→PUBCHEM_CID的多对多映射...")
            
            # 统计每个CAS对应的PUBCHEM_CID数量
            cas_counts = all_mappings.groupby('casn_normalized')['PUBCHEM_CID'].nunique()
            multi_mapping = cas_counts[cas_counts > 1]
            
            print(f"  一对一映射: {(cas_counts == 1).sum():,} 个CAS")
            print(f"  一对多映射: {len(multi_mapping):,} 个CAS")
            
            # 对于一对多的情况，选择最小的PUBCHEM_CID（通常是parent compound）
            final_mapping = all_mappings.groupby('casn_normalized').agg({
                'PUBCHEM_CID': 'min',  # 选择最小的CID
                'source': 'first'
            }).reset_index()
            
            print(f"  最终映射: {len(final_mapping):,} 个CAS→PUBCHEM_CID")
            
            self.cas_pubchem_mapping = final_mapping
            return final_mapping
        else:
            print("❌ 没有找到CAS→PUBCHEM_CID映射数据")
            return pd.DataFrame()
    
    def create_complete_bridge_table(self):
        """创建完整的桥表"""
        print("\n🌉 创建完整的chid→PUBCHEM_CID桥表")
        print("-" * 50)
        
        if self.bridge_table is None or self.cas_pubchem_mapping is None:
            print("❌ 缺少基础数据")
            return None
        
        # 通过CAS号连接
        bridge_with_pubchem = self.bridge_table.merge(
            self.cas_pubchem_mapping[['casn_normalized', 'PUBCHEM_CID']], 
            on='casn_normalized', 
            how='left'
        )
        
        # 统计映射结果
        total_chids = len(bridge_with_pubchem)
        mapped_chids = bridge_with_pubchem['PUBCHEM_CID'].notna().sum()
        
        print(f"桥表统计:")
        print(f"  总chid数: {total_chids:,}")
        print(f"  映射到PUBCHEM_CID: {mapped_chids:,} ({mapped_chids/total_chids*100:.1f}%)")
        print(f"  唯一PUBCHEM_CID: {bridge_with_pubchem['PUBCHEM_CID'].nunique():,}")
        
        # 保存桥表
        bridge_with_pubchem.to_csv('output/data/chemical_bridge_table.csv', index=False)
        print(f"✅ 桥表已保存: output/data/chemical_bridge_table.csv")
        
        self.bridge_table = bridge_with_pubchem
        return bridge_with_pubchem
    
    def load_mc56_data(self):
        """加载MC5-6数据"""
        print("\n📊 加载MC5-6数据")
        print("-" * 50)
        
        mc56_file = 'data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/mc5-6_winning_model_fits-flags_invitrodbv4_2_SEPT2024.csv'
        
        # 分块加载以节省内存
        print("分块加载MC5-6数据...")
        chunks = []
        chunk_size = 50000
        
        for chunk in pd.read_csv(mc56_file, chunksize=chunk_size):
            # 只保留必要的列
            essential_cols = ['chid', 'aeid', 'hitc', 'mc6_flags', 'aenm']
            chunk_filtered = chunk[essential_cols].copy()
            
            # 过滤有效的hitc值
            valid_hitc = chunk_filtered['hitc'].isin([-1, 0, 1])
            chunk_filtered = chunk_filtered[valid_hitc]
            
            chunks.append(chunk_filtered)
        
        self.mc56_data = pd.concat(chunks, ignore_index=True)
        
        print(f"MC5-6数据: {len(self.mc56_data):,} 记录")
        print(f"唯一chid: {self.mc56_data['chid'].nunique():,}")
        print(f"唯一aeid: {self.mc56_data['aeid'].nunique():,}")
        
        # hitc分布
        hitc_dist = self.mc56_data['hitc'].value_counts().sort_index()
        print(f"hitc分布: {dict(hitc_dist)}")
        
        return self.mc56_data
    
    def calculate_chemical_scores(self):
        """计算化学品级别的tcpl分数"""
        print("\n🧮 计算化学品级别的tcpl分数")
        print("-" * 50)
        
        if self.mc56_data is None:
            print("❌ 缺少MC5-6数据")
            return None
        
        # 严格的数据清理
        print("应用严格的数据清理规则...")
        
        # 1. 移除hitc = -1的记录（不确定的结果）
        clean_data = self.mc56_data[self.mc56_data['hitc'] != -1].copy()
        print(f"  移除hitc=-1: {len(clean_data):,} 记录保留")
        
        # 2. 移除有mc6_flags的记录（质量标记）
        if 'mc6_flags' in clean_data.columns:
            flagged = clean_data['mc6_flags'].notna() & (clean_data['mc6_flags'] != '')
            clean_data = clean_data[~flagged]
            print(f"  移除mc6_flags: {len(clean_data):,} 记录保留")
        
        # 3. 移除cytotox相关的端点（可选，如果需要更严格）
        cytotox_keywords = ['cytotox', 'viability', 'cell_death']
        if 'aenm' in clean_data.columns:
            cytotox_mask = clean_data['aenm'].str.lower().str.contains('|'.join(cytotox_keywords), na=False)
            clean_data = clean_data[~cytotox_mask]
            print(f"  移除cytotox端点: {len(clean_data):,} 记录保留")
        
        # 按chid聚合计算分数
        print("按chid聚合计算分数...")
        
        chem_stats = clean_data.groupby('chid').agg({
            'hitc': ['count', 'sum'],  # count=总测试数, sum=命中数
            'aeid': 'nunique'  # 测试的端点数
        }).reset_index()
        
        # 展平列名
        chem_stats.columns = ['chid', 'total_tested', 'total_hits', 'unique_endpoints']
        
        # 计算基础分数
        chem_stats['S_chid_basic'] = chem_stats['total_hits'] / chem_stats['total_tested']
        
        # Beta-Binomial收缩（更稳健的估计）
        alpha, beta = 0.5, 0.5
        chem_stats['S_chid_robust'] = (
            (chem_stats['total_hits'] + alpha) / 
            (chem_stats['total_tested'] + alpha + beta)
        )
        
        # 使用稳健估计作为最终分数
        chem_stats['S_chid'] = chem_stats['S_chid_robust']
        
        print(f"化学品级分数计算完成:")
        print(f"  化学品数: {len(chem_stats):,}")
        print(f"  分数范围: {chem_stats['S_chid'].min():.6f} - {chem_stats['S_chid'].max():.6f}")
        print(f"  平均测试数: {chem_stats['total_tested'].mean():.1f}")
        print(f"  平均端点数: {chem_stats['unique_endpoints'].mean():.1f}")
        
        self.chemical_scores = chem_stats
        return chem_stats
    
    def aggregate_to_pubchem(self):
        """聚合到PUBCHEM_CID级别"""
        print("\n🔄 聚合到PUBCHEM_CID级别")
        print("-" * 50)
        
        if self.chemical_scores is None or self.bridge_table is None:
            print("❌ 缺少必要数据")
            return None
        
        # 连接化学品分数和桥表
        chem_with_pubchem = self.chemical_scores.merge(
            self.bridge_table[['chid', 'PUBCHEM_CID']], 
            on='chid', 
            how='inner'
        )
        
        print(f"连接结果: {len(chem_with_pubchem):,} 个chid有PUBCHEM_CID")
        
        # 按PUBCHEM_CID聚合（加权平均，按测试数加权）
        pubchem_stats = chem_with_pubchem.groupby('PUBCHEM_CID').apply(
            lambda group: pd.Series({
                'total_hits': group['total_hits'].sum(),
                'total_tested': group['total_tested'].sum(),
                'unique_endpoints': group['unique_endpoints'].sum(),
                'num_chids': len(group),
                'S_cid': group['total_hits'].sum() / group['total_tested'].sum() if group['total_tested'].sum() > 0 else 0
            })
        ).reset_index()
        
        # Beta-Binomial收缩
        alpha, beta = 0.5, 0.5
        pubchem_stats['S_cid_robust'] = (
            (pubchem_stats['total_hits'] + alpha) / 
            (pubchem_stats['total_tested'] + alpha + beta)
        )
        
        # 使用稳健估计
        pubchem_stats['S_global_real_tcpl'] = pubchem_stats['S_cid_robust']
        
        print(f"PUBCHEM_CID级分数:")
        print(f"  PUBCHEM_CID数: {len(pubchem_stats):,}")
        print(f"  分数范围: {pubchem_stats['S_global_real_tcpl'].min():.6f} - {pubchem_stats['S_global_real_tcpl'].max():.6f}")
        print(f"  平均测试数: {pubchem_stats['total_tested'].mean():.1f}")
        print(f"  平均chid数: {pubchem_stats['num_chids'].mean():.1f}")
        
        self.pubchem_scores = pubchem_stats
        return pubchem_stats

    def anchor_thresholds_with_toxrefdb(self):
        """使用ToxRefDB POD数据锚定阈值"""
        print("\n⚓ 使用ToxRefDB POD数据锚定阈值")
        print("-" * 50)

        if self.pubchem_scores is None:
            print("❌ 缺少PUBCHEM_CID分数")
            return None

        # 加载ToxRefDB数据
        try:
            tox21_data = pd.read_csv('tox21_toxrefdb_matched_via_cas.csv')

            # 连接POD数据
            scores_with_pod = self.pubchem_scores.merge(
                tox21_data[['PUBCHEM_CID', 'POD_MGKGDAY']].drop_duplicates(),
                on='PUBCHEM_CID',
                how='left'
            )

            # 只使用有POD数据的记录进行阈值优化
            pod_data = scores_with_pod.dropna(subset=['POD_MGKGDAY'])

            if len(pod_data) == 0:
                print("❌ 没有POD数据，使用分位数阈值")
                return self.use_quantile_thresholds()

            print(f"有POD数据的化学品: {len(pod_data):,}")

            # 计算不同POD阈值的Youden J指数
            thresholds = {}

            # 二分类：POD ≤ 10 mg/kg/day
            y_binary = (pod_data['POD_MGKGDAY'] <= 10).astype(int)
            if len(np.unique(y_binary)) > 1:
                fpr, tpr, thresh = roc_curve(y_binary, pod_data['S_global_real_tcpl'])
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                thresholds['binary'] = thresh[best_idx]
                print(f"二分类阈值 (POD≤10): {thresholds['binary']:.6f}")
            else:
                thresholds['binary'] = 0.1  # 默认值
                print("二分类阈值: 使用默认值 0.1")

            # 三分类：POD ≤ 3 和 ≤ 30
            y_low = (pod_data['POD_MGKGDAY'] <= 3).astype(int)
            y_high = (pod_data['POD_MGKGDAY'] <= 30).astype(int)

            if len(np.unique(y_low)) > 1:
                fpr, tpr, thresh = roc_curve(y_low, pod_data['S_global_real_tcpl'])
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                thresholds['ternary_high'] = thresh[best_idx]
                print(f"三分类高阈值 (POD≤3): {thresholds['ternary_high']:.6f}")
            else:
                thresholds['ternary_high'] = pod_data['S_global_real_tcpl'].quantile(0.67)
                print(f"三分类高阈值: 使用P67 {thresholds['ternary_high']:.6f}")

            if len(np.unique(y_high)) > 1:
                fpr, tpr, thresh = roc_curve(y_high, pod_data['S_global_real_tcpl'])
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                thresholds['ternary_low'] = thresh[best_idx]
                print(f"三分类低阈值 (POD≤30): {thresholds['ternary_low']:.6f}")
            else:
                thresholds['ternary_low'] = pod_data['S_global_real_tcpl'].quantile(0.33)
                print(f"三分类低阈值: 使用P33 {thresholds['ternary_low']:.6f}")

            return thresholds

        except Exception as e:
            print(f"❌ ToxRefDB阈值锚定失败: {str(e)}")
            return self.use_quantile_thresholds()

    def use_quantile_thresholds(self):
        """使用分位数阈值作为保底方案"""
        print("\n📊 使用分位数阈值")
        print("-" * 50)

        if self.pubchem_scores is None:
            print("❌ 缺少分数数据")
            return None

        scores = self.pubchem_scores['S_global_real_tcpl']

        thresholds = {
            'binary': scores.quantile(0.75),  # P75作为二分类阈值
            'ternary_low': scores.quantile(0.33),  # P33作为三分类低阈值
            'ternary_high': scores.quantile(0.67)  # P67作为三分类高阈值
        }

        print(f"分位数阈值:")
        print(f"  二分类 (P75): {thresholds['binary']:.6f}")
        print(f"  三分类低 (P33): {thresholds['ternary_low']:.6f}")
        print(f"  三分类高 (P67): {thresholds['ternary_high']:.6f}")

        return thresholds

    def create_classification_labels(self, thresholds):
        """创建分类标签"""
        print("\n🏷️ 创建分类标签")
        print("-" * 50)

        if self.pubchem_scores is None or thresholds is None:
            print("❌ 缺少必要数据")
            return None

        scores_with_labels = self.pubchem_scores.copy()

        # 二分类标签
        scores_with_labels['tcpl_binary'] = (
            scores_with_labels['S_global_real_tcpl'] >= thresholds['binary']
        ).astype(int)

        # 三分类标签
        scores_with_labels['tcpl_ternary'] = 0  # 默认低毒性
        scores_with_labels.loc[
            scores_with_labels['S_global_real_tcpl'] >= thresholds['ternary_low'],
            'tcpl_ternary'
        ] = 1  # 中等毒性
        scores_with_labels.loc[
            scores_with_labels['S_global_real_tcpl'] >= thresholds['ternary_high'],
            'tcpl_ternary'
        ] = 2  # 高毒性

        # 统计分布
        binary_dist = scores_with_labels['tcpl_binary'].value_counts().sort_index()
        ternary_dist = scores_with_labels['tcpl_ternary'].value_counts().sort_index()

        print(f"分类标签分布:")
        print(f"  二分类 - 低毒性: {binary_dist.get(0, 0):,}, 高毒性: {binary_dist.get(1, 0):,}")
        print(f"  三分类 - 低毒性: {ternary_dist.get(0, 0):,}, 中等: {ternary_dist.get(1, 0):,}, 高毒性: {ternary_dist.get(2, 0):,}")

        # 保存阈值信息
        scores_with_labels['threshold_binary'] = thresholds['binary']
        scores_with_labels['threshold_ternary_low'] = thresholds['ternary_low']
        scores_with_labels['threshold_ternary_high'] = thresholds['ternary_high']

        self.pubchem_scores = scores_with_labels
        return scores_with_labels

    def add_labels_to_original_dataset(self, original_file='output/data/processed_final8k213.csv'):
        """将tcpl标签添加到原始数据集"""
        print("\n📝 将tcpl标签添加到原始数据集")
        print("-" * 50)

        if self.pubchem_scores is None:
            print("❌ 缺少PUBCHEM_CID分数")
            return None

        # 加载原始数据集
        original_data = pd.read_csv(original_file)
        print(f"原始数据集: {len(original_data):,} 记录")

        # 准备要添加的标签
        labels_to_add = self.pubchem_scores[[
            'PUBCHEM_CID', 'S_global_real_tcpl', 'tcpl_binary', 'tcpl_ternary',
            'total_hits', 'total_tested', 'unique_endpoints', 'num_chids'
        ]].copy()

        # 左连接
        enhanced_data = original_data.merge(labels_to_add, on='PUBCHEM_CID', how='left')

        # 填充缺失值
        enhanced_data['S_global_real_tcpl'].fillna(-1, inplace=True)
        enhanced_data['tcpl_binary'].fillna(-1, inplace=True)
        enhanced_data['tcpl_ternary'].fillna(-1, inplace=True)
        enhanced_data['total_hits'].fillna(0, inplace=True)
        enhanced_data['total_tested'].fillna(0, inplace=True)
        enhanced_data['unique_endpoints'].fillna(0, inplace=True)
        enhanced_data['num_chids'].fillna(0, inplace=True)

        # 转换数据类型
        enhanced_data['tcpl_binary'] = enhanced_data['tcpl_binary'].astype(int)
        enhanced_data['tcpl_ternary'] = enhanced_data['tcpl_ternary'].astype(int)
        enhanced_data['total_hits'] = enhanced_data['total_hits'].astype(int)
        enhanced_data['total_tested'] = enhanced_data['total_tested'].astype(int)
        enhanced_data['unique_endpoints'] = enhanced_data['unique_endpoints'].astype(int)
        enhanced_data['num_chids'] = enhanced_data['num_chids'].astype(int)

        # 统计结果
        total = len(enhanced_data)
        mapped = (enhanced_data['tcpl_binary'] != -1).sum()

        print(f"标签添加结果:")
        print(f"  总记录数: {total:,}")
        print(f"  成功映射: {mapped:,} ({mapped/total*100:.2f}%)")
        print(f"  未映射: {total-mapped:,} ({(total-mapped)/total*100:.2f}%)")

        return enhanced_data

    def run_complete_system(self):
        """运行完整的桥表系统"""
        print("🚀 运行完整的化学品桥表系统")
        print("=" * 60)

        # 1. 构建基础桥表
        self.load_sc2_data()

        # 2. 加载CAS→PUBCHEM_CID映射
        self.load_cas_pubchem_mappings()

        # 3. 创建完整桥表
        self.create_complete_bridge_table()

        # 4. 加载MC5-6数据
        self.load_mc56_data()

        # 5. 计算化学品级分数
        self.calculate_chemical_scores()

        # 6. 聚合到PUBCHEM_CID级别
        self.aggregate_to_pubchem()

        # 7. 锚定阈值
        thresholds = self.anchor_thresholds_with_toxrefdb()

        # 8. 创建分类标签
        self.create_classification_labels(thresholds)

        # 9. 添加到原始数据集
        enhanced_data = self.add_labels_to_original_dataset()

        # 10. 保存结果
        if enhanced_data is not None:
            output_file = 'output/data/processed_final8k213_bridge_system_labels.csv'
            enhanced_data.to_csv(output_file, index=False)
            print(f"\n💾 最终结果已保存: {output_file}")

            # 保存PUBCHEM_CID级分数
            pubchem_file = 'output/data/pubchem_tcpl_scores_bridge_system.csv'
            self.pubchem_scores.to_csv(pubchem_file, index=False)
            print(f"PUBCHEM_CID分数已保存: {pubchem_file}")

        print("\n🎉 化学品桥表系统运行完成!")
        return enhanced_data

def main():
    """主函数"""
    bridge_system = ChemicalBridgeSystem()
    result = bridge_system.run_complete_system()
    return result

if __name__ == "__main__":
    result = main()
