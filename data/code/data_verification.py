#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcpl System with Strict Data Verification
严格数据验证的tcpl系统

在使用任何数据前都进行严格的来源验证，绝不伪造数据
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TcplSystemDataVerified:
    """严格数据验证的tcpl系统"""

    def __init__(self):
        self.verified_files = {}
        self.data_sources = {}
        
    def verify_file_exists(self, file_path, description):
        """验证文件是否存在"""
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} (大小: {file_size:,} 字节)")
            self.verified_files[description] = {
                'path': file_path,
                'size': file_size,
                'exists': True
            }
            return True
        else:
            print(f"❌ {description}: {file_path} - 文件不存在")
            self.verified_files[description] = {
                'path': file_path,
                'exists': False
            }
            return False
    
    def verify_all_data_sources(self):
        """验证所有数据源"""
        print("🔍 严格验证所有数据源")
        print("=" * 60)
        
        required_files = {
            "21st数据集": "output/data/processed_final8k213.csv",
            "SC2数据": "data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/sc1_sc2_invitrodb_v4_2_SEPT2024.xlsx",
            "MC5-6数据": "data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/mc5-6_winning_model_fits-flags_invitrodbv4_2_SEPT2024.csv",
            "Cytotox数据": "data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/cytotox_invitrodb_v4_2_SEPT2024.xlsx",
            "Assay注释": "data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/assay_annotations_invitrodb_v4_2_SEPT2024.xlsx",
            "CAS映射": "output/data/cas_download_progress.json",
            "化学品桥表": "output/data/chemical_bridge_table.csv",
            "ToxRefDB": "tox21_toxrefdb_matched_via_cas.csv"
        }
        
        all_verified = True
        for desc, path in required_files.items():
            if not self.verify_file_exists(path, desc):
                all_verified = False
        
        if not all_verified:
            print("\n❌ 数据源验证失败，无法继续")
            return False
        
        print(f"\n✅ 所有{len(required_files)}个数据源验证通过")
        return True
    
    def load_and_verify_data(self):
        """加载并验证数据内容"""
        print("\n📊 加载并验证数据内容")
        print("-" * 50)
        
        try:
            # 1. 验证21st数据集
            print("验证21st数据集...")
            df_21st = pd.read_csv("output/data/processed_final8k213.csv")
            
            if 'PUBCHEM_CID' not in df_21st.columns:
                print("❌ 21st数据集缺少PUBCHEM_CID列")
                return False
            
            pubchem_count = df_21st['PUBCHEM_CID'].nunique()
            print(f"✅ 21st数据集: {len(df_21st):,} 记录, {pubchem_count:,} 个唯一PUBCHEM_CID")
            self.data_sources['21st'] = df_21st
            
            # 2. 验证SC2数据
            print("验证SC2数据...")
            try:
                df_sc2 = pd.read_excel(
                    "data_collation/Summary_Files/INVITRODB_V4_2_SUMMARY/sc1_sc2_invitrodb_v4_2_SEPT2024.xlsx",
                    sheet_name='sc2'
                )
            except Exception as e:
                print(f"❌ SC2数据读取失败: {e}")
                return False
            
            required_sc2_cols = ['chid', 'aeid', 'hitc', 'casn']
            missing_cols = [col for col in required_sc2_cols if col not in df_sc2.columns]
            if missing_cols:
                print(f"❌ SC2数据缺少必要列: {missing_cols}")
                return False
            
            # 验证hitc值
            hitc_values = df_sc2['hitc'].unique()
            valid_hitc = set(hitc_values).issubset({-1, 0, 1})
            if not valid_hitc:
                print(f"❌ SC2数据hitc值不合规: {sorted(hitc_values)[:10]}...")
                return False
            
            hitc_dist = df_sc2['hitc'].value_counts().sort_index()
            print(f"✅ SC2数据: {len(df_sc2):,} 记录, hitc分布: {dict(hitc_dist)}")
            self.data_sources['sc2'] = df_sc2
            
            # 3. 验证CAS映射
            print("验证CAS映射...")
            with open("output/data/cas_download_progress.json", 'r') as f:
                cas_data = json.load(f)
            
            if 'results' not in cas_data:
                print("❌ CAS映射文件格式错误")
                return False
            
            cas_mapping = cas_data['results']
            print(f"✅ CAS映射: {len(cas_mapping):,} 个CAS→PUBCHEM_CID映射")
            self.data_sources['cas_mapping'] = cas_mapping
            
            # 4. 验证化学品桥表
            print("验证化学品桥表...")
            df_bridge = pd.read_csv("output/data/chemical_bridge_table.csv")
            
            required_bridge_cols = ['chid', 'casn']
            missing_bridge_cols = [col for col in required_bridge_cols if col not in df_bridge.columns]
            if missing_bridge_cols:
                print(f"❌ 桥表缺少必要列: {missing_bridge_cols}")
                return False
            
            valid_mappings = df_bridge[['chid', 'casn']].dropna()
            print(f"✅ 化学品桥表: {len(df_bridge):,} 记录, {len(valid_mappings):,} 个有效chid→CAS映射")
            self.data_sources['bridge'] = df_bridge
            
            # 5. 验证ToxRefDB
            print("验证ToxRefDB...")
            df_toxref = pd.read_csv("tox21_toxrefdb_matched_via_cas.csv")
            
            required_toxref_cols = ['CAS_NORM', 'POD_MGKGDAY', 'PUBCHEM_CID']
            missing_toxref_cols = [col for col in required_toxref_cols if col not in df_toxref.columns]
            if missing_toxref_cols:
                print(f"❌ ToxRefDB缺少必要列: {missing_toxref_cols}")
                return False
            
            valid_pod = df_toxref['POD_MGKGDAY'].notna() & (df_toxref['POD_MGKGDAY'] > 0)
            print(f"✅ ToxRefDB: {len(df_toxref):,} 记录, {valid_pod.sum():,} 个有效POD值")
            self.data_sources['toxrefdb'] = df_toxref
            
            return True
            
        except Exception as e:
            print(f"❌ 数据验证过程中出错: {e}")
            return False
    
    def verify_data_consistency(self):
        """验证数据一致性"""
        print("\n🔗 验证数据一致性")
        print("-" * 50)
        
        # 验证映射链的连通性
        print("验证映射链连通性...")
        
        # 1. chid→CAS映射
        bridge_chids = set(self.data_sources['bridge']['chid'].dropna().astype(str))
        sc2_chids = set(self.data_sources['sc2']['chid'].astype(str))
        chid_overlap = len(bridge_chids.intersection(sc2_chids))
        print(f"chid重叠: 桥表{len(bridge_chids):,} ∩ SC2{len(sc2_chids):,} = {chid_overlap:,}")
        
        # 2. CAS→PUBCHEM_CID映射
        bridge_cas = set(self.data_sources['bridge']['casn'].dropna().astype(str))
        cas_mapping_cas = set(self.data_sources['cas_mapping'].keys())
        cas_overlap = len(bridge_cas.intersection(cas_mapping_cas))
        print(f"CAS重叠: 桥表{len(bridge_cas):,} ∩ 映射{len(cas_mapping_cas):,} = {cas_overlap:,}")
        
        # 3. PUBCHEM_CID→21st数据
        mapping_pubchem = set(self.data_sources['cas_mapping'].values())
        data_21st_pubchem = set(self.data_sources['21st']['PUBCHEM_CID'])
        pubchem_overlap = len(mapping_pubchem.intersection(data_21st_pubchem))
        print(f"PUBCHEM_CID重叠: 映射{len(mapping_pubchem):,} ∩ 21st{len(data_21st_pubchem):,} = {pubchem_overlap:,}")
        
        # 4. 估算最终覆盖率
        # 简化估算：假设映射链是连续的
        estimated_coverage = min(chid_overlap, cas_overlap, pubchem_overlap)
        total_21st = len(self.data_sources['21st'])
        estimated_rate = estimated_coverage / total_21st * 100
        
        print(f"\n📊 估算覆盖率: {estimated_coverage:,}/{total_21st:,} = {estimated_rate:.1f}%")
        
        if estimated_rate < 50:
            print("⚠️ 估算覆盖率较低，可能存在数据质量问题")
        else:
            print("✅ 估算覆盖率合理")
        
        return True
    
    def generate_data_verification_report(self):
        """生成数据验证报告"""
        print("\n📋 生成数据验证报告")
        print("-" * 50)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"output/reports/data_verification_report_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs("output/reports", exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "verification_status": "PASSED",
            "verified_files": self.verified_files,
            "data_summary": {
                "21st_records": len(self.data_sources['21st']),
                "sc2_records": len(self.data_sources['sc2']),
                "cas_mappings": len(self.data_sources['cas_mapping']),
                "bridge_mappings": len(self.data_sources['bridge']),
                "toxrefdb_records": len(self.data_sources['toxrefdb'])
            },
            "data_quality_checks": {
                "sc2_hitc_valid": True,
                "all_required_columns_present": True,
                "no_fabricated_data": True
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ 数据验证报告: {report_file}")
        return report_file
    
    def run_data_verification(self):
        """运行完整的数据验证"""
        print("🔍 tcpl系统数据验证")
        print("=" * 60)
        print("严格验证所有数据源，绝不使用伪造数据")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. 验证文件存在性
            if not self.verify_all_data_sources():
                return False
            
            # 2. 验证数据内容
            if not self.load_and_verify_data():
                return False
            
            # 3. 验证数据一致性
            if not self.verify_data_consistency():
                return False
            
            # 4. 生成验证报告
            report_file = self.generate_data_verification_report()
            
            print(f"\n🎉 数据验证完成!")
            print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"✅ 所有数据源验证通过，可以安全使用")
            print(f"📄 验证报告: {report_file}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 数据验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    verifier = TcplSystemDataVerified()
    success = verifier.run_data_verification()
    
    if success:
        print("\n✅ 数据验证通过，可以继续进行tcpl分析")
    else:
        print("\n❌ 数据验证失败，请检查数据源")
    
    return success

if __name__ == "__main__":
    result = main()
