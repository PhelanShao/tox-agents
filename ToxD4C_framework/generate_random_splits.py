#!/usr/bin/env python3
"""
生成真正的随机切分数据
用于A1任务的随机vs scaffold性能对比
"""

import os
import sys
import json
import lmdb
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.splitter import MolecularSplitter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomSplitGenerator:
    """随机切分生成器"""
    
    def __init__(self, random_state: int = 42):
        self.splitter = MolecularSplitter(random_state=random_state)
        self.random_state = random_state
    
    def load_lmdb_data(self, lmdb_path: str) -> Tuple[List[str], List[Dict]]:
        """从LMDB文件加载数据"""
        logger.info(f"Loading data from: {lmdb_path}")

        smiles_list = []
        data_list = []

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in tqdm(cursor, desc="Loading LMDB data"):
                # 跳过LMDB元数据键
                if key == b'__keys__' or key == b'__len__':
                    continue

                try:
                    data = pickle.loads(value)
                    # 验证数据完整性
                    if 'smiles' not in data:
                        logger.warning(f"Entry {key} missing 'smiles' field")
                        continue

                    smiles_list.append(data['smiles'])
                    data_list.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load entry {key}: {e}")
                    continue

        env.close()
        logger.info(f"Loaded {len(smiles_list)} molecules")
        return smiles_list, data_list
    
    def save_lmdb_split(self, data_list: List[Dict], indices: List[int], output_path: str):
        """保存切分后的数据到LMDB"""
        logger.info(f"Saving {len(indices)} samples to: {output_path}")
        
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 如果文件已存在，删除它
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        
        env = lmdb.open(output_path, map_size=10**12)
        
        with env.begin(write=True) as txn:
            for i, idx in enumerate(tqdm(indices, desc=f"Writing {Path(output_path).name}")):
                data = data_list[idx]
                key = f"{i:08d}".encode()
                value = pickle.dumps(data)
                txn.put(key, value)
        
        env.close()
        logger.info(f"Saved {len(indices)} samples to {output_path}")
    
    def create_random_splits(self, 
                            input_lmdb: str, 
                            output_dir: str,
                            train_size: float = 0.8,
                            val_size: float = 0.1):
        """创建随机切分"""
        
        logger.info("🎲 Starting random split generation...")
        logger.info(f"Input LMDB: {input_lmdb}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Split ratios - Train: {train_size}, Val: {val_size}, Test: {1-train_size-val_size}")
        
        # 加载数据
        smiles_list, data_list = self.load_lmdb_data(input_lmdb)
        
        if len(smiles_list) == 0:
            logger.error("No data loaded from LMDB file")
            return
        
        # 执行随机切分
        logger.info("🔀 Performing random split...")
        train_indices, val_indices, test_indices = self.splitter.random_split(
            smiles_list, 
            train_size=train_size, 
            val_size=val_size
        )
        
        # 验证切分结果
        total_samples = len(smiles_list)
        actual_train_size = len(train_indices) / total_samples
        actual_val_size = len(val_indices) / total_samples
        actual_test_size = len(test_indices) / total_samples
        
        logger.info("📊 Split statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Train: {len(train_indices)} ({actual_train_size:.3f})")
        logger.info(f"  Validation: {len(val_indices)} ({actual_val_size:.3f})")
        logger.info(f"  Test: {len(test_indices)} ({actual_test_size:.3f})")
        
        # 检查重叠
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        overlap_train_val = len(train_set & val_set)
        overlap_train_test = len(train_set & test_set)
        overlap_val_test = len(val_set & test_set)
        
        if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
            logger.error(f"Split overlap detected! Train-Val: {overlap_train_val}, Train-Test: {overlap_train_test}, Val-Test: {overlap_val_test}")
            return
        
        logger.info("✅ No overlap between splits")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存切分后的数据
        logger.info("💾 Saving split datasets...")
        
        self.save_lmdb_split(data_list, train_indices, str(output_path / "train.lmdb"))
        self.save_lmdb_split(data_list, val_indices, str(output_path / "valid.lmdb"))
        self.save_lmdb_split(data_list, test_indices, str(output_path / "test.lmdb"))
        
        # 保存切分信息
        split_info = {
            'split_method': 'random',
            'random_state': self.random_state,
            'total_samples': total_samples,
            'train_size': actual_train_size,
            'val_size': actual_val_size,
            'test_size': actual_test_size,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
        
        split_info_path = output_path / "split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split information saved to: {split_info_path}")
        logger.info("🎉 Random split generation completed!")
        
        return {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'output_dir': str(output_path)
        }

def find_original_data():
    """查找原始数据文件"""
    
    possible_paths = [
        "data/dataset",
        "data/original", 
        "data/raw",
        "data",
        "../data/dataset",
        "../data"
    ]
    
    for path in possible_paths:
        full_path = Path(path)
        if full_path.exists():
            # 查找包含所有数据的单个LMDB文件
            all_lmdb = full_path / "all.lmdb"
            if all_lmdb.exists():
                return str(all_lmdb)
            
            # 查找train.lmdb（假设包含所有数据）
            train_lmdb = full_path / "train.lmdb"
            if train_lmdb.exists():
                return str(train_lmdb)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate random splits for ToxD4C dataset')
    parser.add_argument('--input_data', type=str, default=None,
                       help='Path to input LMDB file containing all data')
    parser.add_argument('--output_dir', type=str, default='data/random_split',
                       help='Output directory for random split data')
    parser.add_argument('--train_size', type=float, default=0.8,
                       help='Training set fraction')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set fraction')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # 查找输入数据
    if args.input_data is None:
        logger.info("🔍 Searching for original data...")
        input_data = find_original_data()
        if input_data is None:
            logger.error("❌ Could not find original data file. Please specify --input_data")
            logger.info("Expected locations:")
            logger.info("  - data/dataset/all.lmdb")
            logger.info("  - data/dataset/train.lmdb")
            logger.info("  - data/original/all.lmdb")
            return
        logger.info(f"✅ Found data at: {input_data}")
    else:
        input_data = args.input_data
    
    # 检查输入文件
    if not os.path.exists(input_data):
        logger.error(f"❌ Input file not found: {input_data}")
        return
    
    # 创建切分器
    splitter = RandomSplitGenerator(random_state=args.random_state)
    
    # 生成随机切分
    result = splitter.create_random_splits(
        input_lmdb=input_data,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size
    )
    
    if result:
        logger.info("🎉 Success! Random split data generated.")
        logger.info(f"📁 Output directory: {result['output_dir']}")
        logger.info("📋 Generated files:")
        logger.info("  - train.lmdb")
        logger.info("  - valid.lmdb") 
        logger.info("  - test.lmdb")
        logger.info("  - split_info.json")
        
        logger.info("\n🚀 Next steps:")
        logger.info(f"python train.py \\")
        logger.info(f"    --experiment_name 'toxd4c_baseline_random' \\")
        logger.info(f"    --data_dir '{result['output_dir']}' \\")
        logger.info(f"    --deterministic \\")
        logger.info(f"    --seed 42 \\")
        logger.info(f"    --batch_size 16 \\")
        logger.info(f"    --num_epochs 50")

if __name__ == "__main__":
    main()
