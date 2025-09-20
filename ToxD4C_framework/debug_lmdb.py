"""
调试LMDB数据加载问题
"""

import lmdb
import pickle
import sys
from pathlib import Path

def test_lmdb_basic():
    """基础LMDB测试"""
    lmdb_path = "/mnt/backup3/toxscan/ToxScan/Toxd4c/Toxd4c/data/dataset/train.lmdb"
    
    print(f"测试LMDB文件: {lmdb_path}")
    print(f"文件是否存在: {Path(lmdb_path).exists()}")
    
    try:
        # 打开LMDB环境
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )
        print("✓ LMDB环境打开成功")
        
        # 获取数据库统计信息
        with env.begin() as txn:
            stat = txn.stat()
            print(f"数据库统计信息:")
            print(f"  页面大小: {stat['psize']}")
            print(f"  树深度: {stat['depth']}")
            print(f"  分支页面数: {stat['branch_pages']}")
            print(f"  叶子页面数: {stat['leaf_pages']}")
            print(f"  条目数: {stat['entries']}")
        
        # 收集键
        print("开始收集键...")
        keys = []
        count = 0
        
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                count += 1
                if count % 1000 == 0:
                    print(f"已处理 {count} 个条目...")
                
                try:
                    key_str = key.decode('ascii')
                    if not key_str.isdigit() and key_str != 'length':
                        keys.append(key_str)
                        if len(keys) <= 5:  # 只显示前5个
                            print(f"  键 {len(keys)}: {key_str}")
                except:
                    continue
                
                # 限制处理数量以避免卡住
                if count >= 10000:
                    print("达到处理限制，停止...")
                    break
        
        print(f"✓ 找到 {len(keys)} 个有效SMILES键")
        
        # 测试读取第一个样本
        if keys:
            print(f"测试读取第一个样本: {keys[0]}")
            with env.begin() as txn:
                data_bytes = txn.get(keys[0].encode('ascii'))
                if data_bytes:
                    data = pickle.loads(data_bytes)
                    print(f"✓ 成功读取样本数据")
                    print(f"  数据键: {list(data.keys())}")
                    if 'atoms' in data:
                        print(f"  原子数量: {len(data['atoms'])}")
                    if 'coordinates' in data:
                        print(f"  坐标形状: {len(data['coordinates']) if isinstance(data['coordinates'], list) else 'N/A'}")
                else:
                    print("✗ 无法读取样本数据")
        
        env.close()
        print("✓ LMDB测试完成")
        
    except Exception as e:
        print(f"✗ LMDB测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lmdb_basic()