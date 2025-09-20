#!/usr/bin/env python3
"""
Debug script to check if preprocessed data is being used correctly
"""

import lmdb
import pickle
import os

def check_preprocessed_data():
    """Check the actual content of preprocessed data."""
    
    lmdb_path = "data/data/processed/train.lmdb"
    print(f"🔍 Checking preprocessed data: {lmdb_path}")
    
    # Auto-detect LMDB format
    subdir_flag = os.path.isdir(lmdb_path)
    print(f"📁 LMDB format: {'directory' if subdir_flag else 'file'}")
    
    env = lmdb.open(lmdb_path, subdir=subdir_flag, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        
        # Check first few samples
        for i, (key, value) in enumerate(cursor):
            if i >= 5:  # Check first 5 samples
                break
                
            key_str = key.decode('ascii')
            print(f"\n📋 Sample {i}: {key_str}")
            
            # Skip special keys
            if key_str in ['__keys__', 'length'] or key_str.isdigit():
                print(f"   ⏭️  Skipping special key: {key_str}")
                continue
            
            try:
                data = pickle.loads(value)
                print(f"   📊 Data keys: {list(data.keys())}")
                
                # Check for preprocessed features
                has_preprocessed = all(k in data for k in ['atom_features', 'coordinates', 'bond_features', 'edge_index'])
                print(f"   ✅ Has preprocessed features: {has_preprocessed}")
                
                if has_preprocessed:
                    print(f"   🧬 atom_features: {type(data['atom_features'])}, shape: {getattr(data['atom_features'], 'shape', 'N/A')}")
                    print(f"   📍 coordinates: {type(data['coordinates'])}, shape: {getattr(data['coordinates'], 'shape', 'N/A')}")
                    print(f"   🔗 bond_features: {type(data['bond_features'])}, shape: {getattr(data['bond_features'], 'shape', 'N/A')}")
                    print(f"   🌐 edge_index: {type(data['edge_index'])}, shape: {getattr(data['edge_index'], 'shape', 'N/A')}")
                else:
                    print(f"   ❌ Missing preprocessed features!")
                    
            except Exception as e:
                print(f"   💥 Error loading data: {e}")
    
    env.close()

def check_data_loading():
    """Test the actual data loading process."""
    print(f"\n🧪 Testing data loading process...")
    
    try:
        from data.lmdb_dataset import LMDBToxD4CDataset
        
        # Test preprocessed data loading
        dataset = LMDBToxD4CDataset("data/data/processed/train.lmdb")
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Test first sample
        sample = dataset[0]
        if sample is not None:
            print(f"✅ Successfully loaded sample 0")
            print(f"   SMILES: {sample['smiles']}")
            print(f"   atom_features shape: {sample['atom_features'].shape}")
            print(f"   coordinates shape: {sample['coordinates'].shape}")
        else:
            print(f"❌ Failed to load sample 0")
            
    except Exception as e:
        print(f"💥 Error in data loading test: {e}")

if __name__ == "__main__":
    check_preprocessed_data()
    check_data_loading()
