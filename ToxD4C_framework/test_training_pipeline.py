#!/usr/bin/env python3
"""
Test the complete training pipeline with preprocessed data
"""

import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading with preprocessed data."""
    logger.info("🧪 Testing data loading...")
    
    try:
        from data.lmdb_dataset import create_lmdb_dataloaders
        
        train_loader, val_loader, test_loader = create_lmdb_dataloaders(
            "data/data/processed",
            batch_size=4
        )
        
        logger.info(f"✅ Data loaders created successfully")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        # Test loading a few batches
        logger.info("🔄 Testing batch loading...")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            
            logger.info(f"   Batch {i}: {len(batch['smiles'])} samples")
            logger.info(f"     atom_features: {batch['atom_features'].shape}")
            logger.info(f"     coordinates: {batch['coordinates'].shape}")
            
            # Check for any issues
            if torch.isnan(batch['atom_features']).any():
                logger.warning(f"     ⚠️  NaN detected in atom_features")
            if torch.isnan(batch['coordinates']).any():
                logger.warning(f"     ⚠️  NaN detected in coordinates")
        
        logger.info("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    logger.info("🧪 Testing model creation...")
    
    try:
        from models.toxd4c import ToxD4C
        from configs.toxd4c_config import get_enhanced_toxd4c_config
        
        config = get_enhanced_toxd4c_config()
        model = ToxD4C(config)
        
        logger.info(f"✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {e}")
        return False

def test_forward_pass():
    """Test model forward pass with real data."""
    logger.info("🧪 Testing model forward pass...")
    
    try:
        from data.lmdb_dataset import create_lmdb_dataloaders
        from models.toxd4c import ToxD4C
        from configs.toxd4c_config import get_enhanced_toxd4c_config
        
        # Create model
        config = get_enhanced_toxd4c_config()
        model = ToxD4C(config)
        
        # Create data loader
        train_loader, _, _ = create_lmdb_dataloaders(
            "data/data/processed",
            batch_size=2
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        logger.info(f"   Batch size: {len(batch['smiles'])}")
        
        # Forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Move batch to device
        for key in ['atom_features', 'coordinates', 'bond_features', 'edge_index', 'batch']:
            if key in batch:
                batch[key] = batch[key].to(device)
        
        logger.info(f"   Using device: {device}")
        
        with torch.no_grad():
            outputs = model(batch, batch['smiles'])
        
        logger.info(f"✅ Forward pass successful!")
        logger.info(f"   Output keys: {list(outputs.keys())}")
        
        if 'predictions' in outputs:
            preds = outputs['predictions']
            logger.info(f"   Prediction keys: {list(preds.keys())}")
            
            if 'classification' in preds:
                logger.info(f"   Classification shape: {preds['classification'].shape}")
            if 'regression' in preds:
                logger.info(f"   Regression shape: {preds['regression'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Testing ToxD4C Training Pipeline")
    logger.info("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Training pipeline is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
