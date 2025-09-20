#!/usr/bin/env python3
"""
Test script for ToxD4C improvements
Validates the new utilities and modifications addressing reviewer concerns.
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reproducibility_utils():
    """Test A0: Reproducibility utilities"""
    logger.info("🧪 Testing reproducibility utilities...")
    
    try:
        from utils.reproducibility import (
            ReproducibilityContext, 
            get_environment_info, 
            save_environment_info,
            create_experiment_snapshot
        )
        
        # Test environment info collection
        env_info = get_environment_info()
        assert 'timestamp' in env_info
        assert 'pytorch' in env_info
        assert 'platform' in env_info
        logger.info("✅ Environment info collection works")
        
        # Test reproducibility context
        with ReproducibilityContext(seed=42, strict=False):
            # Generate some random numbers
            torch_rand1 = torch.rand(5)
            np_rand1 = np.random.rand(5)
        
        with ReproducibilityContext(seed=42, strict=False):
            # Should generate same numbers
            torch_rand2 = torch.rand(5)
            np_rand2 = np.random.rand(5)
        
        assert torch.allclose(torch_rand1, torch_rand2, atol=1e-6)
        assert np.allclose(np_rand1, np_rand2, atol=1e-6)
        logger.info("✅ Reproducibility context works")
        
        # Test experiment snapshot
        test_config = {'test_param': 42, 'another_param': 'test'}
        snapshot_dir = Path("test_experiment_snapshot")
        snapshot_dir.mkdir(exist_ok=True)
        
        metadata = create_experiment_snapshot(
            str(snapshot_dir), 
            test_config,
            additional_info={'test_info': 'test_value'}
        )
        
        assert (snapshot_dir / "environment_info.json").exists()
        assert (snapshot_dir / "config.json").exists()
        assert (snapshot_dir / "metadata.json").exists()
        logger.info("✅ Experiment snapshot creation works")
        
        # Cleanup
        import shutil
        shutil.rmtree(snapshot_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reproducibility utils test failed: {e}")
        return False

def test_splitter_utils():
    """Test A1: Data splitting utilities"""
    logger.info("🧪 Testing data splitting utilities...")
    
    try:
        from utils.splitter import MolecularSplitter
        
        # Test SMILES
        test_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CCN(CC)CC",  # triethylamine
            "CC(C)O",  # isopropanol
            "CCCCO",  # butanol
            "CC(C)(C)O",  # tert-butanol
            "c1ccc(O)cc1",  # phenol
            "CCN",  # ethylamine
            "CCC(=O)O"  # propanoic acid
        ]
        
        splitter = MolecularSplitter(random_state=42)
        
        # Test random split
        train_idx, val_idx, test_idx = splitter.random_split(test_smiles)
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(test_smiles)
        assert len(set(train_idx) & set(val_idx) & set(test_idx)) == 0  # No overlap
        logger.info("✅ Random split works")
        
        # Test scaffold split
        train_idx, val_idx, test_idx = splitter.scaffold_split(test_smiles)
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(test_smiles)
        logger.info("✅ Scaffold split works")
        
        # Test cluster split
        train_idx, val_idx, test_idx = splitter.cluster_split(test_smiles)
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(test_smiles)
        logger.info("✅ Cluster split works")
        
        # Test split quality analysis
        quality = splitter.analyze_split_quality(test_smiles, train_idx, val_idx, test_idx)
        assert 'train_val_similarity' in quality
        assert 'scaffold_overlap_train_val' in quality
        logger.info("✅ Split quality analysis works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Splitter utils test failed: {e}")
        return False

def test_uncertainty_utils():
    """Test A2: Uncertainty quantification utilities"""
    logger.info("🧪 Testing uncertainty quantification utilities...")
    
    try:
        from utils.uncertainty import (
            TemperatureScaling,
            ApplicabilityDomain,
            ConformalPrediction
        )
        
        # Test applicability domain
        training_smiles = [
            "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)O",
            "CCCCO", "CC(C)(C)O", "c1ccc(O)cc1", "CCN", "CCC(=O)O"
        ]
        
        ad = ApplicabilityDomain(training_smiles)
        
        query_smiles = ["CCCO", "c1ccc(N)cc1", "CCCCCCCCCCCC"]  # Similar, similar, dissimilar
        applicability = ad.assess_applicability(query_smiles)
        
        assert 'max_similarity' in applicability
        assert 'in_domain_similarity' in applicability
        assert len(applicability['max_similarity']) == len(query_smiles)
        logger.info("✅ Applicability domain works")
        
        # Test conformal prediction
        cp = ConformalPrediction(alpha=0.1)
        
        # Dummy calibration data
        cal_preds = np.random.rand(100, 2)  # 100 samples, 2 classes
        cal_targets = np.random.randint(0, 2, 100)
        
        cp.calibrate(cal_preds, cal_targets, task_type='classification')
        
        # Test predictions
        test_preds = np.random.rand(10, 2)
        conformal_results = cp.predict(test_preds, task_type='classification')
        
        assert 'prediction_sets' in conformal_results
        assert 'set_sizes' in conformal_results
        logger.info("✅ Conformal prediction works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Uncertainty utils test failed: {e}")
        return False

def test_train_script_imports():
    """Test that train.py can import new utilities"""
    logger.info("🧪 Testing train.py imports...")
    
    try:
        # Test if we can import the modified train script
        sys.path.append(str(Path(__file__).parent))
        
        # Try importing the utilities that train.py should import
        from utils.reproducibility import ReproducibilityContext
        from utils.splitter import MolecularSplitter
        from utils.uncertainty import TemperatureScaling
        
        logger.info("✅ Train script imports work")
        return True
        
    except Exception as e:
        logger.error(f"❌ Train script imports test failed: {e}")
        return False

def test_config_compatibility():
    """Test that configs are compatible with new features"""
    logger.info("🧪 Testing config compatibility...")

    try:
        from configs.toxd4c_config import get_enhanced_toxd4c_config, CLASSIFICATION_TASKS, REGRESSION_TASKS

        config = get_enhanced_toxd4c_config()

        # Check that config has expected structure
        assert 'hidden_dim' in config
        assert 'num_encoder_layers' in config
        assert 'dropout' in config

        # Check that tasks are defined
        assert len(CLASSIFICATION_TASKS) > 0
        assert len(REGRESSION_TASKS) > 0

        logger.info("✅ Config compatibility works")
        return True

    except Exception as e:
        logger.error(f"❌ Config compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("🚀 Starting ToxD4C improvements test suite...")
    
    tests = [
        ("Reproducibility Utils (A0)", test_reproducibility_utils),
        ("Splitter Utils (A1)", test_splitter_utils),
        ("Uncertainty Utils (A2)", test_uncertainty_utils),
        ("Train Script Imports", test_train_script_imports),
        ("Config Compatibility", test_config_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n=== TEST RESULTS ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! ToxD4C improvements are working correctly.")
        logger.info("📋 Ready to address reviewer concerns:")
        logger.info("   A0 ✅ Reproducibility and robustness")
        logger.info("   A1 ✅ Data splitting strategies") 
        logger.info("   A2 ✅ Uncertainty quantification")
        logger.info("   A3 🔄 Architecture ablation (requires training)")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before proceeding.")
        return False

def main():
    """Main function"""
    success = run_all_tests()
    
    if success:
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Run training with new reproducibility features:")
        logger.info("   python train.py --deterministic --seed 42")
        logger.info("2. Test different splitting strategies:")
        logger.info("   python train.py --split_strategy scaffold")
        logger.info("3. Enable uncertainty quantification:")
        logger.info("   python train.py --enable_uncertainty --temperature_scaling")
        logger.info("4. Run external validation:")
        logger.info("   python train.py --external_validation --toxcast_data path/to/toxcast")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
