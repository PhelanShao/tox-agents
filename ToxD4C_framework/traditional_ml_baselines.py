#!/usr/bin/env python3
"""
A6: Traditional ML baselines for ToxD4C comparison
Simple interpretable models: Logistic Regression, Random Forest, XGBoost
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import lmdb

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MolecularFeatureExtractor:
    """Extract molecular descriptors for traditional ML."""
    
    def __init__(self):
        self.descriptor_names = []
        self.scaler = StandardScaler()
        
    def extract_rdkit_descriptors(self, smiles: str) -> Dict:
        """Extract RDKit molecular descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Basic descriptors
            descriptors = {}
            
            # Molecular properties
            descriptors['MW'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['HBD'] = Descriptors.NumHDonors(mol)
            descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
            descriptors['RotBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['SaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
            try:
                descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            except AttributeError:
                # Fallback for older RDKit versions
                descriptors['FractionCsp3'] = 0.0
            
            # Connectivity descriptors
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            
            # Electrotopological descriptors
            descriptors['EState_VSA1'] = Descriptors.EState_VSA1(mol)
            descriptors['EState_VSA2'] = Descriptors.EState_VSA2(mol)
            descriptors['EState_VSA3'] = Descriptors.EState_VSA3(mol)
            
            # ECFP fingerprints (selected bits)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            for i in range(1024):
                descriptors[f'ECFP_{i}'] = fp[i]
            
            return descriptors
            
        except Exception as e:
            logger.warning(f"Failed to extract descriptors for {smiles}: {e}")
            return None
    
    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Extract and scale features for training."""
        logger.info(f"Extracting features for {len(smiles_list)} molecules...")
        
        features_list = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            descriptors = self.extract_rdkit_descriptors(smiles)
            if descriptors is not None:
                features_list.append(list(descriptors.values()))
                valid_indices.append(i)
                if len(self.descriptor_names) == 0:
                    self.descriptor_names = list(descriptors.keys())
        
        if not features_list:
            raise ValueError("No valid molecular descriptors extracted")
        
        features = np.array(features_list)
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit scaler and transform
        features_scaled = self.scaler.fit_transform(features)
        
        logger.info(f"Extracted {features_scaled.shape[1]} features for {features_scaled.shape[0]} molecules")
        return features_scaled, valid_indices
    
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform new data using fitted scaler."""
        features_list = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            descriptors = self.extract_rdkit_descriptors(smiles)
            if descriptors is not None:
                features_list.append(list(descriptors.values()))
                valid_indices.append(i)
        
        if not features_list:
            return np.array([]), []
        
        features = np.array(features_list)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)
        
        return features_scaled, valid_indices

class TraditionalMLBaselines:
    """Traditional ML baselines for toxicity prediction."""
    
    def __init__(self):
        self.feature_extractor = MolecularFeatureExtractor()
        self.models = {}
        self.feature_selectors = {}
        
    def load_data(self, lmdb_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Load data from LMDB."""
        logger.info(f"Loading data from {lmdb_path}")
        
        subdir_flag = Path(lmdb_path).is_dir()
        env = lmdb.open(lmdb_path, subdir=subdir_flag, readonly=True, lock=False, readahead=False, meminit=False)
        
        smiles_list = []
        cls_targets = []
        reg_targets = []
        
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    key_str = key.decode('ascii')
                    if key_str.isdigit() or key_str in ['length', '__keys__']:
                        continue
                    
                    data = pickle.loads(value)
                    smiles_list.append(key_str)
                    
                    # Extract targets
                    if 'classification_target' in data:
                        cls_targets.append(data['classification_target'])
                    if 'regression_target' in data:
                        reg_targets.append(data['regression_target'])
                        
                except Exception as e:
                    continue
        
        env.close()
        
        cls_targets = np.array(cls_targets) if cls_targets else None
        reg_targets = np.array(reg_targets) if reg_targets else None
        
        logger.info(f"Loaded {len(smiles_list)} molecules")
        if cls_targets is not None:
            logger.info(f"Classification targets shape: {cls_targets.shape}")
        if reg_targets is not None:
            logger.info(f"Regression targets shape: {reg_targets.shape}")
        
        return smiles_list, cls_targets, reg_targets
    
    def train_classification_models(self, X: np.ndarray, y: np.ndarray, endpoint_idx: int = 0):
        """Train classification models for a specific endpoint."""
        logger.info(f"Training classification models for endpoint {endpoint_idx}")
        
        # Filter valid samples
        valid_mask = y[:, endpoint_idx] != -10000
        X_valid = X[valid_mask]
        y_valid = y[valid_mask, endpoint_idx]
        
        if len(np.unique(y_valid)) < 2:
            logger.warning(f"Endpoint {endpoint_idx} has insufficient classes")
            return None
        
        logger.info(f"Valid samples: {len(y_valid)}, Positive rate: {np.mean(y_valid):.3f}")
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(500, X_valid.shape[1]))
        X_selected = selector.fit_transform(X_valid, y_valid)
        self.feature_selectors[f'cls_{endpoint_idx}'] = selector
        
        logger.info(f"Selected {X_selected.shape[1]} features")
        
        # Define models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Hyperparameter grids
        param_grids = {
            'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100]},
            'RandomForest': {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
            'XGBoost': {'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_selected, y_valid)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_selected, y_valid, cv=5, scoring='roc_auc')
            
            results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_auc_mean': np.mean(cv_scores),
                'cv_auc_std': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"{model_name} - CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        self.models[f'cls_{endpoint_idx}'] = results
        return results
    
    def train_regression_models(self, X: np.ndarray, y: np.ndarray, endpoint_idx: int = 0):
        """Train regression models for a specific endpoint."""
        logger.info(f"Training regression models for endpoint {endpoint_idx}")
        
        # Filter valid samples
        valid_mask = y[:, endpoint_idx] != -10000.0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask, endpoint_idx]
        
        if len(y_valid) < 10:
            logger.warning(f"Endpoint {endpoint_idx} has insufficient samples")
            return None
        
        logger.info(f"Valid samples: {len(y_valid)}, Mean: {np.mean(y_valid):.3f}, Std: {np.std(y_valid):.3f}")
        
        # Feature selection
        selector = SelectKBest(f_regression, k=min(500, X_valid.shape[1]))
        X_selected = selector.fit_transform(X_valid, y_valid)
        self.feature_selectors[f'reg_{endpoint_idx}'] = selector
        
        # Define models
        models = {
            'Ridge': Ridge(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBRegressor(random_state=42)
        }
        
        # Hyperparameter grids
        param_grids = {
            'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
            'RandomForest': {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
            'XGBoost': {'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_selected, y_valid)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_selected, y_valid, cv=5, scoring='r2')
            
            results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_r2_mean': np.mean(cv_scores),
                'cv_r2_std': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"{model_name} - CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        self.models[f'reg_{endpoint_idx}'] = results
        return results

def main():
    """Main function for traditional ML baselines."""
    logger.info("🚀 A6: Traditional ML Baselines for ToxD4C")
    logger.info("="*60)
    
    # Initialize baseline trainer
    baseline_trainer = TraditionalMLBaselines()
    
    # Load training data
    train_smiles, train_cls, train_reg = baseline_trainer.load_data("data/data/processed/train.lmdb")
    
    # Extract features
    X_train, valid_indices = baseline_trainer.feature_extractor.fit_transform(train_smiles)
    
    # Filter targets to match valid features
    if train_cls is not None:
        train_cls = train_cls[valid_indices]
    if train_reg is not None:
        train_reg = train_reg[valid_indices]
    
    # Train models for key endpoints
    all_results = {}
    
    # Classification endpoints (select high-coverage ones)
    if train_cls is not None:
        high_coverage_cls = [1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Based on previous analysis
        
        for endpoint_idx in high_coverage_cls[:3]:  # Train first 3 for demo
            results = baseline_trainer.train_classification_models(X_train, train_cls, endpoint_idx)
            if results:
                all_results[f'classification_endpoint_{endpoint_idx}'] = results
    
    # Regression endpoints
    if train_reg is not None:
        for endpoint_idx in range(min(3, train_reg.shape[1])):  # Train first 3 for demo
            results = baseline_trainer.train_regression_models(X_train, train_reg, endpoint_idx)
            if results:
                all_results[f'regression_endpoint_{endpoint_idx}'] = results
    
    # Save results
    output_dir = Path("traditional_ml_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    with open(output_dir / "baseline_models.pkl", 'wb') as f:
        pickle.dump(baseline_trainer.models, f)
    
    # Save feature extractor
    with open(output_dir / "feature_extractor.pkl", 'wb') as f:
        pickle.dump(baseline_trainer.feature_extractor, f)
    
    # Save results summary
    results_summary = {
        'experiment_type': 'traditional_ml_baselines',
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0],
        'results': all_results
    }
    
    with open(output_dir / "baseline_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved to: {output_dir}")
    
    # Print summary
    logger.info("\n📊 TRADITIONAL ML BASELINES SUMMARY")
    logger.info("="*60)
    
    for endpoint_name, endpoint_results in all_results.items():
        logger.info(f"\n🎯 {endpoint_name}:")
        
        for model_name, model_results in endpoint_results.items():
            if 'cv_auc_mean' in model_results:
                logger.info(f"   {model_name}: AUC = {model_results['cv_auc_mean']:.4f} ± {model_results['cv_auc_std']:.4f}")
            elif 'cv_r2_mean' in model_results:
                logger.info(f"   {model_name}: R² = {model_results['cv_r2_mean']:.4f} ± {model_results['cv_r2_std']:.4f}")
    
    logger.info("\n🎉 Traditional ML baselines completed!")
    logger.info("📝 Next: Compare with ToxD4C results for A6 analysis")

if __name__ == "__main__":
    main()
