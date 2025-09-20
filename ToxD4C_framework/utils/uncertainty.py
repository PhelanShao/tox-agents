#!/usr/bin/env python3
"""
Uncertainty quantification utilities for ToxD4C
Addresses A2 reviewer concerns about uncertainty and applicability domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import warnings

logger = logging.getLogger(__name__)

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrating neural network predictions.
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)
    
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def calibrate(self, val_loader, device='cpu', max_iter=50):
        """
        Calibrate temperature using validation set.
        
        Args:
            val_loader: Validation data loader
            device: Device to run calibration on
            max_iter: Maximum optimization iterations
        """
        self.model.eval()
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                data = {k: v.to(device) for k, v in batch.items() if k != 'smiles'}
                outputs = self.model(data, batch['smiles'])
                
                # Handle classification outputs
                if 'classification' in outputs['predictions']:
                    logits = outputs['predictions']['classification']
                    labels = batch['classification_labels'].to(device)
                    mask = batch['classification_mask'].to(device)
                    
                    # Apply mask
                    valid_logits = logits[mask]
                    valid_labels = labels[mask]
                    
                    logits_list.append(valid_logits.cpu())
                    labels_list.append(valid_labels.cpu())
        
        if not logits_list:
            logger.warning("No validation data for temperature calibration")
            return
        
        logits = torch.cat(logits_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.temperature_scale(logits), labels.long())
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        logger.info(f"Temperature calibration completed. Optimal temperature: {self.temperature.item():.4f}")

class DeepEnsemble:
    """
    Deep ensemble for uncertainty quantification.
    
    Reference: Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty 
    Estimation using Deep Ensembles" (NIPS 2017)
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.n_models = len(models)
        
    def predict(self, data_loader, device='cpu') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Make ensemble predictions with uncertainty estimates.
        
        Args:
            data_loader: Data loader for predictions
            device: Device to run predictions on
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        all_predictions = {
            'classification': [],
            'regression': []
        }
        
        # Collect predictions from all models
        for model in self.models:
            model.eval()
            model_preds = {'classification': [], 'regression': []}
            
            with torch.no_grad():
                for batch in data_loader:
                    data = {k: v.to(device) for k, v in batch.items() if k != 'smiles'}
                    outputs = model(data, batch['smiles'])
                    
                    if 'classification' in outputs['predictions']:
                        cls_probs = torch.sigmoid(outputs['predictions']['classification'])
                        model_preds['classification'].append(cls_probs.cpu().numpy())
                    
                    if 'regression' in outputs['predictions']:
                        reg_preds = outputs['predictions']['regression']
                        model_preds['regression'].append(reg_preds.cpu().numpy())
            
            # Concatenate batch predictions
            for task_type in ['classification', 'regression']:
                if model_preds[task_type]:
                    model_preds[task_type] = np.concatenate(model_preds[task_type], axis=0)
                    all_predictions[task_type].append(model_preds[task_type])
        
        # Calculate ensemble statistics
        results = {}
        
        for task_type in ['classification', 'regression']:
            if all_predictions[task_type]:
                # Stack predictions from all models
                stacked_preds = np.stack(all_predictions[task_type], axis=0)  # (n_models, n_samples, n_tasks)
                
                # Calculate mean and uncertainty
                mean_pred = np.mean(stacked_preds, axis=0)
                std_pred = np.std(stacked_preds, axis=0)
                
                # For classification, also calculate predictive entropy
                if task_type == 'classification':
                    # Predictive entropy: H[y|x] = -sum(p_mean * log(p_mean))
                    epsilon = 1e-8  # For numerical stability
                    pred_entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon) + 
                                         (1 - mean_pred) * np.log(1 - mean_pred + epsilon), axis=-1)
                    
                    # Mutual information: I[y,θ|x] = H[y|x] - E[H[y|θ,x]]
                    model_entropies = []
                    for model_pred in stacked_preds:
                        model_entropy = -np.sum(model_pred * np.log(model_pred + epsilon) + 
                                              (1 - model_pred) * np.log(1 - model_pred + epsilon), axis=-1)
                        model_entropies.append(model_entropy)
                    
                    expected_entropy = np.mean(model_entropies, axis=0)
                    mutual_info = pred_entropy - expected_entropy
                    
                    results[task_type] = {
                        'mean': mean_pred,
                        'std': std_pred,
                        'predictive_entropy': pred_entropy,
                        'mutual_information': mutual_info,
                        'individual_predictions': stacked_preds
                    }
                else:
                    results[task_type] = {
                        'mean': mean_pred,
                        'std': std_pred,
                        'individual_predictions': stacked_preds
                    }
        
        return results

class ApplicabilityDomain:
    """
    Applicability domain analysis for molecular predictions.
    """
    
    def __init__(self, training_smiles: List[str], 
                 similarity_threshold: float = 0.3,
                 fingerprint_type: str = 'morgan'):
        """
        Initialize applicability domain analyzer.
        
        Args:
            training_smiles: SMILES from training set
            similarity_threshold: Minimum similarity threshold
            fingerprint_type: Type of fingerprint to use
        """
        self.training_smiles = training_smiles
        self.similarity_threshold = similarity_threshold
        self.fingerprint_type = fingerprint_type
        
        # Generate training fingerprints
        self.training_fps = self._generate_fingerprints(training_smiles)
        logger.info(f"Generated {len(self.training_fps)} training fingerprints")
        
        # Fit Mahalanobis distance model (if enough data)
        self.mahalanobis_model = None
        if len(self.training_fps) > 100:  # Need sufficient data
            self._fit_mahalanobis_model()
    
    def _generate_fingerprints(self, smiles_list: List[str]) -> List[np.ndarray]:
        """Generate molecular fingerprints."""
        fingerprints = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                if self.fingerprint_type == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                elif self.fingerprint_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol)
                elif self.fingerprint_type == 'maccs':
                    fp = AllChem.GetMACCSKeysFingerprint(mol)
                else:
                    raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")
                
                fingerprints.append(np.array(fp))
                
            except Exception as e:
                logger.warning(f"Error generating fingerprint for {smiles}: {e}")
                continue
        
        return fingerprints
    
    def _fit_mahalanobis_model(self):
        """Fit Mahalanobis distance model on training fingerprints."""
        try:
            fp_matrix = np.array(self.training_fps)
            
            # Use empirical covariance
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(fp_matrix)
            
            self.mahalanobis_model = {
                'mean': np.mean(fp_matrix, axis=0),
                'inv_cov': cov_estimator.precision_,
                'threshold': None  # Will be set based on training data
            }
            
            # Calculate threshold based on training data (e.g., 95th percentile)
            train_distances = []
            for fp in self.training_fps:
                dist = mahalanobis(fp, self.mahalanobis_model['mean'], 
                                 self.mahalanobis_model['inv_cov'])
                train_distances.append(dist)
            
            self.mahalanobis_model['threshold'] = np.percentile(train_distances, 95)
            
            logger.info("Mahalanobis distance model fitted successfully")
            
        except Exception as e:
            logger.warning(f"Failed to fit Mahalanobis model: {e}")
            self.mahalanobis_model = None
    
    def assess_applicability(self, query_smiles: List[str]) -> Dict[str, np.ndarray]:
        """
        Assess applicability domain for query molecules.
        
        Args:
            query_smiles: SMILES to assess
            
        Returns:
            Dictionary with applicability metrics
        """
        query_fps = self._generate_fingerprints(query_smiles)
        
        results = {
            'max_similarity': [],
            'mean_similarity': [],
            'in_domain_similarity': [],
            'mahalanobis_distance': [],
            'in_domain_mahalanobis': []
        }
        
        for i, query_fp in enumerate(query_fps):
            # Calculate Tanimoto similarities
            similarities = []
            for train_fp in self.training_fps:
                if self.fingerprint_type in ['morgan', 'rdkit', 'maccs']:
                    # For bit vectors - convert numpy arrays back to RDKit fingerprints
                    query_mol = Chem.MolFromSmiles(query_smiles[i])
                    if query_mol is None:
                        similarities.append(0.0)
                        continue

                    # Generate fresh fingerprints for comparison
                    if self.fingerprint_type == 'morgan':
                        query_rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
                    elif self.fingerprint_type == 'rdkit':
                        query_rdkit_fp = Chem.RDKFingerprint(query_mol)
                    elif self.fingerprint_type == 'maccs':
                        query_rdkit_fp = AllChem.GetMACCSKeysFingerprint(query_mol)

                    # Find corresponding training molecule and generate its fingerprint
                    # For now, use numpy-based similarity as approximation
                    sim = np.sum(query_fp & train_fp) / np.sum(query_fp | train_fp) if np.sum(query_fp | train_fp) > 0 else 0.0
                else:
                    # For count vectors
                    sim = np.dot(query_fp, train_fp) / (np.linalg.norm(query_fp) * np.linalg.norm(train_fp))

                similarities.append(sim)
            
            similarities = np.array(similarities)
            
            results['max_similarity'].append(np.max(similarities))
            results['mean_similarity'].append(np.mean(similarities))
            results['in_domain_similarity'].append(np.max(similarities) >= self.similarity_threshold)
            
            # Mahalanobis distance
            if self.mahalanobis_model is not None:
                try:
                    dist = mahalanobis(query_fp, self.mahalanobis_model['mean'], 
                                     self.mahalanobis_model['inv_cov'])
                    results['mahalanobis_distance'].append(dist)
                    results['in_domain_mahalanobis'].append(dist <= self.mahalanobis_model['threshold'])
                except:
                    results['mahalanobis_distance'].append(np.nan)
                    results['in_domain_mahalanobis'].append(False)
            else:
                results['mahalanobis_distance'].append(np.nan)
                results['in_domain_mahalanobis'].append(False)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results

class ConformalPrediction:
    """
    Conformal prediction for uncertainty quantification.
    
    Reference: Vovk et al. "Algorithmic Learning in a Random World" (2005)
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is the confidence level)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
    
    def calibrate(self, predictions: np.ndarray, targets: np.ndarray, 
                 task_type: str = 'classification'):
        """
        Calibrate conformal predictor using calibration set.
        
        Args:
            predictions: Model predictions on calibration set
            targets: True targets for calibration set
            task_type: 'classification' or 'regression'
        """
        if task_type == 'classification':
            # For classification, use 1 - max(predicted probabilities) as nonconformity score
            self.calibration_scores = 1 - np.max(predictions, axis=1)
        else:
            # For regression, use absolute residuals
            self.calibration_scores = np.abs(predictions.flatten() - targets.flatten())
        
        # Calculate quantile
        n = len(self.calibration_scores)
        self.quantile = np.quantile(self.calibration_scores, 
                                   (1 - self.alpha) * (n + 1) / n)
        
        logger.info(f"Conformal prediction calibrated with quantile: {self.quantile:.4f}")
    
    def predict(self, predictions: np.ndarray, 
               task_type: str = 'classification') -> Dict[str, np.ndarray]:
        """
        Make conformal predictions.
        
        Args:
            predictions: Model predictions
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with prediction sets/intervals
        """
        if self.quantile is None:
            raise ValueError("Conformal predictor not calibrated")
        
        if task_type == 'classification':
            # Prediction sets: include all classes with score <= quantile
            nonconformity_scores = 1 - predictions
            prediction_sets = nonconformity_scores <= self.quantile
            
            return {
                'prediction_sets': prediction_sets,
                'set_sizes': np.sum(prediction_sets, axis=1),
                'nonconformity_scores': nonconformity_scores
            }
        else:
            # Prediction intervals
            lower_bound = predictions.flatten() - self.quantile
            upper_bound = predictions.flatten() + self.quantile
            interval_width = upper_bound - lower_bound
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'point_prediction': predictions.flatten()
            }

# Example usage
if __name__ == "__main__":
    # Test uncertainty quantification
    
    # Example training SMILES
    training_smiles = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)O",
        "CCCCO", "CC(C)(C)O", "c1ccc(O)cc1", "CCN", "CCC(=O)O"
    ]
    
    # Test applicability domain
    ad = ApplicabilityDomain(training_smiles)
    
    query_smiles = ["CCCO", "c1ccc(N)cc1", "CCCCCCCCCCCC"]  # Similar, similar, dissimilar
    applicability = ad.assess_applicability(query_smiles)
    
    print("Applicability Domain Assessment:")
    for i, smiles in enumerate(query_smiles):
        print(f"  {smiles}:")
        print(f"    Max similarity: {applicability['max_similarity'][i]:.3f}")
        print(f"    In domain: {applicability['in_domain_similarity'][i]}")
    
    # Test conformal prediction
    cp = ConformalPrediction(alpha=0.1)
    
    # Dummy calibration data
    cal_preds = np.random.rand(100, 2)  # 100 samples, 2 classes
    cal_targets = np.random.randint(0, 2, 100)
    
    cp.calibrate(cal_preds, cal_targets, task_type='classification')
    
    # Test predictions
    test_preds = np.random.rand(10, 2)
    conformal_results = cp.predict(test_preds, task_type='classification')
    
    print(f"\nConformal Prediction Results:")
    print(f"  Average set size: {np.mean(conformal_results['set_sizes']):.2f}")
    print(f"  Coverage guarantee: {1-cp.alpha:.1%}")
