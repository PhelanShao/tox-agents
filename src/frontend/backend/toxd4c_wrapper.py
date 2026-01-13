"""
ToxD4C Wrapper Module

This module provides wrapper functions for ToxD4C toxicity predictions.
It bridges the frontend backend API with the ToxD4C model.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Add ToxD4C_framework to path
TOXD4C_PATH = Path(__file__).parent.parent.parent.parent / "ToxD4C_framework"
if TOXD4C_PATH.exists():
    sys.path.insert(0, str(TOXD4C_PATH))

# Global predictor instance (singleton pattern)
_predictor = None


def get_toxd4c_wrapper():
    """Get or create ToxD4C predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ToxD4CWrapper()
    return _predictor


class ToxD4CWrapper:
    """Wrapper class for ToxD4C model predictions."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize ToxD4C wrapper.
        
        Args:
            model_path: Path to trained model checkpoint. If None, uses default path.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.device = device
        self.predictor = None
        self.model_path = model_path
        
        try:
            from inference_toxd4c import ToxD4CPredictor
            from configs.toxd4c_config import get_enhanced_toxd4c_config
            
            config = get_enhanced_toxd4c_config()
            
            # Find model path
            if model_path is None:
                # Try common locations
                possible_paths = [
                    TOXD4C_PATH / "checkpoints" / "best_model.pth",
                    TOXD4C_PATH / "checkpoints" / "toxd4c_model.pth",
                    Path("./checkpoints/best_model.pth"),
                    Path("./models/toxd4c_model.pth"),
                ]
                for p in possible_paths:
                    if p.exists():
                        model_path = str(p)
                        break
            
            if model_path and Path(model_path).exists():
                self.predictor = ToxD4CPredictor(model_path, config, device)
                logger.info(f"ToxD4C model loaded from {model_path}")
            else:
                logger.warning("ToxD4C model not found. Predictions will use placeholder values.")
                
        except ImportError as e:
            logger.warning(f"Could not import ToxD4C modules: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize ToxD4C: {e}")
    
    def predict(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Predict toxicity for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            Dictionary with prediction results.
        """
        if self.predictor is None:
            return self._placeholder_predictions(smiles_list)
        
        try:
            results_df, interpretations = self.predictor.predict_from_smiles(smiles_list)
            return {
                'success': True,
                'num_molecules': len(smiles_list),
                'predictions': results_df.to_dict(orient='records'),
                'interpretations': interpretations
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'num_molecules': 0,
                'predictions': []
            }
    
    def _placeholder_predictions(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Return placeholder predictions when model is not available."""
        from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS
        
        predictions = []
        for smiles in smiles_list:
            pred = {'smiles': smiles}
            for task in CLASSIFICATION_TASKS:
                pred[f'{task}_prob'] = 0.5
                pred[f'{task}_pred'] = 'Unknown'
            for task in REGRESSION_TASKS:
                pred[task] = 0.0
            predictions.append(pred)
        
        return {
            'success': True,
            'num_molecules': len(smiles_list),
            'predictions': predictions,
            'interpretations': [],
            'warning': 'Model not loaded. Using placeholder values.'
        }


def predict_toxicity_from_smiles(smiles_input: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Predict toxicity from SMILES string(s).
    
    Args:
        smiles_input: Single SMILES string or list of SMILES strings.
        
    Returns:
        Dictionary with prediction results.
    """
    wrapper = get_toxd4c_wrapper()
    
    if isinstance(smiles_input, str):
        smiles_list = [s.strip() for s in smiles_input.strip().split('\n') if s.strip()]
    else:
        smiles_list = smiles_input
    
    return wrapper.predict(smiles_list)


def predict_toxicity_from_xyz(xyz_file: str) -> Dict[str, Any]:
    """
    Predict toxicity from XYZ file.
    
    Args:
        xyz_file: Path to XYZ file.
        
    Returns:
        Dictionary with prediction results.
    """
    wrapper = get_toxd4c_wrapper()
    
    if wrapper.predictor is None:
        # Return placeholder for XYZ predictions
        return {
            'success': True,
            'num_molecules': 1,
            'predictions': [],
            'interpretations': [],
            'warning': 'XYZ prediction requires model to be loaded.'
        }
    
    try:
        results_df, interpretations = wrapper.predictor.predict_from_xyz(xyz_file)
        return {
            'success': True,
            'num_molecules': len(results_df),
            'predictions': results_df.to_dict(orient='records'),
            'interpretations': interpretations
        }
    except Exception as e:
        logger.error(f"XYZ prediction error: {e}")
        return {
            'success': False,
            'error': str(e),
            'num_molecules': 0,
            'predictions': []
        }


if __name__ == '__main__':
    # Test the wrapper
    result = predict_toxicity_from_smiles("CCO")
    print(f"Test result: {result}")
