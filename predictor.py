import os
import glob
import numpy as np
import pandas as pd
import torch
import logging
from unimol_tools import MolPredict

# Configure logging
logger = logging.getLogger(__name__)

class BinaryPredictor:
    def __init__(self, model_base_dir):
        """
        Initialize binary predictor with trained model directory
        Args:
            model_base_dir (str): Base directory containing exp_* subdirectories with trained models
        """
        if not os.path.exists(model_base_dir):
            raise FileNotFoundError(f"Model directory not found: {model_base_dir}")
            
        self.model_base_dir = model_base_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing binary predictor with base directory: {model_base_dir}")
        
        # Find all exp_* directories
        self.exp_dirs = [
            d for d in glob.glob(os.path.join(model_base_dir, "exp_*"))
            if os.path.isdir(d)
        ]
        
        if not self.exp_dirs:
            raise ValueError(f"No exp_* directories found in {model_base_dir}")
        
        logger.info(f"Found {len(self.exp_dirs)} experiment directories")
        
        # Collect all model paths
        self.model_paths = []
        for exp_dir in self.exp_dirs:
            # Check if directory contains model files
            model_files = glob.glob(os.path.join(exp_dir, "model_*.pth"))
            if model_files:
                self.model_paths.append(exp_dir)
                logger.info(f"Found {len(model_files)} model files in {os.path.basename(exp_dir)}")
            
        if not self.model_paths:
            raise ValueError("No valid model directories found")
        
        logger.info(f"Total number of model directories to use: {len(self.model_paths)}")

    def prepare_data(self, data_path):
        """
        Prepare input data for prediction
        Args:
            data_path (str): Path to NPZ file containing molecular coordinates
        """
        logger.info(f"Loading prediction data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        if 'coord' not in data or 'symbol' not in data:
            raise ValueError("Input data must contain 'coord' and 'symbol' arrays")
        
        # Extract coordinates and atoms
        coordinates = [np.array(coord) for coord in data['coord']]
        atoms = [list(symbols) for symbols in data['symbol']]
        
        logger.info(f"Loaded {len(coordinates)} molecules for prediction")
        
        # Create dummy targets with both classes (0 and 1)
        num_samples = len(coordinates)
        dummy_targets = np.zeros(num_samples)
        # Set some targets to 1 to ensure both classes are present
        if num_samples > 1:
            dummy_targets[0] = 1
        
        prepared_data = {
            'coordinates': coordinates,
            'atoms': atoms,
            'target': dummy_targets.tolist()
        }
        
        return prepared_data, data

    def predict(self, data_path, output_path=None):
        """Generate binary predictions using ensemble of models"""
        if not output_path:
            output_path = 'binary_predictions.csv'
        
        # Prepare input data
        test_data, input_data = self.prepare_data(data_path)
        
        # Generate predictions from each model directory
        all_predictions = []
        
        for model_dir in self.model_paths:
            try:
                logger.info(f"Processing models in: {os.path.basename(model_dir)}")
                
                # Initialize predictor with this experiment's directory
                pred_clf = MolPredict(load_model=model_dir)
                
                # Generate predictions
                predictions = pred_clf.predict(test_data)
                # Ensure predictions are 1-dimensional
                if predictions.ndim > 1:
                    predictions = predictions.ravel()
                all_predictions.append(predictions)
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in model directory {model_dir}: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("No successful predictions from any model!")
        
        # Convert predictions to numpy arrays and ensure they're 1D
        all_predictions = [np.array(pred).ravel() for pred in all_predictions]
        
        # Check if all predictions have the same shape
        shape = all_predictions[0].shape
        if not all(pred.shape == shape for pred in all_predictions):
            raise ValueError("Predictions from different models have inconsistent shapes")
        
        # Average predictions across all models
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Ensure final predictions are 1D
        ensemble_predictions = ensemble_predictions.ravel()
        
        # Convert to binary predictions
        binary_predictions = (ensemble_predictions > 0.5).astype(int)
        
        # Create predictions DataFrame with 1D arrays
        predictions_df = pd.DataFrame({
            'probability': ensemble_predictions.tolist(),
            'prediction': binary_predictions.tolist()
        })
        
        # Add sample IDs if available
        if 'id' in input_data:
            predictions_df.insert(0, 'id', input_data['id'])
        
        # Save predictions
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        return predictions_df
