# 该代码用于实现分子属性预测并返回csv列表
import os
import numpy as np
import pandas as pd
import glob
import logging
import gradio as gr
from datetime import datetime
import torch
from sklearn.preprocessing import StandardScaler
from unimol_tools import MolPredict
import yaml
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MoleculePredictor:
    def __init__(self):
        self.output_dir = "prediction_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.EXCLUDE_COLUMNS = ['symbol', 'coord', 'id', 'SampleName']
        
    def load_config(self, model_dir):
        """Load model configuration"""
        config_path = os.path.join(model_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    def apply_tta(self, data, pred_clf, n_augments=5):
        """Apply test-time augmentation"""
        augmented_predictions = []
        for _ in range(n_augments):
            data_aug = {
                'coordinates': [
                    coord + np.random.randn(*coord.shape).clip(-3, 3) * 0.0025 
                    for coord in data['coordinates']
                ],
                'atoms': data['atoms'],
                'target': data['target']
            }
            augmented_predictions.append(pred_clf.predict(data_aug))
        return np.mean(augmented_predictions, axis=0)

    def predict(self, input_file, reference_file, model_dir):
        """Run prediction process"""
        try:
            # Load reference data for standardization
            logger.info(f"Loading reference data: {reference_file}")
            reference_data = np.load(reference_file, allow_pickle=True)
            
            # Get target columns from reference data
            target_columns = [col for col in reference_data.files 
                            if col not in self.EXCLUDE_COLUMNS 
                            and np.issubdtype(reference_data[col].dtype, np.number)]
            logger.info(f"Target features: {target_columns}")
            
            # Initialize scaler with reference data
            targets_dict = {key: reference_data[key] for key in target_columns}
            targets_df = pd.DataFrame(targets_dict)
            scaler = StandardScaler()
            scaler.fit(targets_df)
            
            # Load input data
            logger.info(f"Loading input data: {input_file}")
            input_data = np.load(input_file, allow_pickle=True)
            
            # Prepare data for prediction
            prepared_data = {
                'coordinates': [np.array(coord) for coord in input_data['coord']],
                'atoms': [list(symbols) for symbols in input_data['symbol']],
                'target': np.zeros((len(input_data['coord']), len(target_columns))).tolist()
            }
            
            # Find model directories
            model_dirs = sorted([d for d in glob.glob(os.path.join(model_dir, "model_*"))
                               if os.path.isdir(d)])
            
            if not model_dirs:
                raise ValueError(f"No valid model directories found in {model_dir}")
            
            all_predictions = []
            for model_path in model_dirs:
                logger.info(f"Using model: {model_path}")
                try:
                    # Initialize predictor
                    pred_clf = MolPredict(load_model=model_path)
                    
                    # Predict with TTA
                    predictions = self.apply_tta(prepared_data, pred_clf)
                    if len(predictions.shape) == 1:
                        predictions = predictions.reshape(-1, 1)
                    
                    all_predictions.append(predictions)
                    logger.info(f"Model {model_path} prediction successful")
                except Exception as e:
                    logger.error(f"Model {model_path} prediction failed: {str(e)}")
                    continue
            
            if not all_predictions:
                raise ValueError("No models completed prediction successfully")
            
            # Ensemble predictions
            ensemble_pred = np.mean(all_predictions, axis=0)
            
            # Inverse transform predictions
            ensemble_pred_orig = scaler.inverse_transform(ensemble_pred)
            
            # Create results DataFrame
            results_df = pd.DataFrame(ensemble_pred_orig, columns=target_columns)
            if 'id' in input_data:
                results_df.insert(0, 'id', input_data['id'])
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f'predictions_{timestamp}.csv')
            results_df.to_csv(output_file, index=False)
            
            logger.info(f"Predictions saved to: {output_file}")
            return output_file, f"Prediction completed successfully! Results saved to {output_file}"
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

def find_model_dirs():
    """Find available model directories"""
    current_dir = os.getcwd()
    model_dirs = []
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and (item.startswith("21stmodel_") or item.startswith("quantum_model")):
            model_dirs.append(item)
    return model_dirs

def create_interface():
    predictor = MoleculePredictor()
    
    def run_prediction(input_temp, model_path, reference_path):
        try:
            if input_temp is None:
                return None, "Please upload an input file."
            if not os.path.exists(model_path):
                return None, f"Model directory not found: {model_path}"
            if not os.path.exists(reference_path):
                return None, f"Reference file not found: {reference_path}"
                
            # Save temporary input file
            input_file = f"temp_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            with open(input_file, 'wb') as f:
                f.write(input_temp)
            
            try:
                output_file, message = predictor.predict(input_file, reference_path, model_path)
                if output_file and os.path.exists(output_file):
                    return output_file, message
                return None, message
            finally:
                # Cleanup temporary file
                if os.path.exists(input_file):
                    os.remove(input_file)
        except Exception as e:
            return None, f"Error during prediction: {str(e)}"
    
    with gr.Blocks(title="Molecule Property Prediction") as interface:
        gr.Markdown("# Molecule Property Prediction Interface")
        
        with gr.Row():
            input_file = gr.File(
                label="Upload Input NPZ File",
                file_types=[".npz"],
                type="binary"
            )
        
        with gr.Row():
            model_path = gr.Textbox(
                label="Model Directory Path",
                placeholder="/path/to/model/directory"
            )
            
        with gr.Row():
            reference_path = gr.Textbox(
                label="Reference NPZ File Path",
                placeholder="/path/to/reference.npz"
            )
        
        run_btn = gr.Button("Run Prediction")
        output_file = gr.File(label="Download Predictions")
        message = gr.Textbox(label="Status Message")
        
        run_btn.click(
            fn=run_prediction,
            inputs=[input_file, model_path, reference_path],
            outputs=[output_file, message]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False  # Set to True if you want to create a public link
    )