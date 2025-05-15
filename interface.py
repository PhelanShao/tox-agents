import os
import io
import json
import logging
import gradio as gr
import tempfile
import shutil
import zipfile
import numpy as np
import pandas as pd
from chatbot import ChatInterface
from predictor import BinaryPredictor
from MoleculePredictor import MoleculePredictor
from reactor import (
    handle_uploaded_files, run_extra,
    run_net, on_timestep_select
)
from file_converter import read_xyz_file, save_npz
from converter import NPZToXYZ
from visualizer import (
    display_molecule_pymol, export_current_image
)
from probability_plot import create_probability_plot

log_stream = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(log_stream),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedInterface:
    def __init__(self):
        self.chat_interface = ChatInterface()
        
    def process_export_for_chat(self, export_path, export_data):
        if not export_path or not export_data:
            return None, "Export failed"
            
        try:
            with open(export_data.name, 'r') as f:
                json_data = json.load(f)
            
            json_prompt = ("You are an expert structural chemist and biologist. You have been provided with data about a molecule, "
                         "including its predicted toxicity probability (where 'probability' of 0 indicates non-toxic and 1 indicates toxic) "
                         "and other property predictions (under 'property_prediction'). Based on your profound knowledge of chemistry and biology, "
                         "please write a detailed analysis report of at least 500 words. Directly address the significance of the toxicity probability, "
                         "discuss the other predicted properties, and explain their potential implications for the molecule's behavior, safety, or applications. Provide your analysis in a direct and continuous format." +
                         json.dumps(json_data, ensure_ascii=False))
            
            with open(export_path.name, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            message_content = [
                {"type": "text", "text": json_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
            
            self.chat_interface.memory.add_message("user", message_content, has_image=True)
            
            history, error = self.chat_interface.process_message(
                json_prompt,
                export_path.name,
                True,
                self.chat_interface.memory.get_display_history(),
                "anthropic/claude-3.5-sonnet"
            )
            
            return history, error
            
        except Exception as e:
            return None, f"Error processing export: {str(e)}"

def convert_xyz_to_npz(xyz_file):
    if not xyz_file:
        return "Error: No file uploaded", None
        
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(xyz_file.name, 'rb') as f:
                content = f.read()
            
            temp_xyz = os.path.join(temp_dir, "temp.xyz")
            with open(temp_xyz, 'wb') as f:
                f.write(content)
            
            data_dict, invalid_frames = read_xyz_file(temp_xyz)
            
            output_file = os.path.join(temp_dir, "converted_sequential.npz")
            
            info = save_npz(data_dict, output_file)
            
            summary = [
                "\nProcessing Report:",
                "-" * 50,
                f"Total frames processed: {len(data_dict['id'])}",
                f"Invalid frames: {len(invalid_frames)}",
                "-" * 50
            ]
            
            final_output = os.path.join(os.getcwd(), "converted_sequential.npz")
            shutil.copy2(output_file, final_output)
            
            return "\n".join([info] + summary), final_output
            
    except Exception as e:
        return f"Error during conversion: {str(e)}", None

def convert_npz_to_xyz(npz_file):
    if not npz_file:
        return "Error: No file uploaded", None
        
    try:
        output_file = os.path.splitext(npz_file.name)[0] + "_converted.xyz"
        
        converter = NPZToXYZ(npz_file.name)
        result = converter.convert(output_file)
        
        return result, output_file
    except Exception as e:
        return f"Error during conversion: {str(e)}", None

def process_molecule_visualization(file, frame_index, representation, rotation_x, rotation_y, rotation_z, zoom):
    if file is None:
        return None, "", gr.Slider(visible=False), None
    
    img, legend, total_frames = display_molecule_pymol(
        file_path=file.name,
        frame_index=frame_index,
        representation=representation,
        rotations=[rotation_x, rotation_y, rotation_z],
        zoom=zoom
    )
    
    if total_frames > 0:
        return img, legend, gr.Slider(minimum=0, maximum=total_frames-1, step=1, visible=True), img
    else:
        return img, legend, gr.Slider(visible=False), img

def process_binary_prediction(file, model_path="/mnt/backup2/ai4s/unimolpy/unimol7300"):
    if not file:
        return None, "Error: No file uploaded"
    
    try:
        predictor = BinaryPredictor(model_base_dir=model_path)
        output_path = file.name.replace('.npz', '_predictions.csv')
        predictions = predictor.predict(file.name, output_path)
        
        probabilities = predictions['probability'].values
        if np.max(probabilities) > 1 or np.min(probabilities) < 0:
            probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))
            predictions['probability'] = probabilities
            predictions['prediction'] = (probabilities > 0.5).astype(int)
            predictions.to_csv(output_path, index=False)
        
        log_messages = [
            f"Predictions: +{(predictions['prediction'] == 1).sum()}, -{(predictions['prediction'] == 0).sum()}",
            f"Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}"
        ]
        
        return output_path, "\n".join(log_messages)
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_property_prediction(file, model_path, reference_path):
    if not file or not model_path or not reference_path:
        return None, "Error: Missing inputs"
    try:
        predictor = MoleculePredictor()
        output_file, message = predictor.predict(file.name, reference_path, model_path)
        return output_file, message
    except Exception as e:
        return None, f"Error: {str(e)}"

def export_frame_data(current_image, export_format, frame_index, binary_pred_file, property_pred_file):
    try:
        temp_dir = tempfile.mkdtemp()
        
        image_path = export_current_image(current_image, export_format)
        
        data = {}
        
        if binary_pred_file and os.path.exists(binary_pred_file):
            binary_df = pd.read_csv(binary_pred_file)
            if len(binary_df) > frame_index:
                row = binary_df.iloc[frame_index]
                binary_data = {col: str(row[col]) for col in binary_df.columns}
                data['binary_prediction'] = binary_data
        
        if property_pred_file and os.path.exists(property_pred_file):
            property_df = pd.read_csv(property_pred_file)
            if len(property_df) > frame_index:
                row = property_df.iloc[frame_index]
                property_data = {col: str(row[col]) for col in property_df.columns}
                data['property_prediction'] = property_data
        
        json_path = os.path.join(temp_dir, 'frame_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return image_path, json_path
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return None, None

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=50001,
        share=False,
        show_api=False
    )


