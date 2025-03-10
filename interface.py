# /mnt/backup2/ai4s/backupunimolpy/interface.py
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

# Configure logging
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
        """Process exported files and send to chat"""
        if not export_path or not export_data:
            return None, "Export failed"
            
        try:
            # Read the JSON data
            with open(export_data.name, 'r') as f:
                json_data = json.load(f)
            
            # Create the prompt with JSON context
            json_prompt = ("你是一名专业的结构化学家、生物学家，现在你得到了一个分子的有关数据，"
                         "我用模型预测它具有毒性的概率是probability，这个值0是无毒、1是有毒，"
                         "property_prediction是全部这个分子的预测参数，请你根据最深刻的化学知识，"
                         "写一份500字以上的分子分析报告，使用直接回答的方式。" + 
                         json.dumps(json_data, ensure_ascii=False))
            
            # Read the image file
            with open(export_path.name, 'rb') as f:
                image_data = f.read()
            
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Send to chat
            message_content = [
                {"type": "text", "text": json_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
            
            # Add to chat memory but make JSON invisible
            self.chat_interface.memory.add_message("user", message_content, has_image=True)
            
            # Process the message through chat interface
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
    """Convert XYZ file to NPZ format"""
    if not xyz_file:
        return "Error: No file uploaded", None
        
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Read the uploaded file in binary mode
            with open(xyz_file.name, 'rb') as f:
                content = f.read()
            
            # Save to temp file in binary mode
            temp_xyz = os.path.join(temp_dir, "temp.xyz")
            with open(temp_xyz, 'wb') as f:
                f.write(content)
            
            # Read and process data
            data_dict, invalid_frames = read_xyz_file(temp_xyz)
            
            # Create output filename in temp directory
            output_file = os.path.join(temp_dir, "converted_sequential.npz")
            
            # Save to NPZ
            info = save_npz(data_dict, output_file)
            
            # Add summary information
            summary = [
                "\nProcessing Report:",
                "-" * 50,
                f"Total frames processed: {len(data_dict['id'])}",
                f"Invalid frames: {len(invalid_frames)}",
                "-" * 50
            ]
            
            # Create a copy of the output file that will survive the temp directory cleanup
            final_output = os.path.join(os.getcwd(), "converted_sequential.npz")
            shutil.copy2(output_file, final_output)
            
            return "\n".join([info] + summary), final_output
            
    except Exception as e:
        return f"Error during conversion: {str(e)}", None

def convert_npz_to_xyz(npz_file):
    """Convert NPZ file to XYZ format"""
    if not npz_file:
        return "Error: No file uploaded", None
        
    try:
        # Create output filename
        output_file = os.path.splitext(npz_file.name)[0] + "_converted.xyz"
        
        # Convert file
        converter = NPZToXYZ(npz_file.name)
        result = converter.convert(output_file)
        
        return result, output_file
    except Exception as e:
        return f"Error during conversion: {str(e)}", None

def process_molecule_visualization(file, frame_index, representation, rotation_x, rotation_y, rotation_z, zoom):
    """处理分子可视化"""
    if file is None:
        return None, "", gr.Slider(visible=False), None
    
    img, legend, total_frames = display_molecule_pymol(
        file_path=file.name,
        frame_index=frame_index,
        representation=representation,
        rotations=[rotation_x, rotation_y, rotation_z],
        zoom=zoom
    )
    
    # 更新帧滑块的范围
    if total_frames > 0:
        return img, legend, gr.Slider(minimum=0, maximum=total_frames-1, step=1, visible=True), img
    else:
        return img, legend, gr.Slider(visible=False), img

def process_binary_prediction(file, model_path="/mnt/backup2/ai4s/unimolpy/unimol7300"):
    """Process uploaded file and return binary predictions"""
    if not file:
        return None, "Error: No file uploaded"
    
    try:
        predictor = BinaryPredictor(model_base_dir=model_path)
        output_path = file.name.replace('.npz', '_predictions.csv')
        predictions = predictor.predict(file.name, output_path)
        
        # 确保概率值在0-1范围内
        probabilities = predictions['probability'].values
        if np.max(probabilities) > 1 or np.min(probabilities) < 0:
            # 如果概率值不在0-1范围内，进行归一化
            probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))
            # 更新预测结果
            predictions['probability'] = probabilities
            predictions['prediction'] = (probabilities > 0.5).astype(int)
            # 保存更新后的预测结果
            predictions.to_csv(output_path, index=False)
        
        log_messages = [
            f"Predictions: +{(predictions['prediction'] == 1).sum()}, -{(predictions['prediction'] == 0).sum()}",
            f"Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}"
        ]
        
        return output_path, "\n".join(log_messages)
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_property_prediction(file, model_path, reference_path):
    """Process property predictions"""
    if not file or not model_path or not reference_path:
        return None, "Error: Missing inputs"
    try:
        predictor = MoleculePredictor()
        output_file, message = predictor.predict(file.name, reference_path, model_path)
        return output_file, message
    except Exception as e:
        return None, f"Error: {str(e)}"

def export_frame_data(current_image, export_format, frame_index, binary_pred_file, property_pred_file):
    """Export current frame image and corresponding data"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 导出图片（替换zip导出为直接图片导出）
        image_path = export_current_image(current_image, export_format)
        
        # 读取并处理CSV数据
        data = {}
        
        # 处理二进制预测数据
        if binary_pred_file and os.path.exists(binary_pred_file):
            binary_df = pd.read_csv(binary_pred_file)
            if len(binary_df) > frame_index:
                row = binary_df.iloc[frame_index]
                binary_data = {col: str(row[col]) for col in binary_df.columns}
                data['binary_prediction'] = binary_data
        
        # 处理属性预测数据
        if property_pred_file and os.path.exists(property_pred_file):
            property_df = pd.read_csv(property_pred_file)
            if len(property_df) > frame_index:
                row = property_df.iloc[frame_index]
                property_data = {col: str(row[col]) for col in property_df.columns}
                data['property_prediction'] = property_data
        
        # 保存JSON数据
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


