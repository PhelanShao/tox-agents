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
from chatbot import ChatInterface

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

def export_frame_data(current_image, export_format, frame_index, binary_pred_file, property_pred_file, chat_interface):
    """Export current frame image and corresponding CSV data"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        chat_prompt = "你是一名专业的结构化学家、生物学家，现在你得到了一个分子的有关数据，我用模型预测它具有毒性的概率是probability，这个值0是无毒、1是有毒，property_prediction是全部这个分子的预测参数，请你根据最深刻的化学知识，写一份500字以上的分子分析报告，使用直接回答的方式。"
        
        # 导出图片
        image_path = export_current_image(current_image, export_format)
        temp_image = os.path.join(temp_dir, os.path.basename(image_path))
        shutil.copy2(image_path, temp_image)
        os.remove(image_path)  # 删除原始图片
        
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
        
        # 创建ZIP文件包含所有导出内容
        zip_path = os.path.join(temp_dir, 'export.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(temp_image, os.path.basename(temp_image))
            zf.write(json_path, os.path.basename(json_path))
        
        # 自动发送到聊天机器人
        if chat_interface and chat_interface.client:
            # 发送图片
            chat_interface.process_message(
                "",
                temp_image,
                True,
                [],
                "google/gemini-2.0-flash-001"
            )
            
            # 发送JSON数据作为上下文
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            chat_interface.process_message(
                chat_prompt + json.dumps(json_data, ensure_ascii=False),
                None,
                False,
                [],
                "google/gemini-2.0-flash-001"
            )
            
        return zip_path, json_path
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return None, None

def create_interface():
    with gr.Blocks() as demo:
        # Initialize chat interface
        chat_interface = ChatInterface()
        
        # XYZ to NPZ Converter Tab
        with gr.Tab("XYZ to NPZ Converter"):
            gr.Markdown("### Step 1: Convert XYZ to NPZ")
            xyz_input = gr.File(label="Upload XYZ file")
            convert_button = gr.Button("Convert to NPZ")
            conversion_output = gr.Textbox(label="Status", lines=2)
            npz_file = gr.File(label="Download NPZ file")

        # Combined Toxicity Prediction and Visualization Tab
        with gr.Tab("Toxicity Prediction & Visualization"):
            # Add chat interface at the top
            chat_component = chat_interface.create_interface()
            
            gr.Markdown("### Step 2: Predict Toxicity and Visualize Structure")
            
            with gr.Row():
                # Left Column - Toxicity Prediction
                with gr.Column(scale=1):
                    # 共用的文件上传
                    npz_file_input = gr.File(label="Upload NPZ file", file_types=[".npz"])
                    
                    # Property Prediction Section
                    gr.Markdown("#### Property Prediction")
                    property_model_path = gr.Textbox(
                        label="Model Directory Path",
                        value="/mnt/backup2/ai4s/unimolpy/21stmodel_20250221",
                        interactive=True
                    )
                    reference_path = gr.Textbox(
                        label="Reference NPZ File Path",
                        value="/mnt/backup2/ai4s/unimolpy/21stregression.npz",
                        interactive=True
                    )
                    property_predictions_output = gr.File(label="Property Predictions")
                    property_logs_output = gr.Textbox(label="Logs", lines=1)
                    
                    # Binary Prediction Section
                    gr.Markdown("#### Binary Prediction")
                    binary_model_path = gr.Textbox(
                        label="Model Directory Path", 
                        value="/mnt/backup2/ai4s/unimolpy/unimol7300",
                        interactive=True
                    )
                    convert_to_xyz = gr.Button("Convert to XYZ")
                    binary_predictions_output = gr.File(label="Binary Predictions")
                    binary_logs_output = gr.Textbox(label="Logs", lines=1)
                    xyz_conversion_output = gr.Textbox(label="XYZ Status", lines=1)
                    xyz_file_output = gr.File(label="XYZ file", visible=False)

                # Right Column - Visualization
                with gr.Column(scale=1):
                    # Probability Plot
                    probability_plot = gr.Plot(label="Probability Analysis")
                    
                    # Molecular Structure
                    output_image = gr.Image(label="Structure View")
                    color_legend = gr.HTML(label="Color Legend")
                    
                    # Hidden XYZ Input for Visualization
                    xyz_input_vis = gr.File(label="XYZ File", visible=False)
                    
                    # Visualization Controls
                    with gr.Accordion("Visualization Controls", open=False):
                        frame_slider = gr.Slider(
                            minimum=0, maximum=0, step=1, value=0,
                            label="Select Frame", visible=False
                        )
                        representation = gr.Dropdown(
                            choices=["sticks", "ball_and_stick", "spacefill", "wireframe", "surface"],
                            value="sticks",
                            label="Representation Style"
                        )
                        with gr.Row():
                            rotation_x = gr.Slider(-180, 180, value=0, label="X Rotation")
                            rotation_y = gr.Slider(-180, 180, value=0, label="Y Rotation")
                            rotation_z = gr.Slider(-180, 180, value=0, label="Z Rotation")
                        zoom = gr.Slider(0.1, 5.0, value=1.0, label="Zoom")
                        with gr.Row():
                            export_format = gr.Dropdown(
                                choices=["PNG", "JPG"],
                                value="PNG",
                                label="Export Format"
                            )
                            export_btn = gr.Button("Export")
                        export_path = gr.File(label="Export Results")
                        export_data = gr.File(label="Export Data")
                        current_image = gr.State(None)

        # Nano Reactor Tab
        with gr.Tab("Nano Reactor"):
            gr.Markdown("# Molreac Nano Reactor Interface")
            
            with gr.Row():
                with gr.Column(scale=1):
                    job_id = gr.Textbox(label="Job ID")
                    param3_file = gr.File(
                        label="Upload parameters3.dat",
                        elem_classes="compact-file"
                    )
                    upload_status = gr.Textbox(label="Upload Status", lines=1)
                    results_download = gr.File(
                        label="Results Archive",
                        elem_classes="compact-file"
                    )
                
                with gr.Column(scale=2):
                    job_files = gr.File(label="Select files for Job ID", file_count="multiple")
                    upload_button = gr.Button("Upload Files")

            with gr.Row():
                run_extra_button = gr.Button("Run Extra")
                run_net_button = gr.Button("Analyze Fragments")

            extra_log_box = gr.Textbox(label="Extra Module Log", lines=1)
            plot_output = gr.Plot(label="Extracted Data Plot")
            
            net_output = gr.Textbox(label="Analysis Output", lines=1)
            timeline_plot = gr.Plot(label="Fragment Timeline")
            
            with gr.Row():
                timestep_input = gr.Textbox(
                    label="Select Timestep (use comma to separate multiple steps, or 'all' for all timesteps)",
                    scale=3
                )
                extract_button = gr.Button("Extract Fragment Coordinates", scale=1)
            
            extract_output = gr.Textbox(label="Extraction Status")
            fragment_files = gr.File(label="Fragment Coordinates")
            species_data_state = gr.State(None)

        # Event Handlers
        convert_button.click(
            fn=convert_xyz_to_npz,
            inputs=[xyz_input],
            outputs=[conversion_output, npz_file]
        )

        def process_predictions(file, prop_model_path, ref_path, bin_model_path):
            """Process both property and binary predictions"""
            if not file:
                return None, "Error: No file", None, "Error: No file", None
            
            try:
                # Property prediction
                prop_output, prop_message = process_property_prediction(
                    file, 
                    prop_model_path,
                    ref_path
                )
                
                # Binary prediction
                bin_output, bin_message = process_binary_prediction(
                    file,
                    bin_model_path
                )
                
                # Create probability plot
                prob_plot = None
                if bin_output:
                    prob_plot = create_probability_plot(bin_output)
                
                return prop_output, prop_message, bin_output, bin_message, prob_plot
            except Exception as e:
                return None, f"Error: {str(e)}", None, f"Error: {str(e)}", None

        # Property and Binary Prediction Event Handler
        npz_file_input.change(
            fn=process_predictions,
            inputs=[
                npz_file_input,
                property_model_path,
                reference_path,
                binary_model_path
            ],
            outputs=[
                property_predictions_output,
                property_logs_output,
                binary_predictions_output,
                binary_logs_output,
                probability_plot
            ]
        )

        # Convert to XYZ and Visualize Handler
        convert_to_xyz.click(
            fn=convert_npz_to_xyz,
            inputs=[npz_file_input],
            outputs=[xyz_conversion_output, xyz_input_vis]
        )

        # Molecule Visualization Event Handlers
        input_components = [
            xyz_input_vis,
            frame_slider,
            representation,
            rotation_x,
            rotation_y,
            rotation_z,
            zoom
        ]

        # 文件上传时更新帧滑块和显示
        xyz_input_vis.change(
            fn=process_molecule_visualization,
            inputs=input_components,
            outputs=[output_image, color_legend, frame_slider, current_image]
        )

        # 其他参数变化时更新显示
        def update_display(*args):
            img, legend, _, new_img = process_molecule_visualization(*args)
            return img, legend, new_img

        for component in [frame_slider, representation, rotation_x, rotation_y, rotation_z, zoom]:
            component.change(
                fn=update_display,
                inputs=input_components,
                outputs=[output_image, color_legend, current_image]
            )

        # 导出按钮事件
        export_btn.click(
            fn=export_frame_data,
            inputs=[
                current_image,
                export_format,
                frame_slider,
                binary_predictions_output,
                property_predictions_output,
                chat_interface
            ],
            outputs=[export_path, export_data]
        )

        # Nano Reactor Event Handlers
        upload_button.click(
            fn=handle_uploaded_files,
            inputs=[job_files, job_id],
            outputs=[upload_status]
        )

        run_extra_button.click(
            fn=run_extra,
            inputs=[param3_file, job_id, job_files],
            outputs=[extra_log_box, plot_output, results_download]
        )

        run_net_button.click(
            fn=run_net,
            inputs=[job_id, job_files],
            outputs=[net_output, timeline_plot, species_data_state]
        )

        extract_button.click(
            fn=on_timestep_select,
            inputs=[job_id, timestep_input, species_data_state],
            outputs=[extract_output, fragment_files]
        )
        
        demo.queue()
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=50001,
        share=False,
        show_api=False
    )
