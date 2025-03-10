"""
toxpre.py - 主程序入口
该代码实现了一个基于gradio的交互式界面，用于处理分子数据的预测和分析。
整合了聊天机器人功能和分子分析功能。
"""

import gradio as gr
from chatbot import ChatInterface
import json
import base64
import os
from interface import (
    convert_xyz_to_npz, 
    convert_npz_to_xyz,
    process_molecule_visualization,
    process_binary_prediction,
    process_property_prediction,
    run_extra,
    run_net,
    on_timestep_select,
    handle_uploaded_files,
    export_frame_data
)
from probability_plot import create_probability_plot
import logging
logger = logging.getLogger(__name__)

class IntegratedInterface:
    def __init__(self):
        self.chat_interface = ChatInterface()
        self.current_model = "google/gemini-2.0-flash-thinking-exp:free"  # 添加当前模型跟踪
        
    def handle_analysis_click(self, export_path, export_data, chatbot, image_input, model_name, custom_model=""):
        """处理Analysis按钮点击事件"""
        try:
            if not self.chat_interface.client:
                return (
                    chatbot,
                    "Please configure API first using the API Configuration panel",
                    gr.update(value=True),
                    gr.update(visible=True)
                )
            
            # 使用当前选择的模型
            actual_model = custom_model if model_name == "custom" else model_name
            
            # 处理导出数据并发送到聊天
            chat_history, error = self.process_export_for_chat(
                export_path, 
                export_data, 
                chatbot, 
                hide_prompt=True,
                model_name=actual_model  # 传递当前选择的模型
            )
            
            # 图片处理逻辑...
            image_update = gr.update(visible=True)
            if export_path and os.path.exists(export_path.name):
                try:
                    image_update = gr.update(value=export_path.name, visible=True)
                except Exception as e:
                    logger.error(f"Error setting image: {str(e)}")
            
            return (
                chat_history,
                error,
                gr.update(value=True),
                image_update
            )
        except Exception as e:
            return (
                chatbot,
                f"Error handling analysis: {str(e)}",
                gr.update(value=True),
                gr.update(visible=True)
            )



    def process_export_for_chat(self, export_path, export_data, chatbot_history, hide_prompt=False, model_name=None):
        """Process the exported data and send it to the chat"""
        if not export_path or not export_data:
            return chatbot_history, "Export files not found"
        
        try:
            # 读取JSON数据并创建提示词...
            with open(export_data.name, 'r') as f:
                json_data = json.load(f)
            
            json_prompt = ("You are an expert structural chemist and biologist. You have been provided with data about a molecule, including its predicted toxicity probability and other property predictions, which are appended below in JSON format. In this data, 'probability' represents the predicted toxicity probability, where 0 indicates non-toxic and 1 indicates toxic, and 'property_prediction' contains all additional predicted parameters of the molecule. Using your profound knowledge in chemistry and biology, please analyze this data and write a detailed report of at least 1000 words. In your report, interpret the significance of the toxicity probability, discuss the other predicted properties, and explain their potential implications for the molecule's behavior, safety, or applications. Provide your analysis in a direct and continuous format without section breaks." + 
                         json.dumps(json_data, ensure_ascii=False))
            
            # 读取图片并创建消息内容...
            with open(export_path.name, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            message_content = [
                {"type": "text", "text": json_prompt, "hidden": hide_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
            
            self.chat_interface.memory.add_message(
                "user", 
                message_content,
                has_image=True,
                hide_prompt=hide_prompt
            )
            
            # 使用指定的模型进行处理
            return self.chat_interface.process_message(
                json_prompt,
                export_path.name,
                True,
                chatbot_history,
                model_name or self.current_model  # 使用指定的模型或默认模型
            )
            
        except Exception as e:
            return chatbot_history, f"Error processing export: {str(e)}"

    def process_predictions(self, file, prop_model_path, ref_path, bin_model_path):
        """处理毒性预测"""
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

    def create_interface(self):
        with gr.Blocks() as demo:
            # State variables
            current_image = gr.State(None)
            species_data_state = gr.State(None)

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
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_copy_button=True
                )
                
                with gr.Row():
                    with gr.Column(scale=8):
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=3
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Send")
                        clear_btn = gr.Button("Clear")

                # Chat configuration components
                multimodal_enabled = gr.Checkbox(
                    label="Enable Image Input",
                    value=False
                )
                image_input = gr.Image(
                    label="Upload Image",
                    visible=False,
                    type="filepath"
                )

                with gr.Accordion("API Configuration", open=False):
                    base_url = gr.Textbox(
                        label="Base URL",
                        placeholder="Enter API base URL (e.g., https://openrouter.ai/api/v1)"
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key",
                        type="password"
                    )
                    model_select = gr.Dropdown(
                        choices=self.chat_interface.default_models,
                        label="Select Model",
                        value="google/gemini-2.0-flash-thinking-exp:free"
                    )
                    custom_model = gr.Textbox(
                        label="Custom Model Name",
                        placeholder="Enter custom model identifier",
                        visible=False
                    )
                    api_config_btn = gr.Button("Configure API")
                    error_box = gr.Textbox(
                        label="Error Messages",
                        visible=True,
                        interactive=False
                    )

                gr.Markdown("""### Here! : Predict & Visualize: First select different models, then upload the npz file, and the prediction task will start immediately after uploading. Clicking on "npz to xyz" allows you to visualize the molecular structure of the sequence. After the prediction is complete, drag the frame selector to view the structure and corresponding results. Clicking on "output" will save the image and results of that frame, while clicking on "analyze" will send the results of that frame to the Agent dialog.""")  
                with gr.Row():
                    # Left Column - Toxicity Prediction
                    with gr.Column(scale=1):
                        npz_file_input = gr.File(label="Upload NPZ file", file_types=[".npz"])
                        
                        gr.Markdown("#### Property Prediction")
                        property_model_path = gr.Textbox(
                            label="Model Directory Path",
                            value="/mnt/backup2/ai4s/backupunimolpy/MD_model",
                            interactive=True
                        )
                        reference_path = gr.Textbox(
                            label="Reference NPZ File Path",
                            value="/mnt/backup2/ai4s/unimolpy/refscale.npz",
                            interactive=True
                        )
                        property_predictions_output = gr.File(label="Property Predictions")
                        property_logs_output = gr.Textbox(label="Logs", lines=1)
                        
                        gr.Markdown("#### Binary Prediction")
                        binary_model_path = gr.Textbox(
                            label="Model Directory Path", 
                            value="/mnt/backup2/ai4s/backupunimolpy/ToxPred_modelmini",
                            interactive=True
                        )
                        convert_to_xyz = gr.Button("Convert to XYZ")
                        binary_predictions_output = gr.File(label="Binary Predictions")
                        binary_logs_output = gr.Textbox(label="Logs", lines=1)
                        xyz_conversion_output = gr.Textbox(label="XYZ Status", lines=1)
                        xyz_file_output = gr.File(label="XYZ file", visible=False)

                    # Right Column - Visualization
                    with gr.Column(scale=1):
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
                                with gr.Row():
                                    export_btn = gr.Button("Export")
                                    analysis_btn = gr.Button("Analysis")

                            export_path = gr.File(label="Export Results")
                            export_data = gr.File(label="Export Data")

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

            # Event Handlers
            # XYZ to NPZ Conversion
            convert_button.click(
                fn=convert_xyz_to_npz,
                inputs=[xyz_input],
                outputs=[conversion_output, npz_file]
            )

            # NPZ to XYZ Conversion and Visualization
            convert_to_xyz.click(
                fn=convert_npz_to_xyz,
                inputs=[npz_file_input],
                outputs=[xyz_conversion_output, xyz_input_vis]
            ).then(
                fn=process_molecule_visualization,
                inputs=[
                    xyz_input_vis,
                    frame_slider,
                    representation,
                    rotation_x,
                    rotation_y,
                    rotation_z,
                    zoom
                ],
                outputs=[output_image, color_legend, frame_slider, current_image]
            )

            # Toxicity Prediction
            npz_file_input.change(
                fn=self.process_predictions,
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

            # Export and Analysis
            export_btn.click(
                fn=export_frame_data,
                inputs=[
                    current_image,
                    export_format,
                    frame_slider,
                    binary_predictions_output,
                    property_predictions_output
                ],
                outputs=[export_path, export_data]
            )

            analysis_btn.click(
                fn=self.handle_analysis_click,
                inputs=[
                    export_path,
                    export_data,
                    chatbot,
                    image_input,
                    model_select,
                    custom_model
                ],
                outputs=[
                    chatbot,
                    error_box,
                    multimodal_enabled,
                    image_input
                ]
            ).then(
                fn=lambda: (gr.update(value=True), gr.update(visible=True)),
                outputs=[multimodal_enabled, image_input]
            )


            input_components = [
                xyz_input_vis,
                frame_slider,
                representation,
                rotation_x,
                rotation_y,
                rotation_z,
                zoom
            ]

            for component in [frame_slider, representation, rotation_x, rotation_y, rotation_z, zoom]:
                component.change(
                    fn=process_molecule_visualization,
                    inputs=input_components,
                    outputs=[output_image, color_legend, frame_slider, current_image]
                )


            def update_model_input(choice):
                return gr.update(visible=choice == "custom")

            def toggle_image_input(enabled):
                return gr.update(visible=enabled)

            def clear_chat():
                self.chat_interface.memory.clear()
                return None, ""
            def handle_api_config(url, key):
                try:
                    result = self.chat_interface.initialize_client(url, key)
                    return result
                except Exception as e:
                    return f"Error configuring API: {str(e)}"

            def handle_model_change(model_name, custom_model=""):
                self.current_model = custom_model if model_name == "custom" else model_name
                return gr.update(visible=model_name == "custom")

            api_config_btn.click(
                fn=handle_api_config,
                inputs=[base_url, api_key],
                outputs=[error_box]
            )

            model_select.change(
                fn=handle_model_change,
                inputs=[model_select, custom_model],
                outputs=[custom_model]
            )


            multimodal_enabled.change(
                fn=lambda enabled: gr.update(visible=enabled),
                inputs=[multimodal_enabled],
                outputs=[image_input]
            )

            submit_btn.click(
                fn=self.chat_interface.process_message,
                inputs=[
                    msg,
                    image_input,
                    multimodal_enabled,
                    chatbot,
                    model_select,
                    custom_model
                ],
                outputs=[chatbot, error_box]
            )

            msg.submit(
                fn=self.chat_interface.process_message,
                inputs=[
                    msg,
                    image_input,
                    multimodal_enabled,
                    chatbot,
                    model_select,
                    custom_model
                ],
                outputs=[chatbot, error_box]
            )

            clear_btn.click(
                fn=lambda: (None, "", gr.update(value=False), gr.update(visible=False)),
                outputs=[chatbot, error_box, multimodal_enabled, image_input]
            )


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

def main():
    interface = IntegratedInterface()
    demo = interface.create_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=50007,
        share=False,
        show_api=False
    )

if __name__ == "__main__":
    main()