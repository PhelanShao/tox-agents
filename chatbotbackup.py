# /mnt/backup2/ai4s/backupunimolpy/chatbot.py
import gradio as gr
import os
from openai import OpenAI
import json
from typing import List, Dict, Union, Optional
import base64
from PIL import Image
import io

class ChatMemory:
    def __init__(self):
        self.conversation_history = []
        
    def add_message(self, role: str, content: Union[str, List], has_image: bool = False):
        # 如果是新的文本消息,移除之前消息中的图片相关内容
        if not has_image:
            self.conversation_history = [
                msg for msg in self.conversation_history 
                if not (isinstance(msg["content"], list) and len(msg["content"]) > 1)
            ]
        
        self.conversation_history.append({
            "role": role,
            "content": content,
            "has_image": has_image
        })
        
    def get_history(self) -> List[Dict]:
        return self.conversation_history
        
    def clear(self):
        self.conversation_history = []

    def get_display_history(self) -> List[tuple]:
        display_history = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # 处理包含图片的消息
                    text = msg["content"][0]["text"]
                    if len(msg["content"]) > 1:
                        image_url = msg["content"][1]["image_url"]["url"]
                        display_history.append((text, None))
                        display_history.append((f"<img src='{image_url}' width='400'>", None))
                    else:
                        display_history.append((text, None))
                else:
                    display_history.append((msg["content"], None))
            else:  # assistant
                if isinstance(msg["content"], list):
                    display_history.append((None, msg["content"][0]["text"]))
                else:
                    display_history.append((None, msg["content"]))
        return display_history

class ChatInterface:
    def __init__(self):
        self.memory = ChatMemory()
        self.client = None
        self.default_models = [
            "google/gemini-2.0-flash-001",
            "openai/o1",
            "openai/o3-mini-high",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1-distill-llama-70b",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
            "openai/gpt-4o-2024-11-20",
            "x-ai/grok-2-vision-1212",
            "mistralai/pixtral-large-2411",
            "qwen/qvq-72b-preview",
            "custom"
        ]
        self.vision_models = {
            "google/gemini-2.0-flash-001": True,
            "openai/o1": True,
            "openai/o3-mini-high": False,
            "deepseek/deepseek-r1": False,
            "deepseek/deepseek-r1-distill-llama-70b": False,
            "anthropic/claude-3.5-sonnet": True,
            "openai/gpt-4o-mini": False,
            "openai/gpt-4o-2024-11-20": True,
            "qwen/qvq-72b-preview": True,
            "mistralai/pixtral-large-2411": True,
            "x-ai/grok-2-vision-1212": True
        }
        
    def initialize_client(self, base_url: str, api_key: str) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def get_image_description(self, image_path: str) -> str:
        try:
            vision_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.client.api_key
            )
            
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            completion = vision_client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct:free",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述这张图片，如果图片包含化学结构请尽可能详细介绍这些结构给与准确答复"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error getting image description: {str(e)}"

    def process_message(self, 
                       user_input: str,
                       image: Optional[str],
                       multimodal_enabled: bool,
                       history: List,
                       model_name: str,
                       custom_model: str = "") -> tuple:
        if not self.client:
            return None, "Please configure API settings first."

        try:
            actual_model = custom_model if model_name == "custom" else model_name
            message_content = []
            
            # 处理用户输入
            if user_input:
                message_content.append({"type": "text", "text": user_input})
            
            # 处理图片
            if multimodal_enabled and image is not None:
                is_vision_model = self.vision_models.get(actual_model, False)
                
                with open(image, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                if not is_vision_model:
                    # 非视觉模型获取图片描述
                    image_description = self.get_image_description(image)
                    message_content = [{
                        "type": "text", 
                        "text": f"{user_input}\n[Image Description: {image_description}]"
                    }]
                else:
                    # 视觉模型直接使用图片
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                    })

            # 添加用户消息到记忆
            self.memory.add_message(
                "user", 
                message_content,
                has_image=(multimodal_enabled and image is not None)
            )

            try:
                # API调用
                api_messages = []
                for msg in self.memory.get_history():
                    api_msg = {"role": msg["role"]}
                    if isinstance(msg["content"], list):
                        if msg["has_image"]:
                            api_msg["content"] = msg["content"]
                        else:
                            api_msg["content"] = msg["content"][0]["text"]
                    else:
                        api_msg["content"] = msg["content"]
                    api_messages.append(api_msg)

                completion = self.client.chat.completions.create(
                    model=actual_model,
                    messages=api_messages,
                    extra_headers={
                        "HTTP-Referer": "localhost",
                        "X-Title": "Gradio Chat Interface",
                    }
                )

                assistant_message = completion.choices[0].message.content
                self.memory.add_message("assistant", [{"type": "text", "text": assistant_message}])
                
            except Exception as e:
                error_msg = f"Error in API call: {str(e)}"
                self.memory.add_message("assistant", [{"type": "text", "text": error_msg}])

            # 返回完整对话历史
            return self.memory.get_display_history(), ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return [(None, error_msg)], error_msg

    def create_interface(self):
        with gr.Blocks() as interface:
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

            # 图片输入选项
            multimodal_enabled = gr.Checkbox(
                label="Enable Image Input",
                value=False
            )
            image_input = gr.Image(
                label="Upload Image",
                visible=False,
                type="filepath"
            )

            # 可折叠的配置面板
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
                    choices=self.default_models,
                    label="Select Model",
                    value="google/gemini-2.0-flash-001"
                )
                custom_model = gr.Textbox(
                    label="Custom Model Name",
                    placeholder="Enter custom model identifier",
                    visible=False,
                    interactive=True
                )
                api_config_btn = gr.Button("Configure API")
                error_box = gr.Textbox(
                    label="Error Messages",
                    visible=True,
                    interactive=False
                )

            def update_model_input(choice):
                return gr.update(visible=choice == "custom")

            def configure_api(url, key):
                try:
                    self.initialize_client(url, key)
                    return "API configured successfully!"
                except Exception as e:
                    return f"Error configuring API: {str(e)}"

            def toggle_image_input(enabled):
                return gr.update(visible=enabled)

            def clear_chat():
                self.memory.clear()
                return None, ""

            # Event handlers
            model_select.change(
                update_model_input,
                inputs=[model_select],
                outputs=[custom_model]
            )

            multimodal_enabled.change(
                toggle_image_input,
                inputs=[multimodal_enabled],
                outputs=[image_input]
            )

            submit_btn.click(
                self.process_message,
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
                self.process_message,
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
                clear_chat,
                outputs=[chatbot, error_box]
            )

            api_config_btn.click(
                configure_api,
                inputs=[base_url, api_key],
                outputs=[error_box]
            )

        return interface

if __name__ == "__main__":
    chat_interface = ChatInterface()
    interface = chat_interface.create_interface()
    interface.launch(share=True)