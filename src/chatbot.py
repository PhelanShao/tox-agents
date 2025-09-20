import gradio as gr
from openai import OpenAI
import json
from typing import List, Dict, Union, Optional, Any
import base64
from pathlib import Path

class ChatMemory:
    def __init__(self):
        self.conversation_history = []
        
    def add_message(self, role: str, content: Union[str, List], has_image: bool = False, hide_prompt: bool = False):
        """添加消息到对话历史"""
        # 如果是新的文本消息,移除之前消息中的图片相关内容
        if not has_image:
            self.conversation_history = [
                msg for msg in self.conversation_history 
                if not (isinstance(msg["content"], list) and len(msg["content"]) > 1)
            ]
        
        self.conversation_history.append({
            "role": role,
            "content": content,
            "has_image": has_image,
            "hide_prompt": hide_prompt  # 添加hide_prompt标记
        })
        
    def get_history(self) -> List[Dict]:
        return self.conversation_history
        
    def clear(self):
        self.conversation_history = []

    def get_display_history(self) -> List[tuple]:
        display_history = []
        for msg in self.conversation_history:
            if msg.get("hide_prompt"):
                continue
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # 检查是否需要隐藏提示词
                    if msg.get("hide_prompt"):
                        # 如果是隐藏提示词的消息，只显示图片
                        has_shown_image = False
                        for content in msg["content"]:
                            if content.get("type") == "image_url" and not has_shown_image:
                                image_url = content["image_url"]["url"]
                                display_history.append((f"<img src='{image_url}' width='400'>", None))
                                has_shown_image = True
                    else:
                        # 正常显示所有内容
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
        self.prompt_config = self._load_prompt_config()
        self.system_prompt = self._build_system_prompt(self.prompt_config)
        self.default_models = [
            "google/gemini-2.0-flash-001",
            "openai/o1",
            "openai/o3-mini-high",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1-distill-llama-70b",
            "anthropic/claude-3.7-sonnet",
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
            "openai/o3-mini-high": True,
            "deepseek/deepseek-r1": False,
            "deepseek/deepseek-r1-distill-llama-70b": False,
            "anthropic/claude-3.7-sonnet": True,
            "openai/gpt-4o-mini": False,
            "openai/gpt-4o-2024-11-20": True,
            "qwen/qvq-72b-preview": True,
            "mistralai/pixtral-large-2411": True,
            "x-ai/grok-2-vision-1212": True
        }
        self._apply_model_defaults()
        self.ensure_system_prompt()
        
    def _config_search_paths(self) -> List[Path]:
        base_dir = Path(__file__).resolve().parent
        return [
            base_dir / "frontend" / "backend" / "llm_report_config.json",
            base_dir / "llm_report_config.json"
        ]

    def _load_prompt_config(self) -> Optional[Dict[str, Any]]:
        for path in self._config_search_paths():
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as config_file:
                        return json.load(config_file)
                except Exception as exc:
                    print(f"⚠️ 无法解析提示词配置 {path}: {exc}")
        return None

    def _build_system_prompt(self, config: Optional[Dict[str, Any]]) -> Optional[str]:
        if not config:
            return None

        prompts = config.get("prompts", {})
        a1_prompt = prompts.get("A1_system_prompt", {})
        shap_digest = prompts.get("A2_shap_thresholds_digest", [])

        lines: List[str] = []

        role = a1_prompt.get("role")
        if role:
            lines.append(f"Role: {role}")

        grounding = a1_prompt.get("grounding")
        if grounding:
            lines.append(f"Grounding: {grounding}")

        evidence_style = a1_prompt.get("evidence_style")
        if evidence_style:
            lines.append(f"Evidence style: {evidence_style}")

        uncertainty = a1_prompt.get("uncertainty_and_applicability_domain")
        if uncertainty:
            lines.append(f"Uncertainty & AD: {uncertainty}")

        tools = a1_prompt.get("tools", [])
        if tools:
            lines.append("Available tools:")
            for tool in tools:
                lines.append(f"- {tool}")

        required_outputs = a1_prompt.get("required_outputs", [])
        if required_outputs:
            lines.append("Required outputs (include each item):")
            for item in required_outputs:
                lines.append(f"- {item}")

        reasoning_policy = a1_prompt.get("reasoning_policy")
        if reasoning_policy:
            lines.append(f"Reasoning policy: {reasoning_policy}")

        if shap_digest:
            lines.append("SHAP thresholds (descriptor | threshold | direction | reliability):")
            for entry in shap_digest:
                descriptor = entry.get("descriptor", "unknown")
                threshold = entry.get("threshold", "unknown")
                direction = entry.get("direction", "unspecified")
                reliability = entry.get("reliability", "unknown")
                lines.append(f"- {descriptor}: threshold {threshold}, direction {direction}, reliability {reliability}")

        lines.append("Always produce auditable decision records citing descriptor, measured value, threshold comparison, direction, and reliability for each claim. State 'AD unknown' whenever applicability metrics are missing.")

        return "\n".join(lines)

    def _apply_model_defaults(self) -> None:
        default_model = None
        if self.prompt_config:
            default_model = self.prompt_config.get("llm_model")
        if default_model and default_model != "TBD" and default_model not in self.default_models:
            self.default_models.insert(0, default_model)

    def ensure_system_prompt(self) -> None:
        if not self.system_prompt:
            return
        if not any(msg for msg in self.memory.get_history() if msg.get("role") == "system"):
            self.memory.add_message("system", self.system_prompt, hide_prompt=True)

    def initialize_client(self, base_url: str, api_key: str) -> str:
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            return "API configured successfully!"
        except Exception as e:
            return f"Error configuring API: {str(e)}"

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
                        {"type": "text", "text": "Please describe this image. If the image contains chemical structures, please provide as much detail as possible and give an accurate response."},
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
            self.ensure_system_prompt()
            actual_model = custom_model if model_name == "custom" else model_name
            message_content = []
            
            # 处理用户输入
            if user_input:
                message_content.append({"type": "text", "text": user_input})
            
            # 处理图片
            if multimodal_enabled and image is not None:
                is_vision_model = self.vision_models.get(actual_model, False)
                try:
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
                            "image_url": {"url": f"data:image/png;base64,{img_data}"}
                        })
                except Exception as e:
                    return history, f"Error processing image: {str(e)}"
            
            # 添加用户消息到记忆
            self.memory.add_message(
                "user", 
                message_content,
                has_image=(multimodal_enabled and image is not None)
            )

            try:
                # 准备API消息
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

                # OpenAI模型特殊处理
                if actual_model.startswith("openai/"):
                    completion = self.client.chat.completions.create(
                        model=actual_model,
                        messages=api_messages,
                        extra_headers={
                            "HTTP-Referer": "localhost",
                            "X-Title": "Gradio Chat Interface",
                        },
                        extra_body={}  # 可以添加额外的参数
                    )
                else:
                    # 其他模型使用原有方式
                    completion = self.client.chat.completions.create(
                        model=actual_model,
                        messages=api_messages,
                        extra_headers={
                            "HTTP-Referer": "localhost",
                            "X-Title": "Gradio Chat Interface",
                        }
                    )

                if completion and completion.choices:
                    assistant_message = completion.choices[0].message.content
                    self.memory.add_message("assistant", [{"type": "text", "text": assistant_message}])
                    return self.memory.get_display_history(), ""
                else:
                    error_msg = "No response from API"
                    self.memory.add_message("assistant", [{"type": "text", "text": error_msg}])
                    return self.memory.get_display_history(), error_msg
                    
            except Exception as e:
                error_msg = f"Error in API call: {str(e)}"
                self.memory.add_message("assistant", [{"type": "text", "text": error_msg}])
                return self.memory.get_display_history(), error_msg

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return history, error_msg

    def clear_chat(self):
        """重置聊天状态并重新注入系统提示词"""
        self.memory.clear()
        self.ensure_system_prompt()
        return self.memory.get_display_history(), "", gr.update(value=False), gr.update(value=None, visible=False)

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
                default_model_value = self.default_models[0] if self.default_models else "custom"
                if self.prompt_config:
                    cfg_model = self.prompt_config.get("llm_model")
                    if cfg_model and cfg_model != "TBD":
                        default_model_value = cfg_model
                base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="Enter API base URL (e.g., https://x/api/v1)"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key",
                    type="password"
                )
                model_select = gr.Dropdown(
                    choices=self.default_models,
                    label="Select Model",
                    value=default_model_value
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
                fn=self.clear_chat,  # 使用类方法
                outputs=[chatbot, error_box, multimodal_enabled, image_input]
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
