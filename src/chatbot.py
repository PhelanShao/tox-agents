# import gradio as gr # Removed Gradio
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
        
    def add_message(self, role: str, content: Union[str, List], has_image: bool = False, hide_prompt: bool = False):
        if not has_image:
            self.conversation_history = [
                msg for msg in self.conversation_history
                if not (isinstance(msg["content"], list) and len(msg["content"]) > 1)
            ]
        
        self.conversation_history.append({
            "role": role,
            "content": content,
            "has_image": has_image,
            "hide_prompt": hide_prompt
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
                    if msg.get("hide_prompt"):
                        has_shown_image = False
                        for content in msg["content"]:
                            if content.get("type") == "image_url" and not has_shown_image:
                                image_url = content["image_url"]["url"]
                                display_history.append((f"<img src='{image_url}' width='400'>", None))
                                has_shown_image = True
                    else:
                        text = msg["content"][0]["text"]
                        if len(msg["content"]) > 1:
                            image_url = msg["content"][1]["image_url"]["url"]
                            display_history.append((text, None))
                            display_history.append((f"<img src='{image_url}' width='400'>", None))
                        else:
                            display_history.append((text, None))
                else:
                    display_history.append((msg["content"], None))
            else:
                if isinstance(msg["content"], list):
                    display_history.append((None, msg["content"][0]["text"]))
                else:
                    display_history.append((None, msg["content"]))
        # This method was primarily for Gradio's Chatbot component.
        # The FastAPI backend should use self.memory.get_history() directly
        # or a similar method that returns raw message list.
        # For now, let it be, but it won't be directly used by FastAPI response.
        display_history = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # Simplified for non-Gradio use: just show text part if prompt not hidden
                    text_content = ""
                    image_info = ""
                    if msg.get("hide_prompt"):
                        # For hidden prompts (like analysis), maybe just indicate an image was sent
                        image_info = "[Image Sent]"
                    else:
                        for part in msg["content"]:
                            if part.get("type") == "text":
                                text_content += part["text"] + " "
                            elif part.get("type") == "image_url":
                                image_info = "[Image Attached]" # Don't include HTML img tag
                        text_content = text_content.strip()
                    display_history.append({"role": "user", "text": f"{text_content} {image_info}".strip()})
                else: # Simple text message
                    display_history.append({"role": "user", "text": msg["content"]})
            else: # Assistant message
                text_content = ""
                if isinstance(msg["content"], list) and msg["content"]: # Should be list of parts
                     for part in msg["content"]:
                        if part.get("type") == "text":
                            text_content += part["text"] + " "
                     text_content = text_content.strip()
                elif isinstance(msg["content"], str): # Simple text string
                    text_content = msg["content"]
                display_history.append({"role": "assistant", "text": text_content})
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
            actual_model = custom_model if model_name == "custom" else model_name
            message_content = []
            
            if user_input:
                message_content.append({"type": "text", "text": user_input})
            
            if multimodal_enabled and image is not None:
                is_vision_model = self.vision_models.get(actual_model, False)
                try:
                    with open(image, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    if not is_vision_model:
                        image_description = self.get_image_description(image)
                        message_content = [{
                            "type": "text",
                            "text": f"{user_input}\n[Image Description: {image_description}]"
                        }]
                    else:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_data}"}
                        })
                except Exception as e:
                    return history, f"Error processing image: {str(e)}"
            
            self.memory.add_message(
                "user",
                message_content,
                has_image=(multimodal_enabled and image is not None)
            )

            try:
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

                if actual_model.startswith("openai/"):
                    completion = self.client.chat.completions.create(
                        model=actual_model,
                        messages=api_messages,
                        extra_headers={
                            "HTTP-Referer": "localhost",
                            "X-Title": "Gradio Chat Interface",
                        },
                        extra_body={}
                    )
                else:
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
                    # Return raw history for API, not Gradio-formatted display history
                    return self.memory.get_history(), "" 
                else:
                    error_msg = "No response from API"
                    self.memory.add_message("assistant", [{"type": "text", "text": error_msg}])
                    return self.memory.get_history(), error_msg
                    
            except Exception as e:
                error_msg = f"Error in API call: {str(e)}"
                self.memory.add_message("assistant", [{"type": "text", "text": error_msg}])
                return self.memory.get_history(), error_msg

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # For FastAPI, the history passed in is the raw list of messages.
            # If an error occurs before API call, return this original history.
            # If it's after memory update, self.memory.get_history() will include the user's last message.
            return self.memory.get_history(), error_msg

    # Removed create_interface() method and if __name__ == "__main__": block
    # as Gradio UI is no longer part of this file when used by FastAPI backend.