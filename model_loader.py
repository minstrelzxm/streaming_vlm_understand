import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time

SYSTEM_PROMPT = """
You are an advanced AI assistant operating inside the user's AR glasses.

VIEWPOINT: The video stream is a First-Person View (Egocentric). You see exactly what the user sees. The camera moves with the user's head.

SPATIAL AWARENESS: "Left" and "Right" refer to the user's left and right. Objects in the center of the frame are what the user is currently focusing on.

HANDS: If you see hands entering the frame, they are the user's hands. Interpret their actions (pointing, holding, crafting) as the user's intent.

ROLE: Act as an intelligent, helpful overlay. Provide concise, direct answers. Do not describe the scene generically unless asked. Focus on the objects the user is interacting with or looking at directly.

OUTPUT STYLE: Keep responses short (under 2 sentences) to avoid cluttering the AR display, unless the user asks for a detailed explanation.
"""

class VLMHandler:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VLMHandler] Loading model {model_id} to {self.device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                _attn_implementation="sdpa" if self.device == "cuda" else "eager",
            ).to(self.device)
            print("[VLMHandler] Model loaded successfully.")
        except Exception as e:
            print(f"[VLMHandler] Error loading model: {e}")
            raise e

    def analyze_frame(self, image: Image.Image, prompt="Describe this image briefly using one sentence."):
        if image is None:
            return "No image provided."
        
        try:
            # Create input messages
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=image, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Cast floating point inputs to match model dtype (e.g. float16 for CUDA)
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=self.model.dtype)

            # Generate
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # The output usually contains the prompt, so we might need to parse it or just return the full text.
            # SmolVLM typically returns the full conversation.
            return generated_texts[0]
        except Exception as e:
            return f"Error during inference: {e}"

    def analyze_video(self, video_path, prompt="Describe this video briefly using one sentence."):
        if not video_path:
            return "No video provided."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_path},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Cast floating point inputs
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=self.model.dtype)

            generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            return generated_texts[0]
        except Exception as e:
            return f"Error during video inference: {e}"

if __name__ == "__main__":
    # Test stub
    print("Initializing VLM Handler...")
    handler = VLMHandler()
    print("Ready.")
