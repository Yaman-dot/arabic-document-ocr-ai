import os
import torch
os.environ["TORCH_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List
import numpy as np

MODEL_ID = "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2"

class QwenOCR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_enable_fp32_cpu_offload=True   # allows overflow to CPU RAM
            )

        )
        self.model.lm_head.weight = self.model.model.language_model.embed_tokens.weight


        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.image = None
        self.IMAGE_PATH = None
        self.predictions = None
        self.dyn_threshold = None
        print("Successfully loaded Qwen OCR")
        print(f"VRAM after Qwen: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    def load_image(self, image_path: str):
        self.IMAGE_PATH = image_path
        self.image = Image.open(image_path).convert("RGB")
    def inference(self, prompt="ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار"):
        if self.image is None:
            print("No image loaded.")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[self.image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        output_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        self.predictions = output_text.strip()
        return self.predictions

    def calculate_dynamic_threshold(self):
        # Qwen doesn't expose per-line confidence scores like Surya,
        # so we return a fixed sentinel — hook this up to your router logic as needed
        self.dyn_threshold = 1.0
        return self.dyn_threshold

    def flag_low_conf_paper(self, static_conf=0.8, use_static_threshold=False):
        # No confidence signal available from Qwen — always return "not flagged"
        return False, 1.0
