import torch
import numpy as np

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class InternVL:
    def __init__(self, size=3, min_px=256, max_px=1280):
        self.model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-4B",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
            

        # default processer
        min_pixels = min_px*28*28
        max_pixels = max_px*28*28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def run(self, image, prompt):
        content = []

        if image:
            content.append({"type": "image", "image": image})
        if prompt:
            content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to('cuda')


        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text