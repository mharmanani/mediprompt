import torch
import numpy as np

from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoProcessor, TorchAoConfig
from qwen_vl_utils import process_vision_info

class Llama:
    def __init__(self, size=3, min_px=256, max_px=1280):
        model_name = "/model-weights/Llama-3.2-11B-Vision-Instruct/"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to('cuda')

        # default processer
        min_pixels = min_px*28*28
        max_pixels = max_px*28*28

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       min_pixels=min_pixels, 
                                                       max_pixels=max_pixels, 
                                                       pad_token="[PAD]")
        self.processor = AutoProcessor.from_pretrained(model_name, 
                                                       min_pixels=min_pixels, 
                                                       max_pixels=max_pixels, pad_token="[PAD]")

    def run(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
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