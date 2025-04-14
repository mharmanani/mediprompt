import torch
import numpy as np

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from open_clip import create_model_from_pretrained, get_tokenizer

from PIL import Image

class BiomedCLIP:
    def __init__(self, device='cuda'):
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.context_length = 256

    def run(self, image, prompt):

        image = self.preprocess(Image.open(image)).to(self.device)
        text = self.tokenizer([prompt], context_length=self.context_length).to(self.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image.unsqueeze(0), text)
            logits = (logit_scale * image_features @ text_features.t()).detach()#.softmax(dim=-1)
        
        logits = logits.cpu().numpy()

        print(logits)
        
        return 1