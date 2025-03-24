import torch
import torch.nn as nn
import torch.nn.functional as F

from medAI.modeling.sam_wrappers import build_medsam
from medAI.modeling.vicreg import vicreg_loss_func

from transformers import AutoTokenizer, AutoModel

class BKClip(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device)
        self.d = 768

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projector1 = nn.Linear(1024, self.d)
        self.projector2 = nn.Linear(256, self.d)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.d, 2),
        )

        self.sim_loss_weight = 25.0
        self.var_loss_weight = 25.0
        self.cov_loss_weight = 1.0

        # Vision encoder
        # self.vision_encoder = BKFound(self.config.vision_backbone).medsam_model.image_encoder
        self.vision_encoder = build_medsam().image_encoder
        self.vision_encoder = self.vision_encoder.to(self.device)

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad") 
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")
        
        if self.config.training.freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        if self.config.training.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, text):
        # Tokenize text input
        text_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_input = text_input.to(self.device)
        text_features = self.text_encoder(**text_input).last_hidden_state.mean(dim=1)
        text_features = self.projector1(text_features)

        # Vision model
        image_features = self.vision_encoder(image)
        image_features = self.avg_pool(image_features).squeeze(-1).squeeze(-1)
        image_features = self.projector2(image_features)

        # Fuse image and text features
        image_features = F.normalize(image_features, p=2, dim=-1).squeeze(1)
        text_features = F.normalize(text_features, p=2, dim=-1).squeeze(1)
        y = self.classifier(image_features)

        # Return combined features (could be used for similarity, classification, etc.)
        return image_features, text_features, y

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()

    def get_params_groups(self):
        """Return the parameters groups for the optimizer,
        which will be used to set different learning rates for different parts of the model.

        Returns:
            Tuple[tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter]]:
                encoder_parameters, warmup_parameters, cnn_parameters
                (warmup_parameters are the parameters for the prompt modules and the prompt encoder and mask decoder)
        """

        from itertools import chain

        vision_encoder_params = [
            p
            for (k, p) in self.vision_encoder.named_parameters()
            if "neck" not in k
        ]
        text_encoder_params = chain(
            self.text_encoder.parameters(),
        )

        return vision_encoder_params, text_encoder_params

    def clip_loss(self, image_features, text_features, temperature=0.07):
        # Normalize the embeddings to unit vectors (if not already normalized)
        #image_features = F.normalize(image_features, p=2, dim=-1).squeeze(1)
        #text_features = F.normalize(text_features, p=2, dim=-1).squeeze(1)

        from medAI.modeling.simclr import SimCLRLoss

        # Compute the contrastive loss
        # simclr_loss = SimCLRLoss(temperature=temperature)
        # print(image_features, text_features)
        # loss = simclr_loss(image_features, text_features)
        # return loss

        # Compute cosine similarity (dot product)
        print(image_features.shape, text_features.shape)
        logits_per_image = torch.matmul(image_features, text_features.t())  # Image-to-text similarity
        print(logits_per_image.shape)
        logits_per_text = logits_per_image.t()  # Text-to-image similarity

        # Labels: diagonal ones, meaning the matching image-text pairs should have the highest similarity
        labels = torch.arange(image_features.size(0)).to(image_features.device)

        # Compute the contrastive loss (cross-entropy loss)
        loss_img_to_text = F.cross_entropy(logits_per_image / temperature, labels)
        loss_text_to_img = F.cross_entropy(logits_per_text / temperature, labels)

        # Combine both losses (since CLIP is symmetric, we average both)
        loss = (loss_img_to_text + loss_text_to_img) / 2
        return loss
    
    def vicreg_loss(self, image_features, text_features):
        return vicreg_loss_func(image_features, 
                                text_features, 
                                self.sim_loss_weight, 
                                self.var_loss_weight, 
                                self.cov_loss_weight)
    
    def reconstruction_loss(self, image_features, text_features):
        return F.mse_loss(image_features, text_features)
