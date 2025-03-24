import torch
import torch.nn as nn
import torch.nn.functional as F


from src.models.bkfound import BKFound
from medAI.modeling.convnext import convnext_small
from transformers import AutoTokenizer, AutoModel

class ConvNext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device)
        self.d = 768

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.d, 2),
        )

        self.encoder = convnext_small(pretrained=True, in_22k=False)
        self.encoder = self.encoder.to(self.device)

    def forward(self, image):
        # Vision model
        image_features = self.encoder.forward_features(image)
        y = self.classifier(image_features)
        return y

    def train(self, mode=True):
        super().train(mode)

    def get_params_groups(self):
        """Return the parameters groups for the optimizer,
        which will be used to set different learning rates for different parts of the model.

        Returns:
            Tuple[tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter]]:
                encoder_parameters, warmup_parameters, cnn_parameters
                (warmup_parameters are the parameters for the prompt modules and the prompt encoder and mask decoder)
        """

        from itertools import chain

        encoder_params = [
            p
            for (k, p) in self.encoder.named_parameters()
            if "neck" not in k
        ]

        return encoder_params

    