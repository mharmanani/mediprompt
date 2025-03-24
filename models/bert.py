import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoTokenizer, AutoModel

class BioBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device)
        self.d = 768

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Linear(1024, self.d)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.d, 2),
        )

        # Text encoder
        self.encoder = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad") 
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")

    def forward(self, text):
        text_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_input = text_input.to(self.device)
        text_features = self.encoder(**text_input).last_hidden_state.mean(dim=1)
        text_features = self.projector(text_features)
        y = self.classifier(text_features)
        return y

    def train(self, mode=True):
        if self.config.architecture.always_eval:
            mode = False
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