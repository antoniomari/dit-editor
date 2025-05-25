### Unclean code courtesy of https://github.com/christophschuhmann/improved-aesthetic-predictor


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_aesthetic_model(device):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "sac+logos+ava1-l14-linearMSE.pth", weights_only=False
    )  # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)

    model.to(device)
    model.eval()

    # also load clip model
    # clip_model, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = clip_model.to(device)

    # processor.image_processor is a torchvision.Compose of:
    #   Resize(224, BICUBIC) → CenterCrop(224) → ToTensor() → Normalize(mean, std)
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return model, clip_model, preprocess


class MLP(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
