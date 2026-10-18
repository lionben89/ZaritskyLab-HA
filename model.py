"""Mitotic / non-mitotic cell classifier.

Usage:
    from model import load_classifier
    model = load_classifier("classifier.pt")
    labels = model.is_mitotic(images)        # images: float tensor [B, 1, H, W], pixels in [0, 1]

Input images are single-channel fluorescence (DAPI/Hoechst) crops with pixel values in [0, 1];
resizing and normalisation are done inside the model.
"""

import torch
from torch import nn
from torchvision import models

IMAGE_SIZE = 96
NORM_MEAN, NORM_STD = 0.5, 0.5

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


class ResNetClassifier(nn.Module):
    """ResNet backbone + linear head for single-channel cell images.

    ``get_embeddings`` returns the pooled backbone features (the input to the head);
    ``get_logits`` returns the class logits (index 0: non-mitotic, 1: mitotic);
    ``is_mitotic`` returns the predicted label per image (1 = mitotic / dividing, 0 = non-mitotic).
    """

    def __init__(self, backbone: str = "resnet18", num_classes: int = 2, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        builder, weights = _BACKBONES[backbone]
        net = builder(weights=weights if pretrained else None)

        if in_channels != 3:
            # Grayscale input: reuse the pretrained RGB filters by averaging over the colour axis.
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                  stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))

        self.embedding_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Linear(self.embedding_dim, num_classes)

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] images in [0, 1] -> resized to IMAGE_SIZE and normalised."""
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                                          align_corners=False, antialias=True)
        return (x - NORM_MEAN) / NORM_STD

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(x))

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_embeddings(x))

    @torch.no_grad()
    def is_mitotic(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted label per image: 1 = mitotic, 0 = non-mitotic (long tensor of shape [B])."""
        return self.get_logits(x).argmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_logits(x)


def load_classifier(path, device="cpu", freeze: bool = True) -> ResNetClassifier:
    """Load a checkpoint saved as {"state_dict": ..., "backbone": ...}; by default frozen and in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ResNetClassifier(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    if freeze:
        model.requires_grad_(False)
    return model
