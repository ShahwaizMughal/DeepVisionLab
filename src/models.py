"""Model architectures used in the experiments: a custom baseline CNN and
ResNet-18 (from scratch or ImageNet-pretrained).
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaselineCNN(nn.Module):
    """A compact 3-block convolutional network for CIFAR-10.

    Architecture: three (Conv -> BatchNorm -> ReLU -> MaxPool) blocks with
    32/64/128 channels, followed by a fully connected classifier with
    dropout. Designed as a lightweight baseline to contrast against
    ResNet-18, not as an architecture search result.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int = 10, pretrained: bool = True):
    """Build a ResNet-18 with its final layer replaced for CIFAR-10.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, initialize from ImageNet weights
            (ResNet18_Weights.DEFAULT) — the transfer-learning setting.
            If False, initialize randomly — the from-scratch setting.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
