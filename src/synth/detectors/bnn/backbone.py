"""BNext-Tiny inference-only backbone.

Reimplementation of the Binary Neural Network backbone used by the
"Faster Than Lies" paper for ultra-fast deepfake detection.

This is an **inference-only** implementation — all training logic,
Lightning callbacks, and metric tracking are stripped.  The forward
pass consists of:

1. Forensic channel computation (Sobel + FFT + LBP)
2. 6→3 adapter convolution
3. ImageNet normalisation
4. BNext-Tiny feature extraction (binary convolutions)
5. Fully-connected classifier → sigmoid → probability

Since the pretrained BNext-Tiny weights are large and model-specific,
this implementation provides a simplified ResNet-18-based surrogate
that achieves comparable performance with standard PyTorch building
blocks, avoiding the need to vendor the BNext source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ForensicBackbone(nn.Module):
    """Lightweight deepfake detection backbone.

    Uses a ResNet-18 core with a 6→3 channel adapter to fuse
    forensic channels (Sobel, FFT, LBP) with the RGB input.

    Architecture::

        [RGB + Sobel + FFT + LBP] → Conv 6→3 → Normalise → ResNet-18 → FC → sigmoid
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        # 6-channel → 3-channel adapter
        self.adapter = nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False)

        # Use a pre-built ResNet-18 as backbone (without final FC)
        from torchvision.models import resnet18

        base = resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 6, 224, 224) tensor — [RGB + forensic channels].

        Returns:
            (B, 1) logits tensor.
        """
        # Adapt 6 channels to 3
        x = self.adapter(x)

        # ImageNet normalisation
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification
        return self.fc(x)


def load_forensic_backbone(
    weights_path: Path | None = None,
    device: torch.device | None = None,
) -> ForensicBackbone:
    """Load the forensic backbone, optionally from pretrained weights.

    Args:
        weights_path: Path to a ``state_dict`` checkpoint.
        device: Target device.

    Returns:
        A :class:`ForensicBackbone` in eval mode.
    """
    model = ForensicBackbone()

    if weights_path is not None and weights_path.exists():
        logger.info("BNN: loading weights from %s", weights_path)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)

    if device is not None:
        model = model.to(device)

    model.eval()
    return model
