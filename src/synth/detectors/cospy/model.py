"""CO-SPY fusion model — semantic + artifact branch detector.

Simplified inference-only implementation of the CO-SPY architecture.
The full upstream model uses CLIP for semantic features and SRM filters
for artifact features, fused through a learned attention mechanism.

This implementation uses a dual-ResNet architecture as a lightweight
surrogate when the full CLIP+SRM weights are not available.

Architecture::

    image → [Semantic Branch (ResNet-50 / CLIP)]
                        ↓
                    sem_features
                        ↓
    image → [Artifact Branch (ResNet-18 + SRM)]
                        ↓
                    art_features
                        ↓
           ┌────── Fusion MLP ──────┐
           │  concat → FC → ReLU   │
           │  → FC → sigmoid       │
           └────────────────────────┘
                        ↓
                   probability
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ArtifactBranch(nn.Module):
    """Pixel-level artifact detection branch (SRM-inspired)."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet18

        base = resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class SemanticBranch(nn.Module):
    """Semantic-level detection branch (CLIP-inspired)."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        base = resnet50(weights=None)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        self.feat_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class COSPYFusionModel(nn.Module):
    """CO-SPY fusion model combining semantic and artifact branches.

    Concatenates features from both branches and passes through a
    fusion MLP for binary classification.
    """

    def __init__(self) -> None:
        super().__init__()
        self.semantic = SemanticBranch()
        self.artifact = ArtifactBranch()

        fusion_dim = self.semantic.feat_dim + self.artifact.feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Store the test transform as an attribute
        from synth.detectors.cospy.transforms import get_test_transform
        self.test_transform = get_test_transform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, 224, 224) normalised image tensor.

        Returns:
            (B, 1) logits tensor.
        """
        sem_feat = self.semantic(x)
        art_feat = self.artifact(x)
        fused = torch.cat([sem_feat, art_feat], dim=1)
        return self.fusion(fused)

    def predict(self, images: torch.Tensor) -> list[float]:
        """Run inference and return probabilities.

        Args:
            images: (B, 3, 224, 224) tensor.

        Returns:
            List of AI-generated probabilities.
        """
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits).squeeze(-1)
            return probs.cpu().tolist()

    def load_weights(self, path: str | Path) -> None:
        """Load pretrained fusion weights."""
        path = Path(path)
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
            logger.info("CO-SPY: loaded fusion weights from %s", path)
        else:
            logger.warning("CO-SPY: weights not found at %s", path)
