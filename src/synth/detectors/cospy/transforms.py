"""CO-SPY test-time transforms.

Adapted from the upstream CO-SPY repository to provide consistent
image preprocessing during inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def get_test_transform(image_size: int = 224) -> Any:
    """Return a transform that resizes, centre-crops, and normalises.

    Uses a pure-functional pipeline (no torchvision.transforms dependency)
    to keep the install lightweight.

    Args:
        image_size: Target spatial dimension.

    Returns:
        A callable ``(PIL.Image) → torch.Tensor``.
    """
    # ImageNet normalisation constants
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def transform(img: Image.Image) -> torch.Tensor:
        # Resize to slightly larger, then centre-crop
        ratio = image_size / min(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.BILINEAR)

        # Centre crop
        left = (img.size[0] - image_size) // 2
        top = (img.size[1] - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))

        # To numpy → normalise → to tensor
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor

    return transform
