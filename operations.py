from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageColor


def _load_mask(mask_path: str) -> Image.Image:
    mask_img = Image.open(mask_path)
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    return mask_img


def _mask_from_polygon(size, polygon):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)
    return mask


def _mask_from_bbox(size, bbox):
    x1, y1, x2, y2 = bbox
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def draw_mask(image: str, mask_path: str | None = None, polygon=None, bbox=None, color="red", alpha=0.4):
    base = Image.open(image).convert("RGB")
    if mask_path:
        mask = _load_mask(mask_path)
        if mask.size != base.size:
            mask = mask.resize(base.size, resample=Image.NEAREST)
    elif polygon:
        mask = _mask_from_polygon(base.size, polygon)
    elif bbox:
        mask = _mask_from_bbox(base.size, bbox)
    else:
        return base

    color = ImageColor.getrgb(color) if isinstance(color, str) else color
    overlay = Image.new("RGB", base.size, color)
    blended = Image.blend(base, overlay, alpha=alpha)
    mask_arr = np.array(mask) > 0
    base_arr = np.array(base)
    blended_arr = np.array(blended)
    base_arr[mask_arr] = blended_arr[mask_arr]
    return Image.fromarray(base_arr)
