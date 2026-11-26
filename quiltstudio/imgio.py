"""图像读写与数组/PNG 转换辅助。约定所有图像为 float32 RGB, 范围 [0, 1]。"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_side is not None and max(im.size) > max_side:
        w, h = im.size
        scale = max_side / max(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


def save_png(arr: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(arr, 0.0, 1.0)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    img = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "RGB")
    img.save(path)
    return path


def save_index_png(index: np.ndarray, path: str | Path) -> Path:
    """H×W 整数图编码到 RGB(低 8 位 R, 高 8 位 G), 供前端精确取 id。"""
    index = index.astype(np.int64)
    r = (index & 255).astype(np.uint8)
    g = ((index >> 8) & 255).astype(np.uint8)
    b = ((index >> 16) & 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(path)
    return path


def encode_png_data_url(arr: np.ndarray) -> str:
    a = np.clip(arr, 0.0, 1.0)
    buf = io.BytesIO()
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
