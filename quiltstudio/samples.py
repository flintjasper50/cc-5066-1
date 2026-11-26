"""准备三类检验样本 + 一对方向一致但局部锚点互斥的双样片。

所有样本均由固定种子的程序化过程生成；也支持载入用户自制纹理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .imgio import load_image


@dataclass
class Sample:
    id: str
    name: str
    image: np.ndarray  # H×W×3 float32 [0,1]
    role: str = ""
    meta: dict = field(default_factory=dict)
    anchor_mask: np.ndarray | None = None  # H×W bool，可选的真值地标掩膜

    def __post_init__(self):
        self.image = np.ascontiguousarray(self.image, dtype=np.float32)
        if self.anchor_mask is not None:
            self.anchor_mask = np.ascontiguousarray(self.anchor_mask, dtype=bool)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _smooth_noise(rng: np.random.Generator, size: int, low: int, amp: float) -> np.ndarray:
    small = rng.normal(0.0, 1.0, (low, low)).astype(np.float32)
    big = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=size / low / 2.0)
    mx = float(np.abs(big).max()) + 1e-6
    return big / mx * amp


def make_grain(seed: int = 7101, size: int = 256) -> Sample:
    rng = _rng(seed)
    img = np.empty((size, size, 3), np.float32)
    base = np.array([0.62, 0.58, 0.52], np.float32)
    mottle = _smooth_noise(rng, size, 18, 0.10)[..., None]
    tint = _smooth_noise(rng, size, 9, 0.05)[..., None]
    img[:] = base
    img += mottle
    img += tint * np.array([1.0, 0.85, 0.6], np.float32)
    img += rng.normal(0.0, 0.045, (size, size, 3)).astype(np.float32)

    # 随机颗粒：大小/明暗/色相都有变化的上千个点
    palette = [
        (0.16, 0.13, 0.11), (0.92, 0.87, 0.76), (0.55, 0.28, 0.20),
        (0.20, 0.34, 0.42), (0.30, 0.42, 0.28), (0.78, 0.62, 0.30),
    ]
    for _ in range(1500):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.choice([1, 1, 1, 2, 2, 3]))
        color = np.array(palette[int(rng.integers(0, len(palette)))], np.float32)
        color = np.clip(color + rng.normal(0, 0.05, 3), 0, 1)
        cv2.circle(img, (x, y), r, tuple(float(v) for v in color), -1, lineType=cv2.LINE_AA)
    # 少量更大的软颗粒
    for _ in range(60):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(3, 7))
        color = np.array(palette[int(rng.integers(0, len(palette)))], np.float32)
        layer = img.copy()
        cv2.circle(layer, (x, y), r, tuple(float(v) for v in color), -1, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)
    return Sample("grain", "随机颗粒样本", np.clip(img, 0, 1),
                  role="随机颗粒：近似平稳随机纹理，预期缝合容易、重复感低",
                  meta={"seed": seed, "kind": "grain"})


def _stripe_field(size: int, seed: int, angle_deg: float = 28.0, period: float = 34.0):
    rng = _rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    th = np.deg2rad(angle_deg)
    q = xx * np.cos(th) + yy * np.sin(th)
    warp = _smooth_noise(rng, size, 10, 2.2)
    phase = 2 * np.pi * (q + warp) / period
    return rng, q, phase


def make_stripes(seed: int = 7202, size: int = 256) -> Sample:
    rng, q, phase = _stripe_field(size, seed)
    c_dark = np.array([0.10, 0.24, 0.30], np.float32)
    c_mid = np.array([0.16, 0.45, 0.52], np.float32)
    c_light = np.array([0.86, 0.80, 0.62], np.float32)
    t = 0.5 + 0.5 * np.sin(phase)
    img = c_dark + (c_mid - c_dark) * np.clip(t * 1.6, 0, 1)[..., None]
    thin = (np.sin(phase * 2.0 + 1.3) > 0.93).astype(np.float32)[..., None]
    img = img * (1 - 0.55 * thin) + c_light * (0.55 * thin)
    broad = (0.5 + 0.5 * np.sin(phase * 0.5 + 0.7))[..., None]
    img = img * (0.85 + 0.3 * broad)
    img += rng.normal(0, 0.028, img.shape).astype(np.float32)
    # 细颗粒
    for _ in range(350):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        v = float(rng.uniform(0.05, 0.95))
        cv2.circle(img, (x, y), 1, (v, v, v * 0.96), -1)
    return Sample("stripes", "定向条纹样本", np.clip(img, 0, 1),
                  role="方向条纹：方向一致但相位/宽度有变化，检验结构连续性",
                  meta={"seed": seed, "kind": "stripes", "angle_deg": 28.0, "period": 34.0})


def _draw_marker(img: np.ndarray, kind: str, cx: int, cy: int, r: int,
                 color: tuple[float, float, float], rng: np.random.Generator) -> None:
    c = tuple(float(v) for v in color)
    if kind == "ring":
        cv2.circle(img, (cx, cy), r, c, max(3, r // 4), lineType=cv2.LINE_AA)
    elif kind == "disc":
        cv2.circle(img, (cx, cy), r, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(2, r // 5),
                   (1, 1, 1), -1, lineType=cv2.LINE_AA)
    elif kind == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r * 3 // 4],
                        [cx + r, cy + r * 3 // 4]], np.int32)
        cv2.fillPoly(img, [pts], c, lineType=cv2.LINE_AA)
    elif kind == "cross":
        cv2.line(img, (cx - r, cy), (cx + r, cy), c, r // 3, lineType=cv2.LINE_AA)
        cv2.line(img, (cx, cy - r), (cx, cy + r), c, r // 3, lineType=cv2.LINE_AA)
    elif kind == "diamond":
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
        cv2.fillPoly(img, [pts], c, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (0.08, 0.08, 0.08), 2, lineType=cv2.LINE_AA)
    elif kind == "star":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = -np.pi / 2 + k * np.pi / 5
            pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], c, lineType=cv2.LINE_AA)
    elif kind == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), c, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (cx - r // 2, cy - r // 2), (cx + r // 2, cy + r // 2),
                      (0.95, 0.95, 0.9), -1, lineType=cv2.LINE_AA)
    elif kind == "xmark":
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), c, max(3, r // 4), cv2.LINE_AA)
        cv2.line(img, (cx + r, cy - r), (cx - r, cy + r), c, max(3, r // 4), cv2.LINE_AA)


def _marker_silhouette(size: int, specs) -> np.ndarray:
    """按标记的真实笔画/外轮廓生成紧凑真值掩膜（而非整个外接圆）。"""
    m = np.zeros((size, size), np.uint8)
    for kind, cx, cy, r, _ in specs:
        if kind == "ring":
            cv2.circle(m, (cx, cy), r, 1, max(5, r // 4) + 6, lineType=cv2.LINE_AA)
        elif kind == "disc":
            cv2.circle(m, (cx, cy), r + 2, 1, -1, lineType=cv2.LINE_AA)
        elif kind == "triangle":
            pts = np.array([[cx, cy - r - 2], [cx - r - 2, cy + r],
                            [cx + r + 2, cy + r]], np.int32)
            cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_AA)
        elif kind == "cross":
            th = r // 3 + 7
            cv2.line(m, (cx - r - 2, cy), (cx + r + 2, cy), 1, th, cv2.LINE_AA)
            cv2.line(m, (cx, cy - r - 2), (cx, cy + r + 2), 1, th, cv2.LINE_AA)
        elif kind == "diamond":
            pts = np.array([[cx, cy - r - 2], [cx + r + 2, cy],
                            [cx, cy + r + 2], [cx - r - 2, cy]], np.int32)
            cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_AA)
        elif kind == "star":
            pts = []
            for k in range(10):
                rr = r + 2 if k % 2 == 0 else int(r * 0.45)
                a = -np.pi / 2 + k * np.pi / 5
                pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
            cv2.fillPoly(m, [np.array(pts, np.int32)], 1, lineType=cv2.LINE_AA)
        elif kind == "square":
            cv2.rectangle(m, (cx - r - 2, cy - r - 2), (cx + r + 2, cy + r + 2), 1, -1)
        elif kind == "xmark":
            th = max(5, r // 4) + 7
            cv2.line(m, (cx - r, cy - r), (cx + r, cy + r), 1, th, cv2.LINE_AA)
            cv2.line(m, (cx + r, cy - r), (cx - r, cy + r), 1, th, cv2.LINE_AA)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
    return m.astype(bool)


def make_markers(seed: int = 7303, size: int = 288) -> Sample:
    rng = _rng(seed)
    # 背景：弱条纹 + 颗粒，但不提供与独特标记相同的局部结构
    _, _, phase = _stripe_field(size, seed + 1, angle_deg=-18.0, period=26.0)
    t = 0.5 + 0.5 * np.sin(phase)
    img = (np.array([0.30, 0.30, 0.34], np.float32)
           + (0.16 * t)[..., None] * np.array([0.5, 0.55, 0.6], np.float32))
    img += rng.normal(0, 0.05, (size, size, 3)).astype(np.float32)
    for _ in range(900):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        v = float(rng.uniform(0.1, 0.9))
        cv2.circle(img, (x, y), int(rng.choice([1, 1, 2])), (v, v, v), -1)

    specs = [
        ("ring",    46,  52, 24, (0.85, 0.15, 0.12)),
        ("triangle", 150, 48, 24, (0.10, 0.45, 0.95)),
        ("cross",   244, 60, 22, (0.98, 0.80, 0.10)),
        ("diamond",  60, 150, 24, (0.60, 0.20, 0.85)),
        ("star",    160, 150, 26, (0.15, 0.75, 0.35)),
        ("square",  246, 158, 22, (0.95, 0.45, 0.10)),
        ("xmark",    66, 246, 22, (0.10, 0.80, 0.85)),
        ("disc",    164, 244, 22, (0.95, 0.35, 0.65)),
        ("ring",    248, 246, 20, (0.75, 0.90, 0.20)),
    ]
    for kind, cx, cy, r, col in specs:
        _draw_marker(img, kind, cx, cy, r, col, rng)

    anchor_mask = _marker_silhouette(size, specs)
    return Sample("markers", "独特标记样本", np.clip(img, 0, 1),
                   role="明显独特标记：每个大标记全局唯一，强结构下 quilting 可能失败",
                   meta={"seed": seed, "kind": "markers", "markers": len(specs)},
                   anchor_mask=anchor_mask.astype(bool))


def make_conflict_pair(seed: int = 7404, size: int = 288) -> list[Sample]:
    """同方向、同条纹底；A 为右向锚点，B 为左向锚点（互斥）。"""
    rng, q, phase = _stripe_field(size, seed, angle_deg=35.0, period=40.0)
    c_dark = np.array([0.16, 0.20, 0.26], np.float32)
    c_mid = np.array([0.30, 0.52, 0.55], np.float32)
    c_light = np.array([0.88, 0.82, 0.66], np.float32)
    t = 0.5 + 0.5 * np.sin(phase)
    base = c_dark + (c_mid - c_dark) * np.clip(t * 1.5, 0, 1)[..., None]
    thin = (np.sin(phase * 2.0 + 0.4) > 0.94).astype(np.float32)[..., None]
    base = base * (1 - 0.5 * thin) + c_light * (0.5 * thin)
    base += rng.normal(0, 0.025, base.shape).astype(np.float32)

    anchor_positions = [(58, 66), (176, 58), (246, 150), (96, 214), (208, 232)]
    amber = (0.98, 0.55, 0.08)
    blue = (0.16, 0.32, 0.85)

    img_a = base.copy()
    img_b = base.copy()
    mask_a = np.zeros((size, size), np.uint8)
    mask_b = np.zeros((size, size), np.uint8)
    for k, (cx, cy) in enumerate(anchor_positions):
        col = amber if k % 2 == 0 else blue
        cv2.circle(img_a, (cx, cy), 19, (0.08, 0.08, 0.10), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_a, (cx, cy), 17, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(img_b, (cx, cy), 19, (0.08, 0.08, 0.10), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_b, (cx, cy), 17, col, -1, lineType=cv2.LINE_AA)
        # A：右向箭头；B：左向箭头 —— 同一锚点位置上的互斥局部结构
        arrow_r = np.array([
            [cx - 9, cy - 11], [cx + 12, cy], [cx - 9, cy + 11],
            [cx - 3, cy]], np.int32)
        arrow_l = np.array([
            [cx + 9, cy - 11], [cx - 12, cy], [cx + 9, cy + 11],
            [cx + 3, cy]], np.int32)
        cv2.fillPoly(img_a, [arrow_r], (0.97, 0.97, 0.94), lineType=cv2.LINE_AA)
        cv2.fillPoly(img_b, [arrow_l], (0.97, 0.97, 0.94), lineType=cv2.LINE_AA)
        cv2.fillPoly(mask_a, [arrow_r], 1)
        cv2.fillPoly(mask_b, [arrow_l], 1)
    k3 = np.ones((3, 3), np.uint8)
    mask_a = cv2.dilate(mask_a, k3, iterations=2).astype(bool)
    mask_b = cv2.dilate(mask_b, k3, iterations=2).astype(bool)

    a = Sample("conflict_a", "互斥锚点 A（右向箭头）", np.clip(img_a, 0, 1),
               role="与 B 条纹方向完全一致，但局部箭头朝右，和 B 互斥",
               meta={"seed": seed, "kind": "conflict", "anchor": "right"},
               anchor_mask=mask_a)
    b = Sample("conflict_b", "互斥锚点 B（左向箭头）", np.clip(img_b, 0, 1),
               role="与 A 条纹方向完全一致，但局部箭头朝左，和 A 互斥",
               meta={"seed": seed, "kind": "conflict", "anchor": "left"},
               anchor_mask=mask_b)
    return [a, b]


def make_diagonal(seed: int = 7505, size: int = 256,
                  angle_deg: float = 45.0, period: float = 22.0) -> Sample:
    """斜纹原图：颜色沿固定对角方向渐变，便于逐像素回读来源验证。

    斜纹的相位随坐标单调变化，因此“某像素取自样本的哪个坐标”可以被
    反查校验：沿接缝两侧各取一行，按 map_y/map_x 回读样本颜色，必须与
    画布像素逐值相等，否则说明修补过程暗中改写了像素（模糊/覆盖）。
    """
    rng = _rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    th = np.deg2rad(angle_deg)
    q = xx * np.cos(th) + yy * np.sin(th)
    # 轻微弯曲让纹理不至于完全平直，但保持单调可回读
    warp = _smooth_noise(rng, size, 8, 1.4)
    phase = 2 * np.pi * (q + warp) / period
    t = 0.5 + 0.5 * np.sin(phase)
    c0 = np.array([0.12, 0.18, 0.26], np.float32)
    c1 = np.array([0.86, 0.72, 0.42], np.float32)
    img = c0 + (c1 - c0) * t[..., None]
    ridge = (np.sin(phase * 3.0 + 0.9) > 0.86).astype(np.float32)[..., None]
    img = img * (1 - 0.35 * ridge) + np.array([0.95, 0.93, 0.88], np.float32) * (0.35 * ridge)
    img += rng.normal(0, 0.018, img.shape).astype(np.float32)
    return Sample("diagonal", "斜纹原图", np.clip(img, 0, 1),
                  role="斜纹：用于沿接缝回读像素来源，验证修补未改锁边",
                  meta={"seed": seed, "kind": "diagonal",
                        "angle_deg": angle_deg, "period": period})


def build_prepared() -> dict[str, list[Sample]]:
    return {
        "grain": [make_grain()],
        "stripes": [make_stripes()],
        "markers": [make_markers()],
        "conflict": make_conflict_pair(),
        "diagonal": [make_diagonal()],
    }


def sample_from_file(path: str | Path, sample_id: str | None = None,
                     name: str | None = None, max_side: int = 320) -> Sample:
    p = Path(path)
    img = load_image(p, max_side=max_side)
    return Sample(sample_id or p.stem, name or f"用户样本：{p.name}", img,
                  role="user", meta={"source_file": str(p)})
