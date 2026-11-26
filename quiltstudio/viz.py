"""可视化产物：接缝路径叠加、来源追溯图、tile-id 编码图、图块墙、对比板。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .imgio import save_index_png, save_png

KIND_COLORS = {
    "left": (34, 211, 238),
    "top": (250, 204, 21),
    "close_right": (244, 114, 182),
    "close_bottom": (251, 146, 60),
}

SAMPLE_PALETTE = [
    (230, 57, 70), (29, 155, 240), (42, 157, 143), (244, 162, 97),
    (155, 93, 229), (128, 200, 80),
]


def seam_segments(points: list[list[int]], w: int, h: int,
                  periodic: bool) -> list[list[list[int]]]:
    """把非折绕坐标路径转成显示坐标，并在环面折回处拆成多段。"""
    segs: list[list[list[int]]] = []
    cur: list[list[int]] = []
    last = None
    for x, y in points:
        dx = x % w if periodic else x
        dy = y % h if periodic else y
        if last is not None and math.hypot(dx - last[0], dy - last[1]) > 3:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
        cur.append([int(dx), int(dy)])
        last = (dx, dy)
    if len(cur) >= 2:
        segs.append(cur)
    return segs


def all_seam_segments(placements: list[dict], w: int, h: int,
                      periodic: bool) -> list[dict]:
    out = []
    for p in placements:
        for r in p.get("regions", []):
            segs = seam_segments(r["points_unwrapped"], w, h, periodic)
            for s in segs:
                out.append({"placement": p["id"], "kind": r["kind"],
                            "color": "#%02x%02x%02x" % KIND_COLORS[r["kind"]],
                            "points": s})
    return out


def render_seams_overlay(result: np.ndarray, placements: list[dict],
                         periodic: bool) -> np.ndarray:
    h, w = result.shape[:2]
    base = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    dark = Image.new("RGB", (w, h), (10, 12, 18))
    base = Image.blend(base, dark, 0.35)
    dr = ImageDraw.Draw(base)
    for p in placements:
        for r in p.get("regions", []):
            color = KIND_COLORS[r["kind"]]
            for seg in seam_segments(r["points_unwrapped"], w, h, periodic):
                if len(seg) >= 2:
                    dr.line([tuple(pt) for pt in seg], fill=color, width=2)
            # 高误差 placement 加红框警示点
        if p.get("flags"):
            ax, ay = p["anchor"]
            dr.rectangle([ax - 2, ay - 2, ax + 8, ay + 8], outline=(255, 60, 60), width=2)
    return np.asarray(base).astype(np.float32) / 255.0


def render_provenance_id(map_sample: np.ndarray, n_samples: int,
                         own: np.ndarray | None = None) -> np.ndarray:
    h, w = map_sample.shape
    out = np.zeros((h, w, 3), np.float32)
    for sid in range(-1, n_samples):
        if sid < 0:
            color = np.array([0.08, 0.08, 0.10])
        else:
            c = SAMPLE_PALETTE[sid % len(SAMPLE_PALETTE)]
            color = np.array(c, np.float32) / 255.0
        out[map_sample == sid] = color
    if own is not None:
        # 同样片时整片同色无法分辨图块：按 tile id 做确定性明暗调制，
        # 色相仍编码 sample 来源。
        o = own.astype(np.uint32)
        hashed = (o * np.uint32(2654435761)) % np.uint32(97)
        factor = 0.72 + 0.42 * hashed.astype(np.float32) / 96.0
        factor = np.where(own >= 0, factor, 0.5)[..., None]
        out = np.clip(out * factor, 0, 1)
    return out


def render_provenance_xy(map_y: np.ndarray, map_x: np.ndarray,
                         ref_h: int, ref_w: int) -> np.ndarray:
    r = np.clip(map_x.astype(np.float32) / max(1, ref_w - 1), 0, 1)
    g = np.clip(map_y.astype(np.float32) / max(1, ref_h - 1), 0, 1)
    b = np.full_like(r, 0.35)
    invalid = map_x < 0
    out = np.stack([r, g, b], axis=-1)
    out[invalid] = 0
    return out


def save_tileid(own: np.ndarray, path) -> None:
    save_index_png(own + 1, path)  # 0 保留为空，id 从 1 开始


def wrapped_rects(ax: int, ay: int, b: int, w: int, h: int,
                  periodic: bool) -> list[dict]:
    if not periodic:
        return [{"x": int(ax), "y": int(ay), "w": b, "h": b}]
    xs = [(ax, min(ax + b, w))]
    if ax + b > w:
        xs.append((0, ax + b - w))
    ys = [(ay, min(ay + b, h))]
    if ay + b > h:
        ys.append((0, ay + b - h))
    rects = []
    for x0, x1 in xs:
        for y0, y1 in ys:
            if x1 > x0 and y1 > y0:
                rects.append({"x": int(x0), "y": int(y0),
                              "w": int(x1 - x0), "h": int(y1 - y0)})
    return rects


def render_tile_grid(result: np.ndarray, placements: list[dict],
                     b: int, periodic: bool) -> np.ndarray:
    h, w = result.shape[:2]
    im = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay)
    for p in placements:
        for rc in p["rects"]:
            x, y, rw, rh = rc["x"], rc["y"], rc["w"], rc["h"]
            hot = bool(p.get("flags"))
            col = (255, 70, 70, 230) if hot else (255, 255, 255, 150)
            dr.rectangle([x, y, x + rw - 1, y + rh - 1], outline=col, width=1)
            dr.text((x + 2, y + 1), str(p["id"]), fill=(255, 230, 80, 255))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def build_patch_wall(samples, placements: list[dict], b: int, path) -> None:
    n = len(placements)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    label_h = 16
    cell = b + label_h
    wall = Image.new("RGB", (cols * cell + 4, rows * cell + 4), (24, 26, 34))
    dr = ImageDraw.Draw(wall)
    for k, p in enumerate(placements):
        cx = (k % cols) * cell + 2
        cy = (k // cols) * cell + 2
        sid = p["chosen"]["sample_id"]
        sy, sx = p["chosen"]["source_y"], p["chosen"]["source_x"]
        patch = samples[sid].image[sy:sy + b, sx:sx + b]
        tim = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
        wall.paste(tim, (cx, cy + label_h))
        dr.text((cx + 1, cy + 2),
                f"#{p['id']} s{sid} {sy},{sx}", fill=(235, 235, 245))
    wall.save(path)


def _panel(img01: np.ndarray, title: str, target_h: int = 300) -> Image.Image:
    im = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
    scale = target_h / im.height
    im = im.resize((max(1, int(im.width * scale)), target_h), Image.NEAREST)
    bar = 26
    panel = Image.new("RGB", (im.width + 8, target_h + bar + 6), (18, 20, 28))
    d = ImageDraw.Draw(panel)
    d.text((6, 6), title, fill=(240, 240, 245))
    panel.paste(im, (4, bar))
    d.rectangle([3, bar - 1, im.width + 4, bar + target_h], outline=(90, 100, 120))
    return panel


def build_comparison(path, panels: list[tuple[str, np.ndarray]], target_h: int = 300) -> None:
    rendered = [_panel(img, title, target_h) for title, img in panels]
    gap = 14
    total_w = sum(p.width for p in rendered) + gap * (len(rendered) + 1)
    max_h = max(p.height for p in rendered) + gap * 2
    board = Image.new("RGB", (total_w, max_h), (10, 12, 18))
    x = gap
    for p in rendered:
        board.paste(p, (x, gap))
        x += p.width + gap
    board.save(path)


def save_all_core_images(run_dir, result: np.ndarray, seams_img: np.ndarray,
                         prov_id: np.ndarray, prov_xy: np.ndarray,
                         grid_img: np.ndarray) -> dict:
    paths = {}
    for name, arr in [("result", result), ("seams", seams_img),
                      ("provenance_id", prov_id), ("provenance_xy", prov_xy),
                      ("tile_grid", grid_img)]:
        p = run_dir / f"{name}.png"
        save_png(arr, p)
        paths[name] = str(p)
    return paths
