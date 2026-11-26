"""生成质量与“失败可见性”指标：真实接缝误差、周期边界、重复痕迹、方向结构。"""
from __future__ import annotations

import cv2
import numpy as np


def ownership_seam_rmse(img: np.ndarray, own: np.ndarray, periodic: bool) -> dict:
    """沿所有 tile 归属发生变化的相邻像素测量真实颜色差（RMSE, 0..1）。

    这是合成后画布上的实际可见接缝误差，不依赖 DP 内部代价。
    periodic=True 时额外纳入 W-1|0、H-1|0 两列首尾邻接。
    """
    total_sse = 0.0
    total_n = 0
    b_sse = 0.0
    b_n = 0

    def accumulate(diff_sq: np.ndarray, mask: np.ndarray, boundary: bool = False):
        nonlocal total_sse, total_n, b_sse, b_n
        m = mask
        n = int(m.sum())
        if n:
            sse = float(diff_sq[m].sum())
            total_sse += sse
            total_n += n * 3
            if boundary:
                b_sse += sse
                b_n += n * 3

    # 水平相邻 (x, x+1)
    d = img[:, 1:, :] - img[:, :-1, :]
    ds = (d * d).sum(axis=2)
    accumulate(ds, own[:, 1:] != own[:, :-1])
    # 垂直相邻 (y, y+1)
    d = img[1:, :, :] - img[:-1, :, :]
    ds = (d * d).sum(axis=2)
    accumulate(ds, own[1:, :] != own[:-1, :])

    if periodic:
        d = img[:, 0, :] - img[:, -1, :]
        ds = (d * d).sum(axis=1)
        accumulate(ds, own[:, 0] != own[:, -1], boundary=True)
        d = img[0, :, :] - img[-1, :, :]
        ds = (d * d).sum(axis=1)
        accumulate(ds, own[0, :] != own[-1, :], boundary=True)

    rmse = float(np.sqrt(total_sse / max(1, total_n)))
    out = {"seam_rmse": rmse, "seam_pair_count": int(total_n / 3)}
    if periodic:
        out["closure_seam_rmse"] = float(np.sqrt(b_sse / max(1, b_n))) if b_n else 0.0
        out["closure_pair_count"] = int(b_n / 3)
    return out


def natural_adjacent_rmse(img: np.ndarray) -> float:
    """原样本内部正常相邻像素的 RMSE：高频颗粒的接缝误差应与此地板比较。"""
    dx = img[:, 1:, :] - img[:, :-1, :]
    dy = img[1:, :, :] - img[:-1, :, :]
    sse = float((dx * dx).sum() + (dy * dy).sum())
    n = dx.size + dy.size
    return float(np.sqrt(sse / max(1, n)))


def periodic_boundary_rmse(img: np.ndarray) -> dict:
    """不依赖归属图：直接测量首尾行/列的颜色跳变。"""
    dx = img[:, 0, :] - img[:, -1, :]
    dy = img[0, :, :] - img[-1, :, :]
    rx = float(np.sqrt((dx * dx).mean()))
    ry = float(np.sqrt((dy * dy).mean()))
    return {"edge_lr_rmse": rx, "edge_tb_rmse": ry,
            "edge_rmse": float(np.sqrt((rx * rx + ry * ry) / 2))}


def tiling_baseline_rmse(sample: np.ndarray, out_w: int, out_h: int) -> dict:
    """直接平铺整张原样本时，重复边界上的接缝误差（对照基线）。"""
    hs, ws = sample.shape[:2]
    reps_y = int(np.ceil(out_h / hs)) + 1
    reps_x = int(np.ceil(out_w / ws)) + 1
    canvas = np.tile(sample, (reps_y, reps_x, 1))[:out_h, :out_w]
    sse = 0.0
    n = 0
    for x in range(ws - 1, out_w - 1, ws):
        if x + 1 < out_w:
            d = canvas[:, x + 1, :] - canvas[:, x, :]
            sse += float((d * d).sum())
            n += d.size
    for y in range(hs - 1, out_h - 1, hs):
        if y + 1 < out_h:
            d = canvas[y + 1, :, :] - canvas[y, :, :]
            sse += float((d * d).sum())
            n += d.size
    return {"naive_tile_seam_rmse": float(np.sqrt(sse / max(1, n))), "canvas": canvas}


def tile_reuse(placements: list[dict]) -> dict:
    keys = [(p["chosen"]["sample_id"], p["chosen"]["source_y"],
             p["chosen"]["source_x"]) for p in placements]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    reused = {f"{k[0]}@{k[1]},{k[2]}": v for k, v in counts.items() if v > 1}
    max_reuse = max(counts.values()) if counts else 0

    # 来源 footprint 互相重叠比例（同一素材内），衡量近似复制
    b = placements[0].get("patch_size", 0) if placements else 0
    max_iou = 0.0
    high_overlap_pairs = 0
    by_sample: dict[int, list[dict]] = {}
    for p in placements:
        by_sample.setdefault(p["chosen"]["sample_id"], []).append(p)
    for ps in by_sample.values():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, c = ps[i]["chosen"], ps[j]["chosen"]
                ox = max(0, min(a["source_x"] + b, c["source_x"] + b)
                         - max(a["source_x"], c["source_x"]))
                oy = max(0, min(a["source_y"] + b, c["source_y"] + b)
                         - max(a["source_y"], c["source_y"]))
                inter = ox * oy
                if inter > 0:
                    union = 2 * b * b - inter
                    iou = inter / union
                    max_iou = max(max_iou, iou)
                    if iou > 0.35:
                        high_overlap_pairs += 1
    return {"distinct_source_tiles": len(counts),
            "placements": len(keys),
            "max_exact_reuse": int(max_reuse),
            "reused_tile_keys": reused,
            "high_source_overlap_pairs": int(high_overlap_pairs),
            "max_source_iou": float(max_iou)}


def autocorrelation_peak(img: np.ndarray, min_lag: int = 16) -> dict:
    """线性（补零）自相关主峰，捕捉成块复制导致的周期性重复痕迹。"""
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    g -= g.mean()
    h, w = g.shape
    f = np.fft.rfft2(g, s=(2 * h, 2 * w))
    ac = np.fft.irfft2(f * np.conj(f), s=(2 * h, 2 * w))
    ac = ac[:h, :w]
    ac /= max(ac[0, 0], 1e-6)
    ac[0, 0] = 0
    # 只看正位移 (dy>=0, dx>=0)，避开零峰邻域
    ac[:min_lag, :min_lag] = 0
    idx = int(np.argmax(ac))
    dy, dx = np.unravel_index(idx, ac.shape)
    return {"autocorr_peak": float(ac[dy, dx]), "autocorr_lag": [int(dx), int(dy)]}


def patch_duplicate_rate(img: np.ndarray, seed: int = 1234,
                        pairs: int = 3000, patch: int = 20) -> dict:
    """随机抽输出图块两两比较，低差对比例越高说明自我重复越严重。"""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if h <= patch + 2 or w <= patch + 2:
        return {"dup_pair_rate": 0.0}
    ys = rng.integers(0, h - patch, size=pairs * 2)
    xs = rng.integers(0, w - patch, size=pairs * 2)
    a = np.stack([img[y:y + patch, x:x + patch]
                  for y, x in zip(ys[:pairs], xs[:pairs])])
    b = np.stack([img[y:y + patch, x:x + patch]
                  for y, x in zip(ys[pairs:], xs[pairs:])])
    rmse = np.sqrt(((a - b) ** 2).mean(axis=(1, 2, 3)))
    rate = float((rmse < 0.045).mean())
    return {"dup_pair_rate": rate, "dup_pair_rmse_median": float(np.median(rmse))}


def feature_cut_metrics(samples, map_sample, map_y, map_x, own, periodic) -> dict:
    """统计归属接缝切过真值“锚点掩膜”的次数（仅对带 anchor_mask 的样本有效）。

    独特标记/互斥箭头这类强语义结构，即使颜色 RMSE 不高，被接缝从中间切开
    也仍是语义损坏。掩膜由样本生成器给出真值，经逐像素来源图映射回输出，
    再沿 ownership 边界统计。用户上传样本没有掩膜时 available=False。
    """
    result = {
        "available": False,
        "anchored_samples": 0,
        "seam_pairs": 0,
        "anchor_inside_hits": 0,
        "anchor_edge_hits": 0,
        "anchor_hit_total": 0,
        "anchor_hit_fraction": 0.0,
        "anchor_seam_density": 0.0,
    }
    masks = []
    any_mask = False
    for s in samples:
        m = getattr(s, "anchor_mask", None)
        if m is not None:
            masks.append(np.asarray(m, bool))
            any_mask = True
        else:
            masks.append(None)
    if not any_mask:
        return result

    h, w = own.shape
    anchor = np.zeros((h, w), bool)
    cut_pixels = np.zeros((h, w), np.uint8)
    valid = map_sample >= 0
    for sid, mask in enumerate(masks):
        if mask is None:
            continue
        sel = valid & (map_sample == sid)
        if not sel.any():
            continue
        ys = map_y[sel]
        xs = map_x[sel]
        mh, mw = mask.shape
        ok = (ys >= 0) & (ys < mh) & (xs >= 0) & (xs < mw)
        pos = np.where(sel)
        anchor[pos[0][ok], pos[1][ok]] = mask[ys[ok], xs[ok]]
        result["anchored_samples"] += 1

    inside = 0
    edge = 0
    pairs = 0

    hcut = (own[:, 1:] != own[:, :-1])
    pairs += int(hcut.sum())
    hi = hcut & anchor[:, 1:] & anchor[:, :-1]
    he = hcut & (anchor[:, 1:] != anchor[:, :-1])
    inside += int(hi.sum())
    edge += int(he.sum())
    cut_pixels[:, :-1] |= hi
    cut_pixels[:, 1:] |= hi

    vcut = (own[1:, :] != own[:-1, :])
    pairs += int(vcut.sum())
    vi = vcut & anchor[1:, :] & anchor[:-1, :]
    ve = vcut & (anchor[1:, :] != anchor[:-1, :])
    inside += int(vi.sum())
    edge += int(ve.sum())
    cut_pixels[:-1, :] |= vi
    cut_pixels[1:, :] |= vi

    if periodic:
        pcut = own[:, 0] != own[:, -1]
        pairs += int(pcut.sum())
        pi = pcut & anchor[:, 0] & anchor[:, -1]
        pe = pcut & (anchor[:, 0] != anchor[:, -1])
        inside += int(pi.sum())
        edge += int(pe.sum())
        rows = np.where(pi)[0]
        cut_pixels[rows, 0] = 1
        cut_pixels[rows, w - 1] = 1

        pcut = own[0, :] != own[-1, :]
        pairs += int(pcut.sum())
        pi = pcut & anchor[0, :] & anchor[-1, :]
        pe = pcut & (anchor[0, :] != anchor[-1, :])
        inside += int(pi.sum())
        edge += int(pe.sum())
        cols = np.where(pi)[0]
        cut_pixels[0, cols] = 1
        cut_pixels[h - 1, cols] = 1

    # 把沿接缝连续的内部命中聚成“锚点被切开事件”，并处理周期边界的跨边连通
    n_ev, labels, stats, _ = cv2.connectedComponentsWithStats(cut_pixels, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_ev > 1 else np.array([], np.int32)
    if periodic and n_ev > 1:
        top = set(np.unique(labels[0, :][cut_pixels[0, :] > 0])) - {0}
        bot = set(np.unique(labels[h - 1, :][cut_pixels[h - 1, :] > 0])) - {0}
        left = set(np.unique(labels[:, 0][cut_pixels[:, 0] > 0])) - {0}
        right = set(np.unique(labels[:, w - 1][cut_pixels[:, w - 1] > 0])) - {0}
        wrap_pairs = [(top, bot), (left, right)]
        merged = []
        for a_set, b_set in wrap_pairs:
            for a in a_set:
                for b in b_set:
                    if a != b:
                        merged.append((a, b))
        parent = list(range(n_ev))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in merged:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        if merged:
            root_area: dict[int, int] = {}
            for lab in range(1, n_ev):
                root_area[find(lab)] = root_area.get(find(lab), 0) + int(areas[lab - 1])
            areas = np.array(sorted(root_area.values(), reverse=True), np.int32)
    event_areas = sorted((int(a) for a in areas), reverse=True)
    cut_events = sum(1 for a in event_areas if a >= 12)
    severe_events = sum(1 for a in event_areas if a >= 40)
    area_100k = max(1, h * w) / 100000.0

    hits = inside + edge
    result.update({
        "available": True,
        "anchored_samples": result["anchored_samples"],
        "seam_pairs": pairs,
        "anchor_inside_hits": inside,
        "anchor_edge_hits": edge,
        "anchor_hit_total": hits,
        # 全部接缝邻接对中，切到锚点的比例
        "anchor_hit_fraction": float(hits / max(1, pairs)),
        # 每个接缝对落在锚点区域内的密度指标（内部双端命中更严重）
        "anchor_seam_density": float((2 * inside + edge) / max(1, 2 * int(anchor.sum()))),
        "anchor_output_pixels": int(anchor.sum()),
        # 离散事件：一条连续接缝穿过同一锚点内部 = 一次切割
        "anchor_cut_events": int(cut_events),
        "anchor_severe_events": int(severe_events),
        "cut_events_per_100kpx": float(cut_events / area_100k),
        "severe_events_per_100kpx": float(severe_events / area_100k),
        "anchor_event_areas": event_areas[:60],
    })
    return result


def orientation_structure(img: np.ndarray) -> dict:
    """结构张量方向相干性：条纹样本应接近 1，随机颗粒应明显更低。"""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 3.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 3.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 3.0)
    denom = jxx + jyy + 1e-6
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / denom
    angle = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    mag = np.sqrt(gx * gx + gy * gy)
    strong = mag > float(np.percentile(mag, 70))
    if strong.sum() > 50:
        # 梯度方向与条纹法线一致；这里返回结构张量主方向
        mean_angle = float(np.median(angle[strong]))
        mean_coh = float(coh[strong].mean())
    else:
        mean_angle, mean_coh = 0.0, float(coh.mean())
    return {"orientation_coherence": mean_coh, "orientation_deg": mean_angle}
