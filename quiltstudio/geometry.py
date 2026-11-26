"""图块网格几何：锚点、互不重叠的重叠区域划分与覆盖自检。

约定（局部 patch 坐标，半开区间）：
- 每个图块精确覆盖画布上一个 B×B 方块；非周期时贴边图块不越界。
- 一个图块的重叠区域按“左/上/右闭环/下闭环”四类条带划分，四类条带
  互不重叠，且并集之外的部分就是该图块无条件写入的新区域。
- 周期模式下，只有最后一列/行可能越过画布右/下边界，越过部分环绕到
  画布另一侧，且对应像素必定已被第 0 列/行图块写过。
"""
from __future__ import annotations

import numpy as np


def _legacy_anchors(length: int, patch: int, overlap: int,
                    periodic: bool) -> list[int]:
    """历史锚点算法，逐字保留以便已入库 run 完全可复现（不要修改）。"""
    step = patch - overlap
    if periodic:
        n = max(2, int(round(length / step)))
        return [int(round(i * length / n)) for i in range(n)]
    if length <= patch:
        return [0]
    span = length - patch
    n = max(2, int(round(span / step)) + 1)
    return sorted({int(round(i * span / (n - 1))) for i in range(n)})


def _robust_anchors(length: int, patch: int, overlap: int,
                    periodic: bool) -> list[int]:
    """安全锚点：相邻间距（含周期环绕）被构造性地限制在 (0, patch]。"""
    step = max(1, patch - overlap)
    if length <= patch:
        return [0] if not periodic else [0, max(1, length // 2)]
    if periodic:
        n = max(2, int(np.ceil(length / step)))
        return [int(round(i * length / n)) for i in range(n)]
    span = length - patch
    n = max(2, int(np.ceil(span / step)) + 1)
    n = min(n, span + 1)
    return [int(round(i * span / (n - 1))) for i in range(n)]


def build_anchors(length: int, patch: int, overlap: int, periodic: bool) -> list[int]:
    """生成图块左上角坐标；不健康的历史结果自动回退到安全算法。

    历史算法在非整除画布 / 窄重叠下可能给出大于 patch 的间距，导致重叠
    宽度为负、区域越界、覆盖留孔。这里先验证历史结果，健康则原样返回
    （保证已入库实验逐像素可复现），否则改用安全构造。
    """
    if patch <= 0:
        raise ValueError("patch 必须为正")
    if length <= patch and not periodic:
        return [0]
    legacy = _legacy_anchors(length, patch, overlap, periodic)
    if anchor_gaps(length, legacy, patch, periodic)["ok"]:
        return legacy
    return _robust_anchors(length, patch, overlap, periodic)


def anchor_gaps(length: int, xs: list[int], patch: int, periodic: bool) -> dict:
    """报告锚点间距的健康度：最大间距不得超过 patch，否则必有孔洞。"""
    if len(xs) < 2:
        return {"max_gap": 0, "wrap_gap": 0, "ok": True, "gaps": []}
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    wrap = (xs[0] + length - xs[-1]) if periodic else 0
    ok = all(0 < g <= patch for g in gaps) and (not periodic or 0 < wrap <= patch)
    return {"max_gap": max(gaps) if gaps else 0,
            "wrap_gap": wrap, "ok": bool(ok), "gaps": gaps}


def build_regions(ix: int, iy: int, xs: list[int], ys: list[int],
                  w: int, h: int, b: int, periodic: bool) -> list[dict]:
    """构造互不相交的重叠条带（局部坐标，半开区间）。

    条带只有在确实存在“已写邻块”时才生成；周期闭环只在越过画布边界时
    生成。返回的矩形全部位于 [0,b)×[0,b) 内且两两不重叠。
    """
    ax, ay = int(xs[ix]), int(ys[iy])
    last_x = ix == len(xs) - 1
    last_y = iy == len(ys) - 1

    lw = b - (ax - xs[ix - 1]) if ix > 0 else 0
    th = b - (ay - ys[iy - 1]) if iy > 0 else 0
    rw = max(0, ax + b - w) if (periodic and last_x) else 0
    bh = max(0, ay + b - h) if (periodic and last_y) else 0

    # 归一化配置保证 overlap <= patch/2-2，因此左右/上下条带不会相互吃掉。
    lw = max(0, min(b, lw))
    th = max(0, min(b, th))
    rw = max(0, min(b - lw, rw))
    bh = max(0, min(b - th, bh))

    mid_c0, mid_c1 = lw, b - rw
    mid_r0, mid_r1 = th, b - bh

    regs: list[dict] = []
    if lw > 0:
        regs.append({"kind": "left", "axis": "v", "orient": "old_left",
                     "r0": 0, "r1": b, "c0": 0, "c1": lw})
    if rw > 0:
        regs.append({"kind": "close_right", "axis": "v", "orient": "new_left",
                     "r0": 0, "r1": b, "c0": b - rw, "c1": b})
    if th > 0 and mid_c1 > mid_c0:
        regs.append({"kind": "top", "axis": "h", "orient": "old_top",
                     "r0": 0, "r1": th, "c0": mid_c0, "c1": mid_c1})
    if bh > 0 and mid_c1 > mid_c0:
        regs.append({"kind": "close_bottom", "axis": "h", "orient": "new_top",
                     "r0": b - bh, "r1": b, "c0": mid_c0, "c1": mid_c1})
    for r in regs:
        r["core_r"] = (mid_r0, mid_r1)
        r["core_c"] = (mid_c0, mid_c1)
    return regs


def region_coords(reg: dict, ax: int, ay: int, w: int, h: int,
                  periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    rows = ay + np.arange(reg["r0"], reg["r1"])
    cols = ax + np.arange(reg["c0"], reg["c1"])
    if periodic:
        rows = rows % h
        cols = cols % w
    return rows.astype(np.int64), cols.astype(np.int64)


def partition_report(regs: list[dict], b: int) -> dict:
    """检查区域是否全部在界内、互不重叠，并给出新区域面积。"""
    occ = np.zeros((b, b), np.int32)
    oob = 0
    for r in regs:
        r0, r1 = max(0, r["r0"]), min(b, r["r1"])
        c0, c1 = max(0, r["c0"]), min(b, r["c1"])
        oob += (r["r1"] - r["r0"]) * (r["c1"] - r["c0"]) - max(0, r1 - r0) * max(0, c1 - c0)
        if r1 > r0 and c1 > c0:
            occ[r0:r1, c0:c1] += 1
    union = int((occ > 0).sum())
    overlap = int((occ > 1).sum())
    return {"out_of_bounds": int(oob),
            "overlap_pixels": overlap,
            "covered_pixels": union,
            "new_pixels": int(b * b - union),
            "disjoint": overlap == 0}
