"""受约束纹理扩展内核：框选样本区 → 受保护种子 → 向外拼接扩展 → 自检。

驱动的场景（与回归测试一一对应）：
- 框选纹理样本并扩展：把样本里的一个矩形作为种子铺到画布，保护其像素，
  再向外用图块拼接把画布扩展到目标尺寸。
- 锁一条不能重写的细边：锁定边像素在扩展前后必须逐位相等。
- 接缝禁区：可截断某条接缝内部（局部绕开）；若把端点围死则连同邻块重选，
  候选池都无解时报告无解，绝不静默改写受保护区域。
- 沿缝两侧回读像素来源：用斜纹原图验证局部修补没有暗中改掉锁边。
- 覆盖完整性：扩展后画布不得留孔，受保护区域像素恒等。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .constraints import ForbiddenZone, LockedEdge, Pin, Rect, SeamConstraints
from .engine import QuiltConfig, QuiltEngine, normalize_config
from .samples import Sample

SEED_OWN_BASE = 1_000_000     # 种子像素在 own 图中的 tile id 基数


@dataclass
class ExtendSpec:
    out_w: int = 384
    out_h: int = 384
    patch: int = 64
    overlap: int = 24
    seed: int = 20261126
    periodic: bool = False
    policy: str = "adaptive"
    # 框选：样本坐标矩形 + 落位到画布的左上角
    seed_rect: tuple[int, int, int, int] | None = None   # (x0,y0,x1,y1) 样本坐标
    place_at: tuple[int, int] = (0, 0)                   # 画布坐标左上角
    # 可选：锁定细边（画布坐标矩形），不能重写
    lock_edge: tuple[int, int, int, int] | None = None
    # 可选：接缝禁区（画布坐标矩形）
    forbidden: list[tuple[int, int, int, int]] = field(default_factory=list)
    # 可选：让某个条带接缝端点借用锁定边端点
    borrow_endpoint: bool = False


def _rect(t):
    return Rect(int(t[0]), int(t[1]), int(t[2]), int(t[3])) if t else None


def build_seed(samples: list[Sample], spec: ExtendSpec,
               cfg: QuiltConfig) -> tuple[dict, dict]:
    """把框选样本区铺成受保护种子，返回 (seed_payload, info)。"""
    if spec.seed_rect is None:
        raise ValueError("扩展必须先框选一个样本区域")
    s = samples[0]
    x0, y0, x1, y1 = spec.seed_rect
    x0 = max(0, min(s.image.shape[1], int(x0)))
    x1 = max(0, min(s.image.shape[1], int(x1)))
    y0 = max(0, min(s.image.shape[0], int(y0)))
    y1 = max(0, min(s.image.shape[0], int(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("框选区域为空")
    region = s.image[y0:y1, x0:x1]

    H, W = cfg.out_h, cfg.out_w
    canvas = np.zeros((H, W, 3), np.float32)
    own = np.full((H, W), -1, np.int32)
    map_sample = np.full((H, W), -1, np.int32)
    map_y = np.full((H, W), -1, np.int32)
    map_x = np.full((H, W), -1, np.int32)
    filled = np.zeros((H, W), bool)

    px, py = int(spec.place_at[0]), int(spec.place_at[1])
    sw, sh = region.shape[1], region.shape[0]
    cx1 = min(W, px + sw)
    cy1 = min(H, py + sh)
    if cx1 <= px or cy1 <= py:
        raise ValueError("框选落位超出画布")
    rw, rh = cx1 - px, cy1 - py
    canvas[py:cy1, px:cx1] = region[:rh, :rw]
    own[py:cy1, px:cx1] = SEED_OWN_BASE
    map_sample[py:cy1, px:cx1] = 0
    map_y[py:cy1, px:cx1] = y0 + np.arange(rh)[:, None]
    map_x[py:cy1, px:cx1] = x0 + np.arange(rw)[None, :]
    filled[py:cy1, px:cx1] = True

    payload = {"canvas": canvas, "own": own, "map_sample": map_sample,
               "map_y": map_y, "map_x": map_x, "filled": filled,
               "protect": filled.copy(), "placements": [],
               "info": {"sample_id": s.id, "sample_name": s.name,
                        "seed_rect": [x0, y0, x1, y1],
                        "place_at": [px, py],
                        "filled_pixels": int(filled.sum())}}
    return payload, payload["info"]


def _seed_placements(info: dict, cfg: QuiltConfig) -> list[dict]:
    """把种子外接矩形折算成“已保护图块”清单，供引擎跳过重解。"""
    x0, y0, x1, y1 = info["seed_rect"]
    px, py = info["place_at"]
    rw = min(cfg.out_w - px, x1 - x0)
    rh = min(cfg.out_h - py, y1 - y0)
    return [{"seed": True, "rect": [px, py, px + rw, py + rh]}]


def build_constraints(samples: list[Sample], spec: ExtendSpec,
                      seed_info: dict) -> tuple[SeamConstraints, list[LockedEdge]]:
    cons = SeamConstraints()
    locking: list[LockedEdge] = []
    if spec.lock_edge is not None:
        r = _rect(spec.lock_edge)
        le = LockedEdge(rect=r, name="locked-edge")
        cons.locked.append(le)
        locking.append(le)
    for i, fr in enumerate(spec.forbidden):
        r = _rect(fr)
        if r.w > 0 and r.h > 0:
            cons.forbidden.append(ForbiddenZone(rect=r, name=f"forbid-{i}"))
    return cons, locking


def extend(samples: list[Sample], spec: ExtendSpec) -> dict:
    """执行一次受约束纹理扩展，返回结果与逐项自检。"""
    cfg = QuiltConfig(out_w=spec.out_w, out_h=spec.out_h, patch=spec.patch,
                      overlap=spec.overlap, seed=spec.seed,
                      periodic=spec.periodic, policy=spec.policy)
    cfg, notes = normalize_config(cfg, samples)

    seed_payload, seed_info = build_seed(samples, spec, cfg)
    cons, locking = build_constraints(samples, spec, seed_info)

    # 锁定细边必须是“已有内容”的一部分：只能锁框选种子内部的像素，
    # 否则没有像素可锁、且会因禁止写入而留孔。
    for le in locking:
        r = le.rect
        central = seed_payload["protect"]
        inside = central[max(0, r.y0):r.y1, max(0, r.x0):r.x1]
        if inside.size == 0 or not inside.all():
            raise ValueError(
                "锁定细边必须完全位于框选样本区域内（锁定的是已有像素，"
                "不能锁尚未生成的空白区）")

    # 锁定边：在种子铺好后立即快照，扩展结束时逐位比对。
    for le in locking:
        le.snapshot(seed_payload["canvas"])

    # 借用端点：把锁定细边端点折算为某个邻接条带的 pin。
    pin_notes: list[dict] = []
    if spec.borrow_endpoint and locking:
        pin_notes = _resolve_borrow(cons, locking[0], cfg)

    eng = QuiltEngine(samples, cfg, constraints=cons, seed=seed_payload)
    res = eng.run()

    report = verify_extension(res, samples, seed_payload, locking, cfg, spec)
    report["engine"] = {
        "anchors_x": res["anchors_x"], "anchors_y": res["anchors_y"],
        "constraint_events": eng.constraint_events,
        "seed_placements": eng.seed_placements,
        "pin_notes": pin_notes,
        "dirty_tiles": _dirty_tile_map(cons, res, spec, cfg),
    }
    return {"result": res, "report": report, "cfg": cfg,
            "config_notes": notes, "seed_info": seed_info,
            "placements": res["placements"]}


def _anchor_list(length, patch, overlap, periodic):
    from .geometry import build_anchors
    return build_anchors(length, patch, overlap, periodic)


def _resolve_borrow(cons: SeamConstraints, edge: LockedEdge,
                    cfg: QuiltConfig) -> list[dict]:
    """把锁定竖边的下端点钉到右邻图块 left 条带的首行，形成端点贴合。"""
    notes: list[dict] = []
    r = edge.rect
    xs = _anchor_list(cfg.out_w, cfg.patch, cfg.overlap, cfg.periodic)
    ys = _anchor_list(cfg.out_h, cfg.patch, cfg.overlap, cfg.periodic)
    ex = r.x1 - 1
    ey = r.y1 - 1
    ix = max((i for i, x in enumerate(xs) if x <= ex), default=0)
    iy = max((i for i, y in enumerate(ys) if y <= ey), default=0)
    nix = min(ix + 1, len(xs) - 1)
    ax, ay = xs[nix], ys[iy]
    local = ey - ay
    if 0 <= local < cfg.patch:
        cons.pins.append(Pin(ix=nix, iy=iy, kind="left",
                             start=int(local), end=None,
                             reason="借用锁定细边下端点"))
        notes.append({"ix": nix, "iy": iy, "kind": "left",
                      "start": int(local), "anchor": [int(ax), int(ay)]})
    return notes


def _dirty_tile_map(cons: SeamConstraints, res: dict, spec: ExtendSpec,
                    cfg: QuiltConfig) -> list[dict]:
    """对每个禁区，列出其覆盖到的图块网格位置（脏区域更新的最小范围）。"""
    xs = _anchor_list(cfg.out_w, cfg.patch, cfg.overlap, cfg.periodic)
    ys = _anchor_list(cfg.out_h, cfg.patch, cfg.overlap, cfg.periodic)
    out = []
    for fz in cons.forbidden:
        tiles = _dirty_tiles_for_rect(fz.rect, xs, ys, cfg.patch,
                                      cfg.out_w, cfg.out_h)
        out.append({"name": fz.name, "rect": fz.rect.as_dict(),
                    "dirty_tiles": tiles})
    return out


def _dirty_tiles_for_rect(rect: Rect, xs: list[int], ys: list[int],
                          patch: int, w: int, h: int) -> list[list[int]]:
    out: list[list[int]] = []
    for iy, ay in enumerate(ys):
        for ix, ax in enumerate(xs):
            tx1 = min(w, ax + patch)
            ty1 = min(h, ay + patch)
            if ax < rect.x1 and tx1 > rect.x0 and ay < rect.y1 and ty1 > rect.y0:
                out.append([int(ix), int(iy)])
    return out


def verify_extension(res: dict, samples: list[Sample], seed_payload: dict,
                     locking: list[LockedEdge], cfg: QuiltConfig,
                     spec: ExtendSpec) -> dict:
    canvas = res["canvas"]
    own = res["own"]
    filled = seed_payload["filled"] | (own >= 0)

    # 1) 覆盖完整性：扩展后整块画布必须被填满，不留孔洞。
    holes = int((~filled).sum())
    coverage = {"width": cfg.out_w, "height": cfg.out_h,
                "covered_pixels": int(filled.sum()),
                "total_pixels": int(filled.size),
                "hole_pixels": holes,
                "complete": holes == 0}

    # 2) 锁定细边：扩展前后逐位相等。
    locked = [le.verify(canvas) for le in locking]

    # 3) 受保护种子：所有种子像素必须恒等（含锁边之外的部分）。
    seed_protect = seed_payload["protect"]
    seed_now = canvas[seed_protect]
    seed_ref = seed_payload["canvas"][seed_protect]
    seed_changed = int(np.any(seed_now != seed_ref, axis=-1).sum())
    protected = {"seed_pixels": int(seed_protect.sum()),
                 "changed_pixels": seed_changed,
                 "identical": seed_changed == 0}

    # 4) 沿缝两侧回读像素来源：每个输出像素必须等于 map 指向的样本像素。
    prov = provenance_readback(canvas, res["map_sample"], res["map_y"],
                               res["map_x"], samples, own, cfg.periodic)

    # 5) 接缝代价（真实画布上的可见接缝）与相对自然地板比值。
    seam = M.ownership_seam_rmse(canvas, own, cfg.periodic)
    nat = M.natural_adjacent_rmse(samples[0].image)
    seam["natural_floor"] = nat
    seam["seam_to_natural"] = float(seam["seam_rmse"] / max(1e-6, nat))

    return {"coverage": coverage, "locked_edges": locked,
            "protected": protected, "provenance": prov, "seam": seam,
            "spec": {"seed_rect": spec.seed_rect, "place_at": list(spec.place_at),
                     "lock_edge": spec.lock_edge,
                     "forbidden": [list(f) for f in spec.forbidden],
                     "borrow_endpoint": spec.borrow_endpoint}}


def provenance_readback(canvas: np.ndarray, map_sample: np.ndarray,
                        map_y: np.ndarray, map_x: np.ndarray,
                        samples: list[Sample], own: np.ndarray,
                        periodic: bool) -> dict:
    """逐像素把画布值与来源样本坐标对照，并单独检查沿缝两侧。"""
    valid = (map_sample >= 0) & (map_y >= 0) & (map_x >= 0)
    total = int(valid.sum())
    mismatch = 0
    max_delta = 0.0
    for sid, s in enumerate(samples):
        sel = valid & (map_sample == sid)
        if not sel.any():
            continue
        ys = map_y[sel]
        xs = map_x[sel]
        hh, ww = s.image.shape[:2]
        ok = (ys < hh) & (xs < ww)
        pos = np.where(sel)
        py = pos[0][ok]
        px = pos[1][ok]
        expected = s.image[ys[ok], xs[ok]]
        actual = canvas[py, px]
        d = np.abs(expected - actual)
        mismatch += int(np.any(d > 1e-6, axis=-1).sum())
        if d.size:
            max_delta = max(max_delta, float(d.max()))

    # 沿缝两侧：找一条竖直归属缝，比较缝两侧像素与其来源
    seam_side = _seam_side_readback(canvas, map_sample, map_y, map_x, samples,
                                    own, periodic)
    return {"pixels_checked": total, "mismatch_pixels": mismatch,
            "max_abs_delta": max_delta,
            "exact": mismatch == 0, "seam_side": seam_side}


def _seam_side_readback(canvas, map_sample, map_y, map_x, samples, own,
                        periodic) -> dict:
    """沿一条真实归属缝取两侧各一列，回读来源，确认来源即实际像素。"""
    h, w = own.shape
    diff = own[:, 1:] != own[:, :-1]
    if not diff.any():
        return {"available": False}
    ys, xs = np.nonzero(diff)
    # 取中间一处的整段缝
    y_mid = h // 2
    row_hits = np.nonzero(diff[y_mid])[0]
    if len(row_hits) == 0:
        return {"available": False}
    sx = int(row_hits[len(row_hits) // 2])
    left_x = sx
    right_x = sx + 1
    checked = 0
    bad = 0
    for yy in range(0, h, max(1, h // 32)):
        for xx in (left_x, right_x):
            sid = int(map_sample[yy, xx])
            sy = int(map_y[yy, xx])
            sxv = int(map_x[yy, xx])
            if sid < 0 or sy < 0 or sxv < 0:
                continue
            if abs(float(canvas[yy, xx].max() - samples[sid].image[sy, sxv].max())) > 1e-6 \
                    or not np.allclose(canvas[yy, xx], samples[sid].image[sy, sxv], atol=1e-6):
                bad += 1
            checked += 1
    return {"available": True, "seam_x": int(left_x),
            "rows_checked": int(checked), "mismatch": int(bad),
            "exact": bad == 0}
