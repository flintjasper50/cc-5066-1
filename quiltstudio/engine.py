"""核心：样本驱动图块选择 + 重叠 SSD + 最小割接缝 + 周期闭环约束 + 来源追溯。"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .mincut import min_cut_path, path_energy, straight_path
from .samples import Sample


def _natural_adj_rmse(img: np.ndarray) -> float:
    dx = img[:, 1:, :] - img[:, :-1, :]
    dy = img[1:, :, :] - img[:-1, :, :]
    sse = float((dx * dx).sum() + (dy * dy).sum())
    n = dx.size + dy.size
    return float(np.sqrt(sse / max(1, n)))


@dataclass
class QuiltConfig:
    out_w: int = 384
    out_h: int = 384
    patch: int = 64
    overlap: int = 24
    seed: int = 20261126
    periodic: bool = False
    seam_mode: str = "mincut"          # mincut | straight
    policy: str = "adaptive"           # off | reroute | resample | adaptive
    candidate_rel_tol: float = 0.18
    candidate_abs_tol: float = 2.0e-4  # MSE 绝对值容差，保证平坦区也有候选池
    seam_k: int = 12
    resample_tries: int = 12
    seam_warn_rmse: float = 0.085
    seam_fail_rmse: float = 0.17
    match_fail_rmse: float = 0.17

    def key(self) -> str:
        d = asdict(self)
        return "|".join(f"{k}={d[k]}" for k in sorted(d))


def normalize_config(cfg: QuiltConfig, samples: list[Sample]) -> tuple[QuiltConfig, list[str]]:
    """根据样本尺寸收敛 patch/overlap，返回 (新配置, 调整说明)。"""
    notes: list[str] = []
    min_h = min(s.image.shape[0] for s in samples)
    min_w = min(s.image.shape[1] for s in samples)
    c = QuiltConfig(**asdict(cfg))
    max_patch = min(min_h, min_w, c.out_w, c.out_h)
    if c.patch > max_patch:
        c.patch = max(24, max_patch)
        notes.append(f"patch 已收敛到 {c.patch} 以适配样本/画布尺寸")
    if c.patch < 20:
        raise ValueError("样本太小：至少需要 20×20 纹理")
    max_ov = c.patch // 2 - 2
    if c.overlap > max_ov:
        c.overlap = max(4, max_ov)
        notes.append(f"overlap 已收敛到 {c.overlap}（需小于 patch/2 以避免区域重叠）")
    if c.overlap < 2:
        c.overlap = max(2, c.patch // 3)
    if c.out_w < c.patch or c.out_h < c.patch:
        raise ValueError("输出画布不能小于 patch")
    return c, notes


def make_anchors(length: int, patch: int, overlap: int, periodic: bool) -> list[int]:
    step = patch - overlap
    if periodic:
        n = max(2, int(round(length / step)))
        xs = [int(round(i * length / n)) for i in range(n)]
        return xs
    if length <= patch:
        return [0]
    span = length - patch
    n = max(2, int(round(span / step)) + 1)
    xs = sorted({int(round(i * span / (n - 1))) for i in range(n)})
    return xs


def build_regions(ix: int, iy: int, xs: list[int], ys: list[int],
                  w: int, h: int, b: int, periodic: bool) -> list[dict]:
    """构造互不相交的重叠矩形区域（局部 patch 坐标，半开区间）。"""
    ax, ay = xs[ix], ys[iy]
    left_w = b - (ax - xs[ix - 1]) if ix > 0 else 0
    close_c0, close_w = b, 0
    if periodic and ix == len(xs) - 1:
        close_c0 = w - ax
        close_w = b - close_c0
    cleft = left_w if ix > 0 else 0
    cright = close_c0 if close_w > 0 else b
    if cleft > cright:  # 极端参数保护
        cleft, cright = 0, b

    regs: list[dict] = []
    if ix > 0 and left_w > 0:
        regs.append({"kind": "left", "axis": "v", "r0": 0, "r1": b,
                     "c0": 0, "c1": left_w, "orient": "old_left"})
    if iy > 0:
        top_h = b - (ay - ys[iy - 1])
        if top_h > 0 and cright > cleft:
            regs.append({"kind": "top", "axis": "h", "r0": 0, "r1": top_h,
                         "c0": cleft, "c1": cright, "orient": "old_top"})
    if close_w > 0:
        regs.append({"kind": "close_right", "axis": "v", "r0": 0, "r1": b,
                     "c0": close_c0, "c1": b, "orient": "new_left"})
    if periodic and iy == len(ys) - 1:
        close_r0 = h - ay
        close_h = b - close_r0
        if close_h > 0 and cright > cleft:
            regs.append({"kind": "close_bottom", "axis": "h",
                         "r0": close_r0, "r1": b, "c0": cleft, "c1": cright,
                         "orient": "new_top"})
    return regs


def region_coords(reg: dict, ax: int, ay: int, w: int, h: int,
                  periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    rows = ay + np.arange(reg["r0"], reg["r1"])
    cols = ax + np.arange(reg["c0"], reg["c1"])
    if periodic:
        rows = rows % h
        cols = cols % w
    return rows.astype(np.int64), cols.astype(np.int64)


class QuiltEngine:
    def __init__(self, samples: list[Sample], cfg: QuiltConfig):
        self.samples = samples
        self.cfg = cfg
        self.b = cfg.patch
        self.w = cfg.out_w
        self.h = cfg.out_h
        self.xs = make_anchors(self.w, self.b, cfg.overlap, cfg.periodic)
        self.ys = make_anchors(self.h, self.b, cfg.overlap, cfg.periodic)
        self.canvas = np.zeros((self.h, self.w, 3), np.float32)
        self.own = np.full((self.h, self.w), -1, np.int32)
        self.map_sample = np.full((self.h, self.w), -1, np.int32)
        self.map_y = np.full((self.h, self.w), -1, np.int32)
        self.map_x = np.full((self.h, self.w), -1, np.int32)
        self.placements: list[dict[str, Any]] = []
        self._root_rng = random.Random(cfg.seed)
        # 绝对 RMSE 阈值对高频白噪声纹理过严（其自然相邻差本身就 >0.17），
        # 用样本自然地板做下限自适应；普通纹理仍沿用配置里的绝对线。
        self.nat_floor = min(_natural_adj_rmse(s.image) for s in samples)
        self.warn_line = max(cfg.seam_warn_rmse, 0.8 * self.nat_floor)
        self.fail_line = max(cfg.seam_fail_rmse, 1.5 * self.nat_floor)

    # ---------- 候选代价 ----------

    def _cost_maps(self, regs: list[dict], ax: int, ay: int) -> list[np.ndarray]:
        """每个样本一张候选位置 MSE 图（形状 Hs-B+1 × Ws-B+1）。"""
        total_px3 = 3 * sum((r["r1"] - r["r0"]) * (r["c1"] - r["c0"]) for r in regs)
        out: list[np.ndarray] = []
        for s in self.samples:
            img = s.image
            hs, ws = img.shape[:2]
            ny, nx = hs - self.b + 1, ws - self.b + 1
            total = np.zeros((ny, nx), np.float64)
            if regs:
                for r in regs:
                    ry, rx = region_coords(r, ax, ay, self.w, self.h, self.cfg.periodic)
                    target = self.canvas[np.ix_(ry, rx)]
                    rh, rw = r["r1"] - r["r0"], r["c1"] - r["c0"]
                    if target.shape[0] != rh or target.shape[1] != rw:
                        raise RuntimeError("overlap target shape mismatch")
                    res = cv2.matchTemplate(img, target, cv2.TM_SQDIFF)
                    sub = res[r["r0"]:r["r0"] + ny, r["c0"]:r["c0"] + nx]
                    total += sub.astype(np.float64)
                total /= max(1, total_px3)
            out.append(total)
        return out

    def _top_candidates(self, maps: list[np.ndarray], k: int) -> list[dict]:
        recs: list[dict] = []
        for sid, m in enumerate(maps):
            flat = m.ravel()
            kk = min(k, flat.size)
            idx = np.argpartition(flat, kk - 1)[:kk]
            for fi in idx:
                sy, sx = divmod(int(fi), m.shape[1])
                recs.append({"sample_id": sid,
                             "sample_name": self.samples[sid].name,
                             "source_y": sy, "source_x": sx,
                             "mse": float(m[sy, sx])})
        recs.sort(key=lambda d: d["mse"])
        return recs[:k]

    def _eligible(self, maps: list[np.ndarray], thr: float):
        elig = []
        for m in maps:
            ys, xs = np.nonzero(m <= thr)
            elig.append(np.stack([ys, xs], axis=1) if len(ys)
                        else np.zeros((0, 2), np.int64))
        return elig

    @staticmethod
    def _draw(rng: random.Random, elig: list[np.ndarray], log: list,
              purpose: str, sid_scope: int | None = None,
              exclude: set | None = None) -> dict | None:
        pools = elig if sid_scope is None else [elig[sid_scope]]
        total = sum(len(e) for e in pools)
        if exclude:
            # 过滤已评估候选
            avail = []
            for li, e in enumerate(pools):
                sid = li if sid_scope is None else sid_scope
                keep = [i for i in range(len(e))
                        if (sid, int(e[i, 0]), int(e[i, 1])) not in exclude]
                avail.append((sid, e[keep] if keep else np.zeros((0, 2), np.int64)))
            total = sum(len(e) for _, e in avail)
        else:
            avail = []
            for li, e in enumerate(pools):
                sid = li if sid_scope is None else sid_scope
                avail.append((sid, e))
        if total <= 0:
            return None
        u = rng.random()
        k = min(int(u * total), total - 1)
        for sid, e in avail:
            if k < len(e):
                sy, sx = int(e[k, 0]), int(e[k, 1])
                rec = {"sample_id": sid, "source_y": sy, "source_x": sx}
                log.append({"purpose": purpose, "u": u, "pool": total,
                            "pick_index": k, "picked": rec})
                return rec
            k -= len(e)
        return None

    # ---------- 切缝规划 ----------

    def _plan(self, cand: dict, regs: list[dict]) -> dict:
        sid, sy, sx = cand["sample_id"], cand["source_y"], cand["source_x"]
        patch = self.samples[sid].image[sy:sy + self.b, sx:sx + self.b]
        region_plans = []
        masks: dict[int, np.ndarray] = {}
        worst = 0.0
        area_sum = 0
        mse_sum = 0.0
        for ri, r in enumerate(regs):
            ry = self._ay + np.arange(r["r0"], r["r1"])
            rx = self._ax + np.arange(r["c0"], r["c1"])
            if self.cfg.periodic:
                ry, rx = ry % self.h, rx % self.w
            ry, rx = ry.astype(np.int64), rx.astype(np.int64)
            old = self.canvas[np.ix_(ry, rx)]
            new = patch[r["r0"]:r["r1"], r["c0"]:r["c1"]]
            cost = ((new - old) ** 2).sum(axis=2)
            cost_for_path = cost if r["axis"] == "v" else cost.T
            if self.cfg.seam_mode == "mincut":
                path, _ = min_cut_path(cost_for_path)
            else:
                path = straight_path(cost_for_path)
            raw_mean = path_energy(cost_for_path, path)
            straight = straight_path(cost_for_path)
            straight_rmse = float(np.sqrt(max(0.0, path_energy(cost_for_path, straight)) / 3.0))
            rmse = float(np.sqrt(max(0.0, raw_mean) / 3.0))
            hh, ww = cost.shape
            if r["axis"] == "v":
                q = np.arange(ww)
                if r["orient"] == "old_left":
                    mask = q[None, :] >= path[:, None]
                else:  # new_left（闭环右缝）
                    mask = q[None, :] < path[:, None]
                pts = [[int(self._ax + r["c0"] + int(path[i])),
                        int(self._ay + r["r0"] + i)] for i in range(hh)]
            else:
                q = np.arange(hh)
                if r["orient"] == "old_top":
                    mask = q[:, None] >= path[None, :]
                else:  # new_top（闭环下缝）
                    mask = q[:, None] < path[None, :]
                pts = [[int(self._ax + r["c0"] + j),
                        int(self._ay + r["r0"] + int(path[j]))] for j in range(ww)]
            masks[ri] = mask
            area = hh * ww
            worst = max(worst, rmse)
            mse_sum += rmse * rmse * area
            area_sum += area
            region_plans.append({"kind": r["kind"], "axis": r["axis"],
                                 "orient": r["orient"], "rmse": rmse,
                                 "straight_rmse": straight_rmse,
                                 "path": [int(v) for v in path],
                                 "points_unwrapped": pts, "area": area})
        mean_rmse = float(np.sqrt(mse_sum / max(1, area_sum)))
        return {"cand": cand, "patch": patch, "masks": masks,
                "worst_rmse": worst, "mean_rmse": mean_rmse,
                "regions": region_plans}

    def _apply(self, tile_id: int, plan: dict, regs: list[dict]) -> None:
        # 非重叠区全部取新 patch；重叠区只取切缝判给新块的部分。
        write = np.ones((self.b, self.b), bool)
        for ri, r in enumerate(regs):
            write[r["r0"]:r["r1"], r["c0"]:r["c1"]] = plan["masks"][ri]
        ly, lx = np.mgrid[0:self.b, 0:self.b]
        gy = self._ay + ly
        gx = self._ax + lx
        syg = plan["cand"]["source_y"] + ly
        sxg = plan["cand"]["source_x"] + lx
        if self.cfg.periodic:
            gy_w, gx_w = gy % self.h, gx % self.w
        else:
            gy_w, gx_w = gy, gx
        sel = write
        yy, xx = gy_w[sel], gx_w[sel]
        self.canvas[yy, xx] = plan["patch"][sel]
        self.own[yy, xx] = tile_id
        self.map_sample[yy, xx] = plan["cand"]["sample_id"]
        self.map_y[yy, xx] = syg[sel]
        self.map_x[yy, xx] = sxg[sel]

    # ---------- 主循环 ----------

    def run(self) -> dict:
        cfg = self.cfg
        tile_id = 0
        for iy, ay in enumerate(self.ys):
            for ix, ax in enumerate(self.xs):
                self._ax, self._ay = ax, ay
                child_seed = self._root_rng.getrandbits(63)
                rng = random.Random(child_seed)
                draw_log: list[dict] = []
                regs = build_regions(ix, iy, self.xs, self.ys,
                                     self.w, self.h, self.b, cfg.periodic)
                maps = self._cost_maps(regs, ax, ay)
                top = self._top_candidates(maps, max(cfg.seam_k, 10))
                gmin = top[0]["mse"] if top else 0.0
                thr = gmin * (1.0 + cfg.candidate_rel_tol) + cfg.candidate_abs_tol
                elig = self._eligible(maps, thr)
                eligible_total = sum(len(e) for e in elig)

                base = self._draw(rng, elig, draw_log, "initial")
                if base is None:  # 理论不会发生（thr>gmin）
                    base = top[0]
                base["mse"] = float(
                    maps[base["sample_id"]][base["source_y"], base["source_x"]])

                evaluated: dict[tuple, dict] = {}

                def key_of(c):
                    return (c["sample_id"], c["source_y"], c["source_x"])

                def evaluate(c: dict, stage: str) -> dict:
                    k = key_of(c)
                    if k in evaluated:
                        return evaluated[k]
                    c = dict(c)
                    c["mse"] = float(maps[c["sample_id"]][c["source_y"], c["source_x"]])
                    p = self._plan(c, regs)
                    rec = {"key": {"sample_id": c["sample_id"],
                                   "source_y": c["source_y"],
                                   "source_x": c["source_x"]},
                           "stage": stage, "mse": c["mse"],
                           "match_rmse": float(np.sqrt(max(0.0, c["mse"]))),
                           "seam_worst_rmse": p["worst_rmse"],
                           "seam_mean_rmse": p["mean_rmse"]}
                    evaluated[k] = {"rec": rec, "plan": p, "cand": c}
                    return evaluated[k]

                base_eval = evaluate(base, "initial")
                chosen_eval = base_eval
                decision = "standard"
                reason = "随机候选的切缝误差在阈值内"
                flags: list[str] = []

                triggered = base_eval["plan"]["worst_rmse"] > self.warn_line
                if not regs:
                    decision = "first-tile-random"
                    reason = "首个图块无重叠约束，在全样本中随机取块"
                    triggered = False

                if triggered and cfg.policy in ("reroute", "adaptive"):
                    for c in top[:cfg.seam_k]:
                        evaluate(c, "shortlist")
                    best = min(evaluated.values(),
                               key=lambda e: (e["plan"]["worst_rmse"],
                                              e["plan"]["mean_rmse"]))
                    if best["plan"]["worst_rmse"] < chosen_eval["plan"]["worst_rmse"] - 1e-4:
                        chosen_eval = best
                        decision = "reroute"
                        reason = ("初始随机块切缝 RMSE=%.3f 超告警线；"
                                  "在外观容差候选中按最低切缝代价重排，改选 %s"
                                  % (base_eval["plan"]["worst_rmse"],
                                     self._key_label(best["cand"])))
                    else:
                        decision = "reroute-reviewed"
                        reason = "切缝重排候选均未显著优于初始选择，保留原块"

                if (chosen_eval["plan"]["worst_rmse"] > self.warn_line
                        and cfg.policy in ("resample", "adaptive") and regs):
                    neighbor_sid = self._neighbor_source(ix, iy)
                    stages = (["same"] * 4 + ["neighbor"] * 4 + ["any"] * 4)
                    best_overall = chosen_eval
                    for t in range(cfg.resample_tries):
                        scope = None
                        st = stages[min(t, len(stages) - 1)]
                        if st == "same":
                            scope = base["sample_id"]
                        elif st == "neighbor" and neighbor_sid is not None:
                            scope = neighbor_sid
                        excl = set(evaluated.keys())
                        c = self._draw(rng, elig, draw_log,
                                       f"resample-{st}-{t}", scope, excl)
                        if c is None:
                            continue
                        ev = evaluate(c, f"resample-{st}")
                        if ev["plan"]["worst_rmse"] < best_overall["plan"]["worst_rmse"]:
                            best_overall = ev
                        if ev["plan"]["worst_rmse"] <= self.warn_line:
                            chosen_eval = ev
                            decision = "resample"
                            reason = ("切缝仍过高，按 %s 范围重采样来源后找到合格块 %s"
                                      % ({"same": "同样片", "neighbor": "邻接来源",
                                          "any": "全部来源"}[st],
                                         self._key_label(ev["cand"])))
                            break
                    if decision != "resample":
                        if best_overall["plan"]["worst_rmse"] < chosen_eval["plan"]["worst_rmse"] - 1e-4:
                            chosen_eval = best_overall
                            decision = "resample-best"
                            reason = "重采样未找到合格块，采用其中切缝代价最低者"
                        else:
                            decision = "unresolved"
                            reason = "重采样后仍无低代价切缝：局部结构互斥/过强，已标记为可能失败"

                final_plan = chosen_eval["plan"]
                final_cand = chosen_eval["cand"]
                if regs and final_plan["worst_rmse"] > self.fail_line:
                    flags.append("high-seam-error")
                if gmin > cfg.match_fail_rmse ** 2 and regs:
                    flags.append("no-good-match")

                self._apply(tile_id, final_plan, regs)

                top_trace = [{k: v for k, v in c.items()} for c in top[:10]]
                p_trace = {
                    "id": tile_id, "ix": ix, "iy": iy,
                    "anchor": [int(ax), int(ay)],
                    "child_seed": child_seed,
                    "patch_size": self.b,
                    "eligible_total": int(eligible_total),
                    "threshold_mse": float(thr),
                    "min_match_rmse": float(np.sqrt(max(0.0, gmin))),
                    "chosen": {"sample_id": final_cand["sample_id"],
                               "sample_name": self.samples[final_cand["sample_id"]].name,
                               "source_y": final_cand["source_y"],
                               "source_x": final_cand["source_x"],
                               "match_rmse": float(np.sqrt(max(0.0, final_cand["mse"])))},
                    "decision": decision, "reason": reason, "flags": flags,
                    "draws": draw_log,
                    "seam_worst_rmse": final_plan["worst_rmse"],
                    "seam_mean_rmse": final_plan["mean_rmse"],
                    "regions": final_plan["regions"],
                    "top_candidates": top_trace,
                    "attempts": [e["rec"] for e in evaluated.values()],
                }
                self.placements.append(p_trace)
                tile_id += 1

        return {"canvas": self.canvas, "own": self.own,
                "map_sample": self.map_sample, "map_y": self.map_y,
                "map_x": self.map_x, "placements": self.placements,
                "anchors_x": self.xs, "anchors_y": self.ys,
                "nat_floor": self.nat_floor,
                "warn_line": self.warn_line, "fail_line": self.fail_line}

    def _neighbor_source(self, ix: int, iy: int) -> int | None:
        # 优先左邻，其次上邻的 sample id
        if ix > 0:
            for p in reversed(self.placements):
                if p["ix"] == ix - 1 and p["iy"] == iy:
                    return int(p["chosen"]["sample_id"])
        if iy > 0:
            for p in reversed(self.placements):
                if p["ix"] == ix and p["iy"] == iy - 1:
                    return int(p["chosen"]["sample_id"])
        return None

    def _key_label(self, c: dict) -> str:
        return f"[{c['sample_id']}]({c['source_y']},{c['source_x']})"
