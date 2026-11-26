"""把一次配置跑成可交付产物：图片、source_map.npz、run.json，并汇总 manifest。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import metrics as M
from .engine import QuiltConfig, QuiltEngine, normalize_config
from .imgio import save_png
from .samples import Sample
from . import viz

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "output"


def _json_safe(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not json safe: {type(o)}")


def _status_and_warnings(cfg: QuiltConfig, metrics: dict, placements: list[dict],
                         sample_struct: list[dict]) -> tuple[str, list[str], list[str]]:
    warns: list[str] = []
    infos: list[str] = []
    n = max(1, len(placements))
    unresolved = sum(1 for p in placements if p["decision"] == "unresolved")
    high_seam_flags = sum(1 for p in placements if "high-seam-error" in p["flags"])
    no_match_flags = sum(1 for p in placements if "no-good-match" in p["flags"])
    reroute = sum(1 for p in placements if p["decision"] == "reroute")
    resampled = sum(1 for p in placements if p["decision"] in ("resample", "resample-best"))

    floors = [s["natural_adj_rmse"] for s in sample_struct if s["natural_adj_rmse"] > 1e-6]
    floor_ref = float(np.mean(floors)) if floors else 0.1
    seam = metrics["seam"]["seam_rmse"]
    ratio = seam / max(floor_ref, 1e-3)
    metrics["seam_to_natural_ratio"] = ratio

    if unresolved:
        warns.append(f"{unresolved}/{n} 个图块在切缝重排与来源重采样后仍无解，"
                     "说明局部锚点/强结构互斥，结果在这些位置可能断裂。")
    if high_seam_flags:
        fail_line = metrics.get("thresholds", {}).get("fail_line", cfg.seam_fail_rmse)
        warns.append(f"{high_seam_flags}/{n} 个图块的规划切缝 RMSE 超过失败线 "
                     f"{fail_line:.2f}（已在图上红框标出）。")
    if no_match_flags and (ratio > 1.15 or unresolved or high_seam_flags):
        warns.append(f"{no_match_flags}/{n} 个位置找不到低重叠代价候选："
                     "样本缺少所需结构，输出只能勉强拼合。")
    elif no_match_flags:
        infos.append(f"{no_match_flags}/{n} 个位置最佳重叠 RMSE 偏高；"
                     "对高频随机纹理这是正常现象，最小割已把实际接缝压到自然噪声水平。")
    if ratio > 1.35:
        warns.append(f"实际接缝 RMSE={seam:.3f} 是原样本自然相邻差 "
                     f"{floor_ref:.3f} 的 {ratio:.2f} 倍，可见断裂偏多。")
    elif ratio > 1.15:
        infos.append(f"实际接缝 RMSE={seam:.3f} 为自然相邻差 {floor_ref:.3f} "
                     f"的 {ratio:.2f} 倍，近距离仍可能看到切痕。")
    if cfg.periodic:
        edge = metrics["periodic"]["edge_rmse"]
        if edge > floor_ref * 1.7:
            warns.append(f"周期首尾边界 RMSE={edge:.3f} 明显高于自然纹理水平，"
                         "闭环约束虽已参与求解但未找到足够低代价的闭合切缝。")
    if metrics["duplicates"]["dup_pair_rate"] > 0.12:
        warns.append("输出图块随机配对重复率偏高，存在可察觉的复制痕迹。")
    if metrics["repetition"]["max_exact_reuse"] >= 3:
        warns.append("同一来源图块被原样重复使用 ≥3 次，重复痕迹可见。")
    ap = metrics["autocorrelation"]["autocorr_peak"]
    if ap > 0.55:
        lag = metrics["autocorrelation"]["autocorr_lag"]
        warns.append(f"自相关在位移 {lag} 处出现 {ap:.2f} 的强峰，存在周期性重复。")
    # 方向结构损失
    out_coh = metrics["orientation"]["orientation_coherence"]
    strong_samples = [s for s in sample_struct if s["orientation_coherence"] > 0.55]
    if strong_samples and out_coh < min(s["orientation_coherence"] for s in strong_samples) - 0.18:
        warns.append("条纹方向相干性相比原样本明显下降：强方向结构在拼接处被打断。")

    fc = metrics.get("feature_cuts", {})
    fc_st = metrics.get("feature_cuts_straight", {})
    anchor_fail = False
    if fc.get("available"):
        events = fc.get("anchor_cut_events", 0)
        severe = fc.get("anchor_severe_events", 0)
        ev_density = fc.get("cut_events_per_100kpx", 0.0)
        sev_density = fc.get("severe_events_per_100kpx", 0.0)
        st_events = fc_st.get("anchor_cut_events") if fc_st.get("available") else None
        if st_events is not None and st_events != events:
            if st_events > events:
                cmp_txt = (f"同种子硬切基线会切断 {st_events} 次锚点，"
                           f"最小割减少 {st_events - events} 次；")
            else:
                cmp_txt = (f"但同种子硬切基线仅切断 {st_events} 次，"
                           "本次最小割在语义锚点上反而更差；")
        else:
            cmp_txt = f"与硬切基线相同（{events} 次）；" if st_events is not None else ""
        if severe >= 8 and sev_density >= 6.0:
            warns.append(f"{cmp_txt}{severe} 段长接缝从独特标记/锚点笔画内部穿过"
                         f"（共 {events} 次锚点切割，{severe} 次严重）：强语义结构"
                         "超出 image quilting 可重排范围，标记被切碎/错位，"
                         "这类损坏无法只看颜色 RMSE。")
            anchor_fail = True
        elif events >= 5 and ev_density >= 3.0:
            warns.append(f"{cmp_txt}{events} 次接缝切到真值锚点（{severe} 次严重）："
                         "部分独特标记/方向箭头可能被切边或错位。")
        elif st_events is not None and st_events > events:
            infos.append(f"锚点切割事件 {events} 次（硬切基线 {st_events} 次），"
                         "最小割绕开了部分地标。")

    unresolved_frac = unresolved / n
    high_fail_frac = high_seam_flags / n
    # 单纯 unresolved 比例高不直接判死：自适应告警线对异常纹理仍可能偏严，
    # 需同时有全局接缝 RMSE 偏高或实际 high-seam 标记作为佐证。
    unresolved_fatal = unresolved_frac >= 0.15 and (
        ratio > 1.2 or high_seam_flags > 0)
    if (ratio > 1.5 or high_fail_frac >= 0.08 or unresolved_fatal
            or anchor_fail):
        status = "failed-likely"
    elif warns:
        status = "warning"
    else:
        status = "ok"
    metrics["interventions"] = {"reroute": reroute, "resample": resampled,
                                "unresolved": unresolved}
    return status, warns, infos


def execute(samples: list[Sample], cfg: QuiltConfig, run_id: str,
            title: str, group: str = "custom",
            out_root: Path | str = OUT_ROOT,
            save_artifacts: bool = True) -> dict:
    t_start = time.time()
    out_root = Path(out_root)
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg, config_notes = normalize_config(cfg, samples)

    engine = QuiltEngine(samples, cfg)
    res = engine.run()
    canvas, own = res["canvas"], res["own"]

    # 同种子、同候选序列的硬切基线（仅切缝方式不同）
    straight_cfg = QuiltConfig(**{**asdict(cfg), "policy": "off",
                                  "seam_mode": "straight"})
    straight_res = QuiltEngine(samples, straight_cfg).run()

    # 几何显示信息
    for p in res["placements"]:
        p["rects"] = viz.wrapped_rects(p["anchor"][0], p["anchor"][1],
                                      cfg.patch, cfg.out_w, cfg.out_h, cfg.periodic)

    seam_metrics = M.ownership_seam_rmse(canvas, own, cfg.periodic)
    straight_seam_metrics = M.ownership_seam_rmse(
        straight_res["canvas"], straight_res["own"], cfg.periodic)
    periodic_metrics = M.periodic_boundary_rmse(canvas)
    periodic_straight = M.periodic_boundary_rmse(straight_res["canvas"])

    naive = M.tiling_baseline_rmse(samples[0].image, cfg.out_w, cfg.out_h)
    naive_canvas = naive.pop("canvas")

    sample_struct = []
    for s in samples:
        ori = M.orientation_structure(s.image)
        sample_struct.append({
            "id": s.id, "name": s.name,
            "natural_adj_rmse": M.natural_adjacent_rmse(s.image),
            **ori,
        })

    metrics = {
        "thresholds": {"natural_floor": res["nat_floor"],
                       "seam_warn_line": res["warn_line"],
                       "seam_fail_line": res["fail_line"],
                       "config_warn": cfg.seam_warn_rmse,
                       "config_fail": cfg.seam_fail_rmse},
        "seam": seam_metrics,
        "straight_seam": straight_seam_metrics,
        "seam_improvement_vs_straight_pct": (
            100.0 * (straight_seam_metrics["seam_rmse"] - seam_metrics["seam_rmse"])
            / max(1e-6, straight_seam_metrics["seam_rmse"])),
        "periodic": periodic_metrics,
        "periodic_straight": periodic_straight,
        "naive_tile": naive,
        "repetition": M.tile_reuse(res["placements"]),
        "autocorrelation": M.autocorrelation_peak(canvas, min_lag=cfg.patch // 3),
        "duplicates": M.patch_duplicate_rate(canvas, seed=cfg.seed % (2 ** 31)),
        "orientation": M.orientation_structure(canvas),
        "feature_cuts": M.feature_cut_metrics(
            samples, res["map_sample"], res["map_y"], res["map_x"], own, cfg.periodic),
        "feature_cuts_straight": M.feature_cut_metrics(
            samples, straight_res["map_sample"], straight_res["map_y"],
            straight_res["map_x"], straight_res["own"], cfg.periodic),
        "samples": sample_struct,
        "elapsed_sec": 0.0,
    }
    metrics["repetition"].pop("reused_tile_keys", None)  # 详细键不进汇总指标? 保留在 trace
    status, warnings, infos = _status_and_warnings(
        cfg, metrics, res["placements"], sample_struct)

    # 渲染
    seams_img = viz.render_seams_overlay(canvas, res["placements"], cfg.periodic)
    prov_id = viz.render_provenance_id(res["map_sample"], len(samples), own)
    ref_h = max(s.image.shape[0] for s in samples)
    ref_w = max(s.image.shape[1] for s in samples)
    prov_xy = viz.render_provenance_xy(res["map_y"], res["map_x"], ref_h, ref_w)
    grid_img = viz.render_tile_grid(canvas, res["placements"], cfg.patch, cfg.periodic)

    files: dict[str, str] = {}
    if save_artifacts:
        for name, arr in [("result", canvas), ("seams", seams_img),
                          ("provenance_id", prov_id), ("provenance_xy", prov_xy),
                          ("tile_grid", grid_img),
                          ("straight", straight_res["canvas"]),
                          ("naive_tile", naive_canvas)]:
            save_png(arr, run_dir / f"{name}.png")
            files[name] = f"output/runs/{run_id}/{name}.png"
        viz.save_tileid(own, run_dir / "tileid.png")
        files["tileid"] = f"output/runs/{run_id}/tileid.png"
        viz.build_patch_wall(samples, res["placements"], cfg.patch,
                             run_dir / "patch_wall.png")
        files["patch_wall"] = f"output/runs/{run_id}/patch_wall.png"

        if cfg.periodic:
            tile2x2 = np.tile(canvas, (2, 2, 1))
            save_png(tile2x2, run_dir / "tile2x2.png")
            files["tile2x2"] = f"output/runs/{run_id}/tile2x2.png"

        panels = [(f"ORIGINAL {i} [{s.id}]", s.image) for i, s in enumerate(samples)]
        panels += [
            ("QUILT + MIN-CUT SEAMS", canvas),
            ("SAME PATCHES, STRAIGHT CUT", straight_res["canvas"]),
            ("NAIVE TILING (whole sample)", naive_canvas),
        ]
        viz.build_comparison(run_dir / "comparison.png", panels, target_h=300)
        files["comparison"] = f"output/runs/{run_id}/comparison.png"

        for i, s in enumerate(samples):
            p = run_dir / f"sample_{i}.png"
            save_png(s.image, p)
            files[f"sample_{i}"] = f"output/runs/{run_id}/sample_{i}.png"

        np.savez_compressed(run_dir / "source_map.npz",
                            own=own, sample=res["map_sample"],
                            y=res["map_y"], x=res["map_x"])
        files["source_map_npz"] = f"output/runs/{run_id}/source_map.npz"

    metrics["elapsed_sec"] = round(time.time() - t_start, 2)

    run_json = {
        "run_id": run_id, "title": title, "group": group,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "config_notes": config_notes,
        "status": status, "warnings": warnings, "infos": infos,
        "metrics": metrics,
        "anchors_x": res["anchors_x"], "anchors_y": res["anchors_y"],
        "n_placements": len(res["placements"]),
        "samples": [{"id": s.id, "name": s.name, "role": s.role, "meta": s.meta}
                    for s in samples],
        "placements": res["placements"],
        "seam_segments": viz.all_seam_segments(
            res["placements"], cfg.out_w, cfg.out_h, cfg.periodic),
        "files": files,
    }
    if save_artifacts:
        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_json, f, ensure_ascii=False, default=_json_safe)
    return run_json


def write_manifest(runs: list[dict], out_root: Path | str = OUT_ROOT) -> Path:
    out_root = Path(out_root)
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": [{k: v for k, v in r.items() if k != "placements" and k != "seam_segments"}
                 for r in runs],
    }
    p = out_root / "manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, default=_json_safe, indent=1)
    web_data = ROOT / "web" / "data"
    web_data.mkdir(parents=True, exist_ok=True)
    js = "window.QS_MANIFEST = " + json.dumps(manifest, ensure_ascii=False,
                                              default=_json_safe) + ";\n"
    (web_data / "manifest.js").write_text(js, encoding="utf-8")
    return p


def _summary(run_json: dict) -> dict:
    return {k: v for k, v in run_json.items()
            if k not in ("placements", "seam_segments")}


def upsert_manifest(run_json: dict, out_root: Path | str = OUT_ROOT) -> Path:
    """把单个 run 合并进现有 manifest（同 id 覆盖），并刷新 web/data。"""
    out_root = Path(out_root)
    p = out_root / "manifest.json"
    runs = []
    if p.exists():
        try:
            runs = json.loads(p.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            runs = []
    runs = [r for r in runs if r.get("run_id") != run_json["run_id"]]
    runs.append(_summary(run_json))
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
    }
    p.write_text(json.dumps(manifest, ensure_ascii=False, default=_json_safe, indent=1),
                 encoding="utf-8")
    web_data = ROOT / "web" / "data"
    (web_data / "runs").mkdir(parents=True, exist_ok=True)
    (web_data / "manifest.js").write_text(
        "window.QS_MANIFEST = " + json.dumps(manifest, ensure_ascii=False,
                                             default=_json_safe) + ";\n",
        encoding="utf-8")
    (web_data / "runs" / f"{run_json['run_id']}.js").write_text(
        "window.QS_RUN_DATA=" + json.dumps(run_json, ensure_ascii=False,
                                           default=_json_safe) + ";\n",
        encoding="utf-8")
    return p
