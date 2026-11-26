#!/usr/bin/env python3
"""跑完整检验矩阵并生成 output/manifest.json 与 web 可直接读取的数据。

用法:
  python3 run_all.py
  /workspace/.venv/bin/python run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / ".pydeps"))

from quiltstudio.engine import QuiltConfig
from quiltstudio.pipeline import OUT_ROOT, execute, write_manifest
from quiltstudio.samples import build_prepared
import json


def main():
    prepared = build_prepared()
    runs = []

    def add(group, run_id, title, samples, **kw):
        cfg = QuiltConfig(**kw)
        print(f"[run] {run_id} ...", flush=True)
        rj = execute(samples, cfg, run_id=run_id, title=title, group=group)
        m = rj["metrics"]
        print(f"       status={rj['status']} seam={m['seam']['seam_rmse']:.4f} "
              f"straight={m['straight_seam']['seam_rmse']:.4f} "
              f"warn={len(rj['warnings'])} {m['elapsed_sec']}s", flush=True)
        runs.append(rj)

    # 1) 三类样本 × 3 个可复现种子（普通画布）
    for key, label in [("grain", "随机颗粒"), ("stripes", "定向条纹"),
                       ("markers", "独特标记")]:
        for seed in (101, 202, 303):
            add(key, f"{key}_std_s{seed}", f"{label} · 常规拼接 · seed {seed}",
                prepared[key], out_w=384, out_h=384, patch=64, overlap=24,
                seed=seed, periodic=False, policy="adaptive")

    # 2) 周期连接：首尾约束必须进入求解
    for key, label in [("grain", "随机颗粒"), ("stripes", "定向条纹"),
                       ("markers", "独特标记")]:
        for seed in (404, 505):
            add(f"{key}-periodic", f"{key}_per_s{seed}",
                f"{label} · 周期闭环 · seed {seed}",
                prepared[key], out_w=384, out_h=384, patch=64, overlap=24,
                seed=seed, periodic=True, policy="adaptive")

    # 3) 方向一致、局部锚点互斥的双样片
    pair = prepared["conflict"]
    for seed in (606, 707, 808):
        add("conflict", f"conflict_adapt_s{seed}",
            f"互斥锚点 A/B · 自适应(重排+重采样) · seed {seed}",
            pair, out_w=384, out_h=384, patch=64, overlap=24,
            seed=seed, periodic=True, policy="adaptive")
    add("conflict", "conflict_reroute_s606",
        "互斥锚点 A/B · 仅切缝重排 · seed 606",
        pair, out_w=384, out_h=384, patch=64, overlap=24,
        seed=606, periodic=True, policy="reroute")
    add("conflict", "conflict_resample_s606",
        "互斥锚点 A/B · 仅来源重采样 · seed 606",
        pair, out_w=384, out_h=384, patch=64, overlap=24,
        seed=606, periodic=True, policy="resample")

    # 4) 压力测试：patch 小于独特标记尺度，强结构下预期失败（展示生成限制）
    add("stress", "markers_stress_s909",
        "压力测试 · patch 小于标记尺度 · seed 909",
        prepared["markers"], out_w=384, out_h=384, patch=40, overlap=14,
        seed=909, periodic=False, policy="adaptive",
        seam_warn_rmse=0.085, seam_fail_rmse=0.16)
    add("stress", "markers_stress_per_s910",
        "压力测试 · 周期闭环 + 小 patch · seed 910",
        prepared["markers"], out_w=384, out_h=384, patch=44, overlap=15,
        seed=910, periodic=True, policy="adaptive")

    mp = write_manifest(runs)

    # 为 file:// 直接打开页面提供每个 run 的 JS 数据包
    runs_web = Path(__file__).resolve().parent / "web" / "data" / "runs"
    runs_web.mkdir(parents=True, exist_ok=True)
    for rj in runs:
        body = json.dumps(rj, ensure_ascii=False, default=str)
        (runs_web / f"{rj['run_id']}.js").write_text(
            f"window.QS_RUN_DATA={body};\n", encoding="utf-8")

    print(f"\nmanifest -> {mp}")
    print(f"runs: {len(runs)}  output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
