"""CLI 与本地 Web 服务共用的生成入口。"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .engine import QuiltConfig
from .pipeline import OUT_ROOT, execute, upsert_manifest
from .samples import Sample, build_prepared, sample_from_file

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = OUT_ROOT / "uploads"

PREPARED: dict[str, list[Sample]] | None = None


def prepared() -> dict[str, list[Sample]]:
    global PREPARED
    if PREPARED is None:
        PREPARED = build_prepared()
    return PREPARED


def resolve_samples(params: dict) -> tuple[list[Sample], str]:
    kind = params.get("sample", "grain")
    if kind in prepared():
        return prepared()[kind], kind
    if kind == "upload":
        names = params.get("uploads") or []
        ss: list[Sample] = []
        for i, name in enumerate(names[:2]):
            p = (UPLOAD_DIR / Path(name).name).resolve()
            if UPLOAD_DIR.resolve() not in p.parents or not p.exists():
                raise ValueError(f"上传样本不存在: {name}")
            ss.append(sample_from_file(p, sample_id=f"upload{i}",
                                       name=f"用户纹理 {i + 1}"))
        if not ss:
            raise ValueError("请先上传至少一张纹理")
        return ss, "upload"
    if kind.startswith("path:"):
        paths = params["paths"]
        return [sample_from_file(p) for p in paths], "path"
    raise ValueError(f"未知样本组: {kind}")


def _cfg_from_params(params: dict) -> QuiltConfig:
    seed = params.get("seed")
    if seed in (None, "", "random"):
        seed = random.Random().randrange(1, 2 ** 31)
    kwargs = dict(
        out_w=int(params.get("out_w", 384)),
        out_h=int(params.get("out_h", 384)),
        patch=int(params.get("patch", 64)),
        overlap=int(params.get("overlap", 24)),
        seed=int(seed),
        periodic=bool(params.get("periodic", False)),
        policy=str(params.get("policy", "adaptive")),
        seam_mode=str(params.get("seam_mode", "mincut")),
    )
    for k in ("candidate_rel_tol", "seam_warn_rmse", "seam_fail_rmse"):
        if k in params and params[k] is not None:
            kwargs[k] = float(params[k])
    return QuiltConfig(**kwargs)


def _extend_spec_from_params(params: dict):
    """从请求参数构造受约束扩展规格；未提供扩展字段时返回 None。"""
    ext = params.get("extend")
    if not ext:
        return None
    from .extend import ExtendSpec
    r = ext.get("seed_rect")
    if not r or len(r) != 4:
        raise ValueError("扩展需要 seed_rect = [x0,y0,x1,y1]（框选样本区域）")
    place = ext.get("place_at") or [0, 0]
    lock = ext.get("lock_edge")
    forbidden = ext.get("forbidden") or []
    return ExtendSpec(
        out_w=int(params.get("out_w", 384)),
        out_h=int(params.get("out_h", 384)),
        patch=int(params.get("patch", 64)),
        overlap=int(params.get("overlap", 24)),
        seed=int(params.get("seed") or 20261126),
        periodic=bool(params.get("periodic", False)),
        policy=str(params.get("policy", "adaptive")),
        seed_rect=tuple(int(v) for v in r),
        place_at=(int(place[0]), int(place[1])),
        lock_edge=tuple(int(v) for v in lock) if lock else None,
        forbidden=[tuple(int(v) for v in f) for f in forbidden],
        borrow_endpoint=bool(ext.get("borrow_endpoint", False)),
    )


def generate(params: dict) -> dict:
    samples, group = resolve_samples(params)
    cfg = _cfg_from_params(params)
    es = _extend_spec_from_params(params)
    h = hashlib.sha1()
    h.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    rid = params.get("run_id") or f"{'extend' if es else 'custom'}_{group}_{cfg.seed}_{h.hexdigest()[:8]}"
    title = params.get("title") or (
        f"{'受约束扩展' if es else '自定义'} · {group} · seed {cfg.seed}"
        + (" · 周期" if cfg.periodic else ""))
    rj = execute(samples, cfg, run_id=rid, title=title,
                 group="extend" if es else "custom", extend_spec=es)
    upsert_manifest(rj)
    rj["reproduce_command"] = build_reproduce_command(params, cfg, es)
    return rj


def build_reproduce_command(params: dict, cfg: QuiltConfig, es=None) -> str:
    parts = ["python3 cli.py", "run",
             f"--sample {params.get('sample', 'grain')}",
             f"--seed {cfg.seed}",
             f"--width {cfg.out_w}", f"--height {cfg.out_h}",
             f"--patch {cfg.patch}", f"--overlap {cfg.overlap}",
             f"--policy {cfg.policy}"]
    if cfg.periodic:
        parts.append("--periodic")
    if params.get("uploads"):
        parts.append("--images " + ",".join(params["uploads"]))
    if es is not None:
        parts.append("--extend")
        parts.append("--seed-rect " + ",".join(str(v) for v in es.seed_rect))
        parts.append("--place-at " + ",".join(str(v) for v in es.place_at))
        if es.lock_edge:
            parts.append("--lock-edge " + ",".join(str(v) for v in es.lock_edge))
        for f in es.forbidden:
            parts.append("--forbid " + ",".join(str(v) for v in f))
        if es.borrow_endpoint:
            parts.append("--borrow-endpoint")
    return " \\\n  ".join(parts)
