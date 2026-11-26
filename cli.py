#!/usr/bin/env python3
"""命令行入口：生成样本、单次合成、跑全矩阵、启动界面服务。

示例:
  python3 cli.py samples
  python3 cli.py run --sample stripes --seed 42 --periodic
  python3 cli.py run --images my.png other.png --periodic --policy adaptive
  python3 cli.py batch
  python3 cli.py serve --port 8765
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / ".pydeps"))

from quiltstudio.api import generate, prepared, UPLOAD_DIR
from quiltstudio.imgio import save_png
from quiltstudio.pipeline import OUT_ROOT


def cmd_samples(args):
    out = OUT_ROOT / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for key, ss in prepared().items():
        for i, s in enumerate(ss):
            name = f"{key}.png" if len(ss) == 1 else f"{key}_{i}.png"
            save_png(s.image, out / name)
            print("saved", out / name)


def cmd_run(args):
    params = {
        "sample": args.sample,
        "seed": args.seed,
        "out_w": args.width, "out_h": args.height,
        "patch": args.patch, "overlap": args.overlap,
        "periodic": args.periodic, "policy": args.policy,
    }
    if args.extend:
        if not args.seed_rect:
            raise SystemExit("--extend 需要 --seed-rect x0,y0,x1,y1")
        def rect(s):
            vals = [int(v) for v in s.split(",")]
            if len(vals) != 4:
                raise SystemExit(f"矩形需 4 个整数: {s}")
            return vals

        def pt(s):
            vals = [int(v) for v in s.split(",")]
            if len(vals) != 2:
                raise SystemExit(f"坐标需 2 个整数: {s}")
            return vals
        params["extend"] = {
            "seed_rect": rect(args.seed_rect),
            "place_at": pt(args.place_at) if args.place_at else [0, 0],
            "lock_edge": rect(args.lock_edge) if args.lock_edge else None,
            "forbidden": [rect(f) for f in (args.forbid or [])],
            "borrow_endpoint": args.borrow_endpoint,
        }
    if args.images:
        # 把用户图片复制进 uploads，保持与服务一致的解析路径
        import shutil
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        names = []
        for p in args.images:
            src = Path(p)
            dst = UPLOAD_DIR / src.name
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            names.append(src.name)
        params["sample"] = "upload"
        params["uploads"] = names
    rj = generate(params)
    out = {
        "run_id": rj["run_id"], "status": rj["status"],
        "warnings": rj["warnings"], "infos": rj["infos"],
        "metrics": {"seam_rmse": rj["metrics"]["seam"]["seam_rmse"],
                    "straight_rmse": rj["metrics"]["straight_seam"]["seam_rmse"],
                    "periodic": rj["metrics"]["periodic"]},
        "result": rj["files"].get("result"),
        "run_json": f"output/runs/{rj['run_id']}/run.json",
        "reproduce": rj["reproduce_command"],
    }
    if "extension" in rj:
        out["extension"] = rj["metrics"]["extension"]
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_batch(args):
    import run_all
    run_all.main()


def cmd_serve(args):
    from server import serve
    serve(args.host, args.port)


def main():
    ap = argparse.ArgumentParser(prog="cli.py", description="样本驱动纹理扩展工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("samples", help="导出内置检验样本").set_defaults(func=cmd_samples)

    r = sub.add_parser("run", help="执行一次合成")
    r.add_argument("--sample", default="grain",
                   choices=["grain", "stripes", "markers", "conflict", "diagonal"])
    r.add_argument("--images", nargs="*", help="用户自制纹理路径（可给 1~2 张）")
    r.add_argument("--seed", type=int, default=20261126)
    r.add_argument("--width", type=int, default=384)
    r.add_argument("--height", type=int, default=384)
    r.add_argument("--patch", type=int, default=64)
    r.add_argument("--overlap", type=int, default=24)
    r.add_argument("--periodic", action="store_true", help="求解周期首尾约束")
    r.add_argument("--policy", default="adaptive",
                   choices=["off", "reroute", "resample", "adaptive"])
    r.add_argument("--extend", action="store_true",
                   help="受约束扩展：先框选样本区作受保护种子再向外拼接")
    r.add_argument("--seed-rect", help="框选样本区域 x0,y0,x1,y1（样本坐标）")
    r.add_argument("--place-at", help="框选落位到画布左上角 x,y")
    r.add_argument("--lock-edge", help="锁定细边 x0,y0,x1,y1（画布坐标，须在框选区内）")
    r.add_argument("--forbid", action="append",
                   help="接缝禁区 x0,y0,x1,y1（可重复）")
    r.add_argument("--borrow-endpoint", action="store_true",
                   help="让相邻接缝借用锁定细边端点")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="运行全部 22 个检验实验")
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("serve", help="启动本地 Web 界面")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
