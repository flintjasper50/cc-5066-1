"""回归：锚点/区域几何必须覆盖整块画布，不允许孔洞、负区域或同图块内重复覆盖。

这些用例是先把“窄重叠 + 非整除画布边”缩成最小复现，再作为修复后的守卫。
运行: /workspace/.venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pydeps"))

import numpy as np

from quiltstudio.engine import QuiltConfig, QuiltEngine, normalize_config
from quiltstudio.samples import build_prepared


CASES = [
    # (W, H, patch, overlap, periodic) —— 窄重叠 / 画布边 / 非整除
    (200, 200, 64, 4, False),
    (257, 257, 64, 4, False),
    (129, 127, 64, 4, False),
    (130, 130, 64, 4, False),
    (400, 300, 64, 2, False),
    (384, 384, 64, 24, False),
    (128, 128, 64, 24, False),
    (128, 128, 64, 2, True),
    (200, 200, 64, 4, True),
    (257, 257, 64, 4, True),
    (129, 128, 64, 4, True),
    (300, 300, 64, 8, True),
]


class GeometryCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = build_prepared()

    def _run(self, W, H, patch, ov, periodic):
        cfg = QuiltConfig(out_w=W, out_h=H, patch=patch, overlap=ov,
                          seed=7, periodic=periodic)
        cfg, _ = normalize_config(cfg, self.prep["grain"])
        eng = QuiltEngine(self.prep["grain"], cfg)
        return cfg, eng.run()

    def test_no_holes_and_complete_coverage(self):
        for W, H, patch, ov, periodic in CASES:
            with self.subTest(W=W, H=H, patch=patch, ov=ov, periodic=periodic):
                cfg, res = self._run(W, H, patch, ov, periodic)
                own = res["own"]
                holes = int((own < 0).sum())
                self.assertEqual(holes, 0, f"覆盖孔洞 {holes} 像素")
                self.assertTrue((res["map_sample"] >= 0).all())

    def test_regions_partition_patch(self):
        """每个图块的重叠区域必须在 [0,b) 内且两两不重叠，并拼满 b×b。"""
        from quiltstudio.geometry import build_regions, partition_report
        for W, H, patch, ov, periodic in CASES:
            cfg = QuiltConfig(out_w=W, out_h=H, patch=patch, overlap=ov,
                              seed=7, periodic=periodic)
            cfg, _ = normalize_config(cfg, self.prep["grain"])
            from quiltstudio.geometry import build_anchors
            xs = build_anchors(W, cfg.patch, cfg.overlap, periodic)
            ys = build_anchors(H, cfg.patch, cfg.overlap, periodic)
            for iy in range(len(ys)):
                for ix in range(len(xs)):
                    regs = build_regions(ix, iy, xs, ys, W, H, cfg.patch, periodic)
                    rep = partition_report(regs, cfg.patch)
                    with self.subTest(W=W, H=H, ov=ov, per=periodic, ix=ix, iy=iy):
                        self.assertEqual(rep["out_of_bounds"], 0)
                        self.assertEqual(rep["overlap_pixels"], 0)
                        self.assertEqual(rep["covered_pixels"] + rep["new_pixels"],
                                         cfg.patch * cfg.patch)


if __name__ == "__main__":
    unittest.main()
