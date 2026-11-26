"""受约束纹理扩展回归：

- 框选样本并扩展：覆盖完整、受保护种子恒等、沿缝回读像素来源一致。
- 锁定细边：扩展前后逐位相等；锁在种子外则显式拒绝。
- 接缝禁区：只截断条带内部可局部绕开；围死端点则必须重选或报告无解。
- 完全堵塞再释放：释放后覆盖恢复完整，脏区域更新可达。
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
from quiltstudio.extend import ExtendSpec, extend
from quiltstudio.samples import build_prepared


class ExtendBasicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = build_prepared()
        cls.diag = cls.prep["diagonal"]

    def test_seed_extension_complete_and_protected(self):
        spec = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24, seed=3,
                          seed_rect=(0, 0, 96, 96), place_at=(0, 0))
        out = extend(self.diag, spec)
        r = out["report"]
        self.assertTrue(r["coverage"]["complete"])
        self.assertEqual(r["coverage"]["hole_pixels"], 0)
        self.assertTrue(r["protected"]["identical"])
        self.assertEqual(r["protected"]["changed_pixels"], 0)
        self.assertTrue(r["provenance"]["exact"])
        self.assertEqual(r["provenance"]["mismatch_pixels"], 0)

    def test_provenance_readback_on_seam(self):
        spec = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24, seed=3,
                          seed_rect=(0, 0, 96, 96), place_at=(0, 0))
        out = extend(self.diag, spec)
        side = out["report"]["provenance"]["seam_side"]
        self.assertTrue(side.get("available"))
        self.assertTrue(side.get("exact"), side)
        self.assertEqual(side.get("mismatch"), 0)


class LockedEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = build_prepared()
        cls.diag = cls.prep["diagonal"]

    def test_locked_edge_identical(self):
        spec = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24, seed=9,
                          seed_rect=(0, 0, 96, 96), place_at=(0, 0),
                          lock_edge=(30, 10, 32, 80), borrow_endpoint=True)
        out = extend(self.diag, spec)
        le = out["report"]["locked_edges"][0]
        self.assertTrue(le["identical"], le)
        self.assertEqual(le["changed_pixels"], 0)
        # 锁边不影响覆盖与种子保护
        self.assertTrue(out["report"]["coverage"]["complete"])
        self.assertTrue(out["report"]["protected"]["identical"])
        self.assertTrue(out["report"]["provenance"]["exact"])

    def test_lock_outside_seed_rejected(self):
        spec = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24, seed=9,
                          seed_rect=(0, 0, 96, 96), place_at=(0, 0),
                          lock_edge=(90, 10, 120, 80))
        with self.assertRaises(ValueError):
            extend(self.diag, spec)


class ForbiddenZoneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = build_prepared()
        cls.diag = cls.prep["diagonal"]
        xs = [0, 43, 85, 128]
        cls.base = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24,
                              seed=11, seed_rect=(0, 0, 43, 43),
                              place_at=(0, 0))

    def _run(self, zone):
        spec = ExtendSpec(**{**self.base.__dict__, "forbidden": [zone]})
        return extend(self.diag, spec)

    def test_interior_truncation_bypasses_locally(self):
        out = self._run((48, 60, 58, 80))   # 只截断 left 条带内部（宽 21 内）
        ev = [e for e in out["report"]["engine"]["constraint_events"]
              if e["ix"] == 1 and e["iy"] == 1]
        self.assertEqual(ev, [], "内部截断应被局部绕开，不应报无解")
        self.assertTrue(out["report"]["coverage"]["complete"])
        # 局部修补未暗中改掉受保护区域
        self.assertTrue(out["report"]["protected"]["identical"])

    def test_endpoint_surround_reports_unresolved_without_holes(self):
        out = self._run((43, 43, 64, 52))   # 围住 left 条带首行端点
        ev = [e for e in out["report"]["engine"]["constraint_events"]
              if e["ix"] == 1 and e["iy"] == 1]
        self.assertTrue(ev, "端点被围死应触发约束事件")
        self.assertEqual(ev[0]["resolved_by"], "none")
        # 关键：即便无解也不能留孔，受保护区域仍恒等
        self.assertTrue(out["report"]["coverage"]["complete"])
        self.assertTrue(out["report"]["protected"]["identical"])

    def test_block_then_release_recovers(self):
        blocked = self._run((43, 43, 64, 52))
        released = self._run((43, 43, 44, 44))  # 只留极小禁区，应可绕开
        self.assertTrue(blocked["report"]["coverage"]["complete"])
        self.assertTrue(released["report"]["coverage"]["complete"])
        # 释放后同一图块不再因端点围死而报无解
        ev = [e for e in released["report"]["engine"]["constraint_events"]
              if e["ix"] == 1 and e["iy"] == 1]
        self.assertEqual(ev, [])


class ConstrainedCoverageParityTest(unittest.TestCase):
    """约束不得破坏覆盖完整性：多种禁区/锁边组合下均无孔洞。"""

    @classmethod
    def setUpClass(cls):
        cls.prep = build_prepared()
        cls.diag = cls.prep["diagonal"]

    def test_many_zones_keep_coverage(self):
        zones = [(0, 0, 10, 10), (30, 30, 60, 60), (43, 70, 64, 76),
                 (85, 85, 100, 120), (120, 0, 140, 60)]
        for z in zones:
            with self.subTest(zone=z):
                spec = ExtendSpec(out_w=192, out_h=192, patch=64, overlap=24,
                                  seed=11, seed_rect=(0, 0, 43, 43),
                                  place_at=(0, 0), forbidden=[z])
                out = extend(self.diag, spec)
                self.assertTrue(out["report"]["coverage"]["complete"])
                self.assertEqual(out["report"]["protected"]["changed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
