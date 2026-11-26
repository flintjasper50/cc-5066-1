"""受约束纹理扩展的语义对象：锁定细边、接缝禁区、端点借用。

坐标：全部为输出画布绝对坐标，半开区间 [x0,x1) × [y0,y1)。
周期模式下由引擎在映射条带时取模，约束仍按画布坐标书写。

语义（关键，决定覆盖完整性为什么恒成立）：
- 锁定细边（LockedEdge）：其像素在扩展前快照，扩展过程中永不重写，
  因而扩展后必须逐位恒等；它们不计入“新写入”，也不产生孔洞。
- 接缝禁区（ForbiddenZone）：只约束**接缝路径**能否经过，不改变像素归属。
  某条带的最小割若必须穿过禁区，则判定该条带无解 → 保留旧邻块像素，
  因此画布仍被完整覆盖，不会因禁区产生孔洞。
- 端点借用（Pin）：把指定图块指定条带接缝的首/末端局部坐标钉住，
  常用来让另一条接缝的端点贴合锁定细边的端点。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)

    def clipped(self, w: int, h: int) -> "Rect":
        return Rect(max(0, self.x0), max(0, self.y0),
                    min(w, self.x1), min(h, self.y1))

    def as_dict(self) -> dict:
        return {"x0": int(self.x0), "y0": int(self.y0),
                "x1": int(self.x1), "y1": int(self.y1)}

    @staticmethod
    def from_dict(d: dict) -> "Rect":
        return Rect(int(d["x0"]), int(d["y0"]), int(d["x1"]), int(d["y1"]))


@dataclass
class LockedEdge:
    """不能重写的细边：扩展前后像素必须逐位相等。"""
    rect: Rect
    name: str = "locked"
    pixels: np.ndarray | None = None

    def snapshot(self, canvas: np.ndarray) -> "LockedEdge":
        r = self.rect
        self.pixels = canvas[r.y0:r.y1, r.x0:r.x1].copy()
        return self

    def verify(self, canvas: np.ndarray) -> dict:
        r = self.rect
        cur = canvas[r.y0:r.y1, r.x0:r.x1]
        if self.pixels is None or cur.shape != self.pixels.shape:
            return {"name": self.name, "identical": False,
                    "changed_pixels": int(cur.size // 3),
                    "pixels": int(cur.size // 3),
                    "detail": "缺少快照或形状变化"}
        changed = int(np.any(cur != self.pixels, axis=-1).sum())
        return {"name": self.name, "identical": changed == 0,
                "changed_pixels": changed,
                "pixels": int(cur.shape[0] * cur.shape[1]),
                "max_abs_delta": float(np.abs(cur - self.pixels).max())
                if changed else 0.0}


@dataclass
class ForbiddenZone:
    """接缝禁区：最小割路径不得经过这些输出像素。"""
    rect: Rect
    name: str = "forbidden"

    def blocked_on(self, gy: np.ndarray, gx: np.ndarray,
                   h: int, w: int, periodic: bool) -> np.ndarray:
        """给定条带的绝对行/列向量，返回该条带的 blocked 布尔矩阵。"""
        ry, rx = gy, gx
        if periodic:
            ry = ry % h
            rx = rx % w
        inside_y = (ry >= self.rect.y0) & (ry < self.rect.y1)
        inside_x = (rx >= self.rect.x0) & (rx < self.rect.x1)
        return inside_y[:, None] & inside_x[None, :]


@dataclass
class Pin:
    """把某图块某条带接缝的首/末端局部坐标固定（端点借用）。"""
    ix: int
    iy: int
    kind: str                     # left | top | close_right | close_bottom
    start: int | None = None
    end: int | None = None
    reason: str = ""


@dataclass
class SeamConstraints:
    locked: list[LockedEdge] = field(default_factory=list)
    forbidden: list[ForbiddenZone] = field(default_factory=list)
    pins: list[Pin] = field(default_factory=list)
    # 扩展前已预先铺好、需要排除在“新写入”之外的锁定矩形（含种子区）
    prefilled_rects: list[Rect] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(self.locked or self.forbidden or self.pins)

    def pin_for(self, ix: int, iy: int, kind: str) -> Pin | None:
        for p in self.pins:
            if p.ix == ix and p.iy == iy and p.kind == kind:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "locked": [{"name": le.name, "rect": le.rect.as_dict()}
                       for le in self.locked],
            "forbidden": [{"name": fz.name, "rect": fz.rect.as_dict()}
                          for fz in self.forbidden],
            "pins": [{"ix": p.ix, "iy": p.iy, "kind": p.kind,
                      "start": p.start, "end": p.end, "reason": p.reason}
                     for p in self.pins],
            "prefilled": [r.as_dict() for r in self.prefilled_rects],
        }


def locked_mask(locked: list[LockedEdge], h: int, w: int) -> np.ndarray:
    keep = np.zeros((h, w), bool)
    for le in locked:
        r = le.rect.clipped(w, h)
        keep[r.y0:r.y1, r.x0:r.x1] = True
    return keep
