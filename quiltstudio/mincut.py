"""重叠区域上的最小误差切缝（Efros–Freeman image quilting 风格的 DP 最小割）。

除经典最小割外，提供带禁行格与固定端点的受约束变体：禁区只截断条带内部
时可在其周围绕行；若端点被围死（首行/末行无可行列，或禁行把条带横向
切断到端点不可达），则判定为无解（返回 None），由上层决定重选邻块或报告。
"""
from __future__ import annotations

import numpy as np

INF = 1e18
BLOCKED = 1e15          # 单个禁行格代价值，远大于任何真实 SSE 之和
INFEASIBLE = 1e13       # 路径总代价达到此量级即视为无解


def min_cut_path(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """在 H×W 代价面上求从第 0 行到最后一行的三邻接最小代价路径。

    返回 path[r] = 第 r 行选中的列，以及沿路径的平均代价（每像素原始 SSE）。
    对水平切缝，调用方对 cost.T 调用本函数即可。
    """
    h, w = cost.shape
    if w == 1:
        path = np.zeros(h, dtype=np.int32)
        return path, float(cost[:, 0].mean())

    dp = np.empty((h, w), dtype=np.float64)
    parent = np.empty((h, w), dtype=np.int32)
    dp[0] = cost[0].astype(np.float64)

    for r in range(1, h):
        prev = dp[r - 1]
        left = np.empty(w, np.float64)
        mid = prev
        right = np.empty(w, np.float64)
        left[0] = INF
        left[1:] = prev[:-1]
        right[:-1] = prev[1:]
        right[-1] = INF
        stack = np.vstack([left, mid, right])
        choice = np.argmin(stack, axis=0)  # 0=左上, 1=正上, 2=右上
        idx = np.arange(w)
        best_prev = stack[choice, idx]
        dp[r] = cost[r] + best_prev
        parent[r] = idx + (choice - 1)

    end = int(np.argmin(dp[h - 1]))
    path = np.empty(h, dtype=np.int32)
    c = end
    total = 0.0
    for r in range(h - 1, -1, -1):
        path[r] = c
        total += float(cost[r, c])
        if r > 0:
            c = int(parent[r, c])
    return path, total / h


def min_cut_path_constrained(cost: np.ndarray, blocked: np.ndarray | None = None,
                             pin_start: int | None = None,
                             pin_end: int | None = None
                             ) -> tuple[np.ndarray, float] | tuple[None, float]:
    """带禁行格 / 端点固定的最小割；无可行路径时返回 (None, inf)。

    blocked: 与 cost 同形状的布尔掩膜，True 表示该格禁行。
    pin_start / pin_end: 固定首行（resp. 末行）选中列，用于“借用端点”。
    """
    h, w = cost.shape
    if w == 1:
        if blocked is not None and bool(blocked[:, 0].any()):
            return None, float("inf")
        if pin_start not in (None, 0) or pin_end not in (None, 0):
            return None, float("inf")
        return np.zeros(h, dtype=np.int32), float(cost[:, 0].mean())

    c = cost.astype(np.float64).copy()
    if blocked is not None:
        c[blocked] = BLOCKED
    if pin_start is not None and not (0 <= pin_start < w):
        return None, float("inf")
    if pin_end is not None and not (0 <= pin_end < w):
        return None, float("inf")

    dp = np.empty((h, w), dtype=np.float64)
    parent = np.full((h, w), -1, dtype=np.int32)
    dp[0] = c[0]
    if pin_start is not None:
        row = np.full(w, INF, np.float64)
        row[pin_start] = c[0, pin_start]
        dp[0] = row

    for r in range(1, h):
        prev = dp[r - 1]
        left = np.empty(w, np.float64)
        mid = prev
        right = np.empty(w, np.float64)
        left[0] = INF
        left[1:] = prev[:-1]
        right[:-1] = prev[1:]
        right[-1] = INF
        stack = np.vstack([left, mid, right])
        choice = np.argmin(stack, axis=0)
        idx = np.arange(w)
        best_prev = stack[choice, idx]
        dp[r] = c[r] + best_prev
        parent[r] = idx + (choice - 1)

    last = dp[h - 1]
    if pin_end is not None:
        end = pin_end if last[pin_end] < INFEASIBLE else None
    else:
        end = int(np.argmin(last))
        if last[end] >= INFEASIBLE:
            end = None
    if end is None:
        return None, float("inf")

    path = np.empty(h, dtype=np.int32)
    col = int(end)
    total = 0.0
    for r in range(h - 1, -1, -1):
        path[r] = col
        if blocked is not None and bool(blocked[r, col]):
            return None, float("inf")
        total += float(cost[r, col])
        if r > 0:
            col = int(parent[r, col])
            if col < 0:
                return None, float("inf")
    return path, total / h


def straight_path(cost: np.ndarray) -> np.ndarray:
    """硬切基线：永远从重叠区正中间直切。"""
    h, w = cost.shape
    return np.full(h, w // 2, dtype=np.int32)


def path_energy(cost: np.ndarray, path: np.ndarray) -> float:
    rows = np.arange(cost.shape[0])
    return float(cost[rows, path].mean())
