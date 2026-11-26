"""重叠区域上的最小误差切缝（Efros–Freeman image quilting 风格的 DP 最小割）。"""
from __future__ import annotations

import numpy as np

INF = 1e18


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


def straight_path(cost: np.ndarray) -> np.ndarray:
    """硬切基线：永远从重叠区正中间直切。"""
    h, w = cost.shape
    return np.full(h, w // 2, dtype=np.int32)


def path_energy(cost: np.ndarray, path: np.ndarray) -> float:
    rows = np.arange(cost.shape[0])
    return float(cost[rows, path].mean())
