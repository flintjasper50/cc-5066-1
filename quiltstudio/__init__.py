"""样本驱动纹理扩展：重叠匹配 + 最小割接缝 + 来源追溯 + 周期约束求解。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEPS = _ROOT / ".pydeps"
if _DEPS.is_dir() and str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))

__version__ = "1.0.0"
