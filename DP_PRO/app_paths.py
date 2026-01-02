from __future__ import annotations

import os
import sys


def resource_path(rel_path: str) -> str:
    """
    获取资源路径（兼容 PyInstaller）：
    - 开发态：基于项目目录
    - 打包态：基于 sys._MEIPASS
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(str(base), rel_path)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, rel_path)


def project_root() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(base)
    return os.path.dirname(os.path.abspath(__file__))


