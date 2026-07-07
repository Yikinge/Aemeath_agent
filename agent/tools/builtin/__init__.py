"""原生工具自动发现（drop-in #1）。

加一个工具 = 在本目录新建一个 .py，写个带 @tool 的 async 函数；启动时 discover 自动入注册表。
"""

from __future__ import annotations

import importlib
import pkgutil

from agent.tools.registry import ToolRegistry


def discover_builtin(registry: ToolRegistry) -> int:
    """扫描本包下所有模块，把被 @tool 装饰（带 _tool 属性）的函数注册进来。返回注册数。"""
    import agent.tools.builtin as pkg

    n = 0
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"agent.tools.builtin.{mod.name}")
        for obj in vars(m).values():
            spec = getattr(obj, "_tool", None)
            if spec is not None and spec.source == "builtin":
                registry.register(spec)
                n += 1
    return n
