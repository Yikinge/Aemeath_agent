"""从 Python 函数签名 + 类型注解生成 OpenAI function 的 JSON Schema。

够用即可：基础类型映射 + 必填判定（无默认值即必填）。复杂参数可在 @tool 显式传 parameters。
"""

from __future__ import annotations

import inspect
from typing import Callable

# Python 类型 → JSON Schema type
_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
# 类型名（字符串）→ JSON type：兼容 `from __future__ import annotations` 下注解为字符串的情况
_NAME_TO_JSON = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
}

# 这些参数是依赖注入用的，不进对外 schema
_SKIP_PARAMS = {"ctx", "self"}


def _json_type(annotation: object) -> str:
    """把注解映射成 JSON 类型；`str | None` / 容器 / 未知一律退回首个已知类型或 string。"""
    # 字符串注解（PEP 563 / future annotations）：取首个已知类型名，如 'int | None' → int
    if isinstance(annotation, str):
        base = annotation.replace(" ", "").split("|", 1)[0].split("[", 1)[0]
        return _NAME_TO_JSON.get(base, "string")
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation]  # type: ignore[index]
    # 处理 typing 容器（list[str] 的 origin 是 list 等）
    origin = getattr(annotation, "__origin__", None)
    if origin in _PY_TO_JSON:
        return _PY_TO_JSON[origin]
    return "string"


def build_schema(fn: Callable) -> dict:
    """从函数签名构造 {"type":"object","properties":{...},"required":[...]}。"""
    sig = inspect.signature(fn)
    props: dict[str, dict] = {}
    required: list[str] = []
    for name, p in sig.parameters.items():
        if name in _SKIP_PARAMS:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[name] = {"type": _json_type(p.annotation)}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def first_line(doc: str | None) -> str:
    """取 docstring 首行作工具描述。"""
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""
