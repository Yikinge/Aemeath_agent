"""计算器：精确算数（模型心算不可靠）。安全的 AST 求值，不碰函数/变量/属性。"""

from __future__ import annotations

import ast
import operator

from agent.tools.registry import ToolContext, tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允许数字")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("不支持的表达式")


@tool
async def calculate(ctx: ToolContext, expression: str) -> str:
    """计算一个数学表达式（+ - * / // % ** 与括号）。需要精确算数时调用，别心算。
    例：calculate("(1234*56 + 78) / 9")。"""
    try:
        result = _eval(ast.parse(expression, mode="eval"))
    except Exception as e:
        return f"[算不了] {e}"
    # 整数结果去掉多余小数
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"
