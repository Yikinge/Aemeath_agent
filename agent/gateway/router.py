"""LLM 网关：封装 LiteLLM，按任务路由到不同模型（MODEL-1~3 的雏形）+ embedding。

业务只调 router.complete(...) / router.embed(...)，不直接碰各家 SDK；换模型只改配置。

兜底策略：litellm 未安装、或没有对应 Key、或调用失败时，complete 退回离线 mock 回复、
embed 退回内置哈希向量——保证「没装全依赖、没配 Key」也能把整条流程跑起来。
配好对应 Key 后自动切真模型 / 真向量。
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field


@dataclass
class AssistantTurn:
    """一轮助手输出：纯文本 + 可选的工具调用请求（OpenAI function-calling）。"""

    content: str
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments(str)}]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# 网关已知的对话 provider → 对应环境变量名
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_FALLBACK_DIM = 256


class LLMRouter:
    def __init__(
        self, default_model: str, fast_model: str,
        embed_model: str | None = None, embed_base_url: str | None = None,
        embed_api_key: str | None = None,
    ) -> None:
        self.default_model = default_model
        self.fast_model = fast_model
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key

    # ---------- 对话 ----------

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None,
        task: str = "default", **kwargs,
    ) -> AssistantTurn:
        """带工具的对话（工具循环用）。tools 为 OpenAI function 格式，可空。

        返回完整一轮 {content, tool_calls}；litellm 未装 / 无 Key / 调用失败时，
        退回无 tool_calls 的 mock turn——工具在离线态自动哑火，整链仍可跑。
        """
        model = self.fast_model if task == "fast" else self.default_model

        try:
            import litellm
        except ImportError:
            return AssistantTurn(self._mock_reply(messages, reason="litellm 未安装"))

        if not self._has_key_for(model):
            return AssistantTurn(self._mock_reply(messages, reason="未配置对应 provider 的 API Key"))

        if tools:
            kwargs["tools"] = tools
        try:
            resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
            msg = resp.choices[0].message
            calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                for tc in (getattr(msg, "tool_calls", None) or [])
            ]
            return AssistantTurn(content=msg.content or "", tool_calls=calls)
        except Exception as e:  # 网络/额度/模型名等问题都不该让 agent 崩溃
            return AssistantTurn(self._mock_reply(messages, reason=f"调用失败: {e}"))

    async def complete(self, messages: list[dict], *, task: str = "default", **kwargs) -> str:
        """纯文本对话（旧接口，向后兼容）：task='fast' 走便宜/快模型；忽略工具，只取最终文本。"""
        return (await self.chat(messages, task=task, **kwargs)).content

    def live(self, task: str = "default") -> bool:
        """是否具备真实对话模型调用条件（litellm 可用 + 对应 Key 在）。"""
        model = self.fast_model if task == "fast" else self.default_model
        try:
            import litellm  # noqa: F401
        except ImportError:
            return False
        return self._has_key_for(model)

    # ---------- 向量 ----------

    def embed_live(self) -> bool:
        """是否具备真实 embedding 条件（配了 embed_model + key）。"""
        return bool(self.embed_model and self.embed_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """把文本批量转向量。没配 embedding key 时退回内置哈希向量（保证可跑）。"""
        if not texts:
            return []
        if not self.embed_live():
            return [_fallback_embed(t) for t in texts]
        try:
            import litellm

            resp = await litellm.aembedding(
                model=self.embed_model, input=texts,
                api_base=self.embed_base_url, api_key=self.embed_api_key,
            )
            return [item["embedding"] for item in resp.data]
        except Exception:
            return [_fallback_embed(t) for t in texts]

    # ---------- 内部 ----------

    @staticmethod
    def _has_key_for(model: str) -> bool:
        """model 形如 '<provider>/<name>'。能取到该 provider 的 Key 才算具备真调用条件。"""
        provider = model.split("/", 1)[0] if "/" in model else ""
        env = _PROVIDER_KEYS.get(provider)
        return bool(env and os.environ.get(env))

    @staticmethod
    def _mock_reply(messages: list[dict], *, reason: str) -> str:
        """离线兜底：复述最近一条用户消息，并体现「记得」更早的对话，用于演示记忆闭环。"""
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        last = user_msgs[-1] if user_msgs else ""
        earlier = user_msgs[:-1]
        recall = f"（我还记得你之前说过：{earlier[0]}）" if earlier else ""
        return (
            f"[离线 mock · {reason}] 收到：「{last}」。{recall} "
            f"这条我先记下啦～ 配好任意一家 API Key 后我就换成真模型回复。"
        )


def _fallback_embed(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    """无 embedding 服务时的确定性兜底向量：字符 3-gram 哈希到固定维 + L2 归一化。

    只够把向量库/检索/注入整条流程跑通与验证；语义质量有限，配好真 embedder 后自动替换。
    """
    vec = [0.0] * dim
    s = f"  {text}  "
    for i in range(len(s) - 2):
        gram = s[i : i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
