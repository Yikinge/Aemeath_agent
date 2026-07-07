"""配置加载：从 config.toml 读取；token 等敏感项环境变量优先于明文。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（agent/ 包的上一级）。相对路径都以它为基准，跟运行目录无关
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 自动加载项目根目录下的 .env（含 API Key）；未装 python-dotenv 时静默跳过
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_SYSTEM = (
    "你是一个长期陪伴的个人智能体，像一个对我有兴趣、想进一步了解我的新朋友。"
    "说话自然、简洁、有温度，不油腻。"
)


@dataclass
class Config:
    default_model: str
    fast_model: str
    embed_model: str | None
    embed_base_url: str | None
    embed_api_key: str | None
    telegram_token: str
    allow_from: list[str]
    db_path: str
    system_prompt: str
    soul_path: str
    console_enabled: bool
    console_host: str
    console_port: int
    memory_md_path: str
    consolidate_threshold: int
    # 记忆生命周期（方案 §13）
    memory_timezone: str
    core_max_commitments: int
    recent_mood_days: int
    narrative_similarity_threshold: float
    # 工具 / MCP / 技能（TOOL-*）
    tools_enabled: bool
    tool_deny: list[str]
    tools_lazy_mcp: bool
    tools_max_eager: int
    skills_dir: str
    mcp_config_path: str
    mcp_timeout: float


def load_config(path: str = "config.toml") -> Config:
    path = os.environ.get("AGENT_CONFIG", path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    llm = data.get("llm", {})
    tg = data.get("telegram", {})
    storage = data.get("storage", {})
    persona = data.get("persona", {})
    embedding = data.get("embedding", {})
    console = data.get("console", {})
    memory = data.get("memory", {})
    tools = data.get("tools", {})
    skills = data.get("skills", {})
    mcp = data.get("mcp", {})

    # 技能目录：默认 agent/skills；支持 ~ 展开，相对路径锚定项目根
    skills_dir = os.path.expanduser(skills.get("dir", "agent/skills"))
    if not os.path.isabs(skills_dir):
        skills_dir = str(_PROJECT_ROOT / skills_dir)

    # MCP 配置文件（唯一真相源，标准 mcpServers 格式）：手编/聊天装/热更新都读写它
    mcp_config_path = os.path.expanduser(mcp.get("path", "data/mcp.json"))
    if not os.path.isabs(mcp_config_path):
        mcp_config_path = str(_PROJECT_ROOT / mcp_config_path)

    md_path = os.path.expanduser(memory.get("md_path", "data/MEMORY.md"))
    if not os.path.isabs(md_path):
        md_path = str(_PROJECT_ROOT / md_path)
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)

    # 支持 ~ 展开；相对路径锚定到项目根目录（而非运行目录），避免到处生成 data/
    db_path = os.path.expanduser(storage.get("db_path", "data/agent.db"))
    if not os.path.isabs(db_path):
        db_path = str(_PROJECT_ROOT / db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    soul_path = os.path.expanduser(persona.get("soul_path", "SOUL.md"))
    if not os.path.isabs(soul_path):
        soul_path = str(_PROJECT_ROOT / soul_path)

    # token 优先取环境变量，方便不把密钥写进文件
    token = os.environ.get("TELEGRAM_BOT_TOKEN", tg.get("token", ""))
    allow_from_raw = os.environ.get("TELEGRAM_ALLOW_FROM")
    allow_from = (
        [x.strip() for x in allow_from_raw.split(",") if x.strip()]
        if allow_from_raw is not None else tg.get("allow_from", [])
    )
    console_host = os.environ.get("CONSOLE_HOST", str(console.get("host", "127.0.0.1")))
    console_port = int(os.environ.get("CONSOLE_PORT", console.get("port", 8787)))

    return Config(
        default_model=llm.get("default_model", "deepseek/deepseek-v4-flash"),
        fast_model=llm.get("fast_model", "deepseek/deepseek-v4-flash"),
        embed_model=embedding.get("model"),
        embed_base_url=embedding.get("base_url"),
        embed_api_key=os.environ.get("SILICONFLOW_API_KEY") or None,
        telegram_token=token,
        allow_from=allow_from,
        db_path=db_path,
        system_prompt=persona.get("system_prompt", DEFAULT_SYSTEM),
        soul_path=soul_path,
        console_enabled=bool(console.get("enabled", True)),
        console_host=console_host,
        console_port=console_port,
        memory_md_path=md_path,
        consolidate_threshold=int(memory.get("consolidate_threshold", 2)),
        memory_timezone=str(memory.get("timezone", "Asia/Shanghai")),
        core_max_commitments=int(memory.get("core_max_commitments", 5)),
        recent_mood_days=int(memory.get("recent_mood_days", 14)),
        narrative_similarity_threshold=float(memory.get("narrative_similarity_threshold", 0.82)),
        tools_enabled=bool(tools.get("enabled", True)),
        tool_deny=list(tools.get("deny", [])),
        tools_lazy_mcp=bool(tools.get("lazy_mcp", False)),
        tools_max_eager=int(tools.get("max_eager", 25)),
        skills_dir=skills_dir,
        mcp_config_path=mcp_config_path,
        mcp_timeout=float(mcp.get("timeout", 30.0)),
    )
