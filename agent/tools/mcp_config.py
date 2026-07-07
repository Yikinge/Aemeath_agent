"""MCP 配置：单一真相源 = 一个标准 mcpServers JSON 文件（默认 data/mcp.json）。

手写编辑、聊天里贴 JSON 装（install_mcp）、热更新——读写的都是这一个文件，标准格式：
    {"mcpServers": {"名字": {"command": ...} 或 {"url": ..., "headers": ...}}}
对外用内部 cfg dict（name/transport/command|url/...），加载时归一化、保存时转回标准格式。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_HTTP_ALIASES = {"http", "streamable_http", "streamable-http", "streamablehttp"}


def _entry_to_cfg(name: str, s: dict) -> dict:
    """标准格式的单个 server 条目 → 内部 cfg。含 command→stdio，含 url→http（type:sse 则 sse）。"""
    cfg: dict = {"name": s.get("name", name), "enabled": bool(s.get("enabled", True))}
    ttype = (s.get("transport") or s.get("type") or "").lower()
    if "command" in s:
        cfg["transport"] = ttype or "stdio"
        cfg["command"] = s["command"]
        cfg["args"] = s.get("args", [])
        if s.get("env"):
            cfg["env"] = s["env"]
    elif "url" in s:
        cfg["transport"] = "sse" if ttype == "sse" else (ttype if ttype and ttype not in _HTTP_ALIASES else "http")
        cfg["url"] = s["url"]
        if s.get("headers"):
            cfg["headers"] = s["headers"]
    else:
        raise ValueError(f"server「{name}」既无 command 也无 url，无法识别传输方式")
    for k in ("dangerous", "dangerous_tools"):
        if k in s:
            cfg[k] = s[k]
    return cfg


def normalize_mcp_obj(data: dict) -> list[dict]:
    """把一段 MCP 配置对象解析成内部 cfg 列表。接受三种写法：
    {"mcpServers": {名字: {...}}} / 裸 {名字: {...}} / 单个 {command|url, name?}。"""
    if not isinstance(data, dict):
        raise ValueError("MCP 配置必须是 JSON 对象")
    if "mcpServers" in data and isinstance(data["mcpServers"], dict):
        servers_map = data["mcpServers"]
    elif "command" in data or "url" in data:
        servers_map = {data.get("name", "server"): data}
    else:
        servers_map = data
    out = [_entry_to_cfg(name, s) for name, s in servers_map.items() if isinstance(s, dict)]
    if not out:
        raise ValueError("没解析出任何 server")
    return out


def normalize_mcp_json(raw: str) -> list[dict]:
    """解析一段 MCP 配置 JSON 字符串（install_mcp 用）。"""
    return normalize_mcp_obj(json.loads(raw))


def cfg_to_entry(cfg: dict) -> dict:
    """内部 cfg → 标准格式的单个 server 条目（写文件时用，保持人类可读/可粘贴）。"""
    if cfg.get("url"):
        entry: dict = {"url": cfg["url"]}
        if cfg.get("transport") and cfg["transport"] != "http":
            entry["type"] = cfg["transport"]
        if cfg.get("headers"):
            entry["headers"] = cfg["headers"]
    else:
        entry = {"command": cfg["command"]}
        if cfg.get("args"):
            entry["args"] = cfg["args"]
        if cfg.get("env"):
            entry["env"] = cfg["env"]
    for k in ("dangerous", "dangerous_tools"):
        if k in cfg:
            entry[k] = cfg[k]
    if cfg.get("enabled") is False:
        entry["enabled"] = False
    return entry


def load_servers(path: str) -> list[dict]:
    """读 MCP 配置文件 → 内部 cfg 列表（标准 mcpServers 格式，归一化）。文件不存在/坏 → 空。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        return normalize_mcp_obj(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        log.warning("读取 MCP 配置失败 %s：%s", path, e)
        return []


def save_server(path: str, cfg: dict) -> None:
    """把一个 server 写进 MCP 配置文件（标准格式，按 name 覆盖）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    servers = data.get("mcpServers") if isinstance(data.get("mcpServers"), dict) else {}
    servers[cfg["name"]] = cfg_to_entry(cfg)
    data["mcpServers"] = servers
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
