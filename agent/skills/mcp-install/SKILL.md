---
name: 安装 MCP
description: 当用户贴出一段 MCP 服务器配置 JSON（含 mcpServers、或 command、或 url）并希望接入/安装/添加这个 MCP 时使用。
---

# 安装 MCP 技能

用户想给你接入一个新的 MCP 工具来源时（通常会贴一段从网上复制的 JSON）：

1. 从用户消息里**原样提取那段 JSON**。支持几种写法，都不用改：
   - Claude Desktop / Cursor 标准：`{"mcpServers": {"名字": {"command": "...", "args": [...]}}}`
   - 远程/托管：`{"mcpServers": {"名字": {"url": "https://...", "headers": {...}}}}`
   - 单个 server：`{"command": "...", "args": [...]}` 或 `{"url": "..."}`
2. 调用 `install_mcp` 工具，把那段 JSON 字符串原样作为 `config_json` 传入。
3. 安装成功后，简短报告新增了哪些工具（形如 `mcp__<名字>__<工具>`），并说一句「装好了，直接说需求就行」。
4. 失败就把错误原样说清楚（多半是命令没装、URL 不通或 JSON 格式问题），给个下一步建议。

注意：装好的 server 会存盘，重启自动重连，不用重复装。
