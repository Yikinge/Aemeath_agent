# Runtime Data

This directory is intentionally empty in Git.

At runtime the agent creates local state here, including:

- `agent.db`
- `MEMORY.md`
- `agent_db.md`
- `journal/*.md`
- `mcp.json`

These files may contain private memories, chat history, API tokens, or local MCP
configuration. Do not commit them.

