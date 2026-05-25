"""SetupAgent-scoped view over local setup tools plus allowed external MCP tools."""

from __future__ import annotations

import json
from typing import Any, Protocol

from models.mcp_config import McpServerView, McpToolInfo


class _SetupToolManagerLike(Protocol):
    def get_all_tools(self) -> list[McpToolInfo]: ...

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


class _ExternalMcpManagerLike(Protocol):
    def list_server_views(self) -> list[McpServerView]: ...

    def get_server_tools(self, server_id: str) -> list[McpToolInfo]: ...

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


class SetupScopedMcpManager:
    """Expose setup private tools and explicitly enabled external MCP tools.

    SetupAgent keeps its own setup-local tool provider because those tools are
    workspace/session scoped. External MCP tools still come from the chat-side
    MCP manager; this adapter only filters them by a tool-level allowlist before
    the runtime can show or execute them.
    """

    def __init__(
        self,
        *,
        setup_tool_manager: _SetupToolManagerLike,
        external_mcp_manager: _ExternalMcpManagerLike,
        external_tool_allowlist: list[str],
    ) -> None:
        self._setup_tool_manager = setup_tool_manager
        self._external_mcp_manager = external_mcp_manager
        self._external_tool_allowlist = {
            item.strip() for item in external_tool_allowlist if item.strip()
        }

    def get_all_tools(self) -> list[McpToolInfo]:
        return [
            *self._setup_tool_manager.get_all_tools(),
            *self._allowed_external_tools(),
        ]

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self._find_tool(
            self._setup_tool_manager.get_all_tools(),
            qualified_name=qualified_name,
        ):
            return await self._setup_tool_manager.call_tool_by_qualified_name(
                qualified_name=qualified_name,
                arguments=arguments,
            )

        external_tool = self._find_tool(
            self._allowed_external_tools(),
            qualified_name=qualified_name,
        )
        if external_tool is not None:
            return await self._external_mcp_manager.call_tool_by_qualified_name(
                qualified_name=qualified_name,
                arguments=arguments,
            )

        return {
            "success": False,
            "content": json.dumps(
                {
                    "error": {
                        "code": "unknown_tool",
                        "message": f"Unknown or disallowed setup tool: {qualified_name}",
                    }
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "error_code": "UNKNOWN_TOOL",
        }

    def _allowed_external_tools(self) -> list[McpToolInfo]:
        if not self._external_tool_allowlist:
            return []

        tools: list[McpToolInfo] = []
        for server in self._external_mcp_manager.list_server_views():
            if not server.connected:
                continue
            for tool in self._external_mcp_manager.get_server_tools(server.id):
                if self._is_external_tool_allowed(tool):
                    tools.append(tool)
        return tools

    def _is_external_tool_allowed(self, tool: McpToolInfo) -> bool:
        return any(
            alias in self._external_tool_allowlist
            for alias in tool.qualified_name_aliases
        )

    @staticmethod
    def _find_tool(
        tools: list[McpToolInfo],
        *,
        qualified_name: str,
    ) -> McpToolInfo | None:
        return next(
            (
                tool
                for tool in tools
                if qualified_name in (*tool.qualified_name_aliases, tool.name)
            ),
            None,
        )
