"""Tests for SetupAgent external MCP allowlist routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from models.mcp_config import McpToolInfo
from rp.tools.setup_scoped_mcp_manager import SetupScopedMcpManager


class _FakeSetupToolManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._tools = [
            McpToolInfo(
                server_id="rp_setup",
                server_name="RP Setup",
                name="setup.memory.search",
                description="Search setup memory",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    def get_all_tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((qualified_name, arguments))
        return {
            "success": True,
            "content": json.dumps({"source": "setup"}, sort_keys=True),
            "error_code": None,
        }


class _FakeExternalMcpManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._tools = [
            McpToolInfo(
                server_id="web",
                server_name="Web MCP",
                name="search",
                description="Search web",
                input_schema={"type": "object", "properties": {}},
            ),
            McpToolInfo(
                server_id="files",
                server_name="Files MCP",
                name="read_file",
                description="Read file",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    def list_server_views(self) -> list[Any]:
        return [
            SimpleNamespace(id="web", connected=True),
            SimpleNamespace(id="files", connected=True),
            SimpleNamespace(id="offline", connected=False),
        ]

    def get_server_tools(self, server_id: str) -> list[McpToolInfo]:
        return [tool for tool in self._tools if tool.server_id == server_id]

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((qualified_name, arguments))
        return {
            "success": True,
            "content": json.dumps({"source": "external"}, sort_keys=True),
            "error_code": None,
        }


def test_setup_scoped_mcp_manager_lists_setup_and_allowlisted_external_tools():
    manager = SetupScopedMcpManager(
        setup_tool_manager=_FakeSetupToolManager(),
        external_mcp_manager=_FakeExternalMcpManager(),
        external_tool_allowlist=["web__search"],
    )

    tools = manager.get_all_tools()
    qualified_names = {tool.qualified_name for tool in tools}

    assert "rp_setup__setup_memory_search" in qualified_names
    assert "web__search" in qualified_names
    assert "files__read_file" not in qualified_names


@pytest.mark.asyncio
async def test_setup_scoped_mcp_manager_delegates_allowed_external_call():
    setup = _FakeSetupToolManager()
    external = _FakeExternalMcpManager()
    manager = SetupScopedMcpManager(
        setup_tool_manager=setup,
        external_mcp_manager=external,
        external_tool_allowlist=["web__search"],
    )

    result = await manager.call_tool_by_qualified_name(
        qualified_name="web__search",
        arguments={"query": "moon forest"},
    )

    assert result["success"] is True
    assert external.calls == [("web__search", {"query": "moon forest"})]
    assert setup.calls == []


@pytest.mark.asyncio
async def test_setup_scoped_mcp_manager_rejects_disallowed_external_call():
    manager = SetupScopedMcpManager(
        setup_tool_manager=_FakeSetupToolManager(),
        external_mcp_manager=_FakeExternalMcpManager(),
        external_tool_allowlist=["web__search"],
    )

    result = await manager.call_tool_by_qualified_name(
        qualified_name="files__read_file",
        arguments={"path": "secret.txt"},
    )
    payload = json.loads(result["content"])

    assert result["success"] is False
    assert result["error_code"] == "UNKNOWN_TOOL"
    assert payload["error"]["code"] == "unknown_tool"
