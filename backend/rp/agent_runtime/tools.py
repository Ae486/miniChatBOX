"""Tool-registry and execution adapters for the RP runtime."""

from __future__ import annotations

import json
from typing import Any, Protocol

from models.mcp_config import McpToolInfo

from .contracts import RuntimeToolCall, RuntimeToolResult


class RuntimeToolManagerLike(Protocol):
    def get_all_tools(self) -> list[McpToolInfo]: ...

    async def call_tool_by_qualified_name(
        self,
        *,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


class RuntimeToolRegistryView:
    """Filter current MCP/local tools into runtime-visible definitions."""

    def __init__(self, *, mcp_manager: RuntimeToolManagerLike) -> None:
        self._mcp_manager = mcp_manager

    def get_visible_tools(self, *, visible_tool_names: list[str]) -> list[McpToolInfo]:
        tools = self._mcp_manager.get_all_tools()
        if not visible_tool_names:
            return tools

        allowed = set(visible_tool_names)
        return [
            tool
            for tool in tools
            if tool.name in allowed
            or any(alias in allowed for alias in tool.qualified_name_aliases)
        ]

    def get_openai_tool_definitions(
        self,
        *,
        visible_tool_names: list[str],
    ) -> list[dict[str, Any]]:
        return [
            tool.to_openai_tool()
            for tool in self.get_visible_tools(visible_tool_names=visible_tool_names)
        ]

    def resolve_tool(
        self,
        *,
        tool_name: str,
        visible_tool_names: list[str],
    ) -> McpToolInfo | None:
        for tool in self.get_visible_tools(visible_tool_names=visible_tool_names):
            if tool_name in (*tool.qualified_name_aliases, tool.name):
                return tool
        return None


class RuntimeToolExecutor:
    """Runtime-facing wrapper over McpManager/local tool providers."""

    def __init__(self, *, mcp_manager: RuntimeToolManagerLike) -> None:
        self._mcp_manager = mcp_manager
        self._registry = RuntimeToolRegistryView(mcp_manager=mcp_manager)

    def get_openai_tool_definitions(
        self,
        *,
        visible_tool_names: list[str],
    ) -> list[dict[str, Any]]:
        return self._registry.get_openai_tool_definitions(
            visible_tool_names=visible_tool_names
        )

    async def execute_tool_call(
        self,
        call: RuntimeToolCall,
        *,
        visible_tool_names: list[str],
    ) -> RuntimeToolResult:
        tool_info = self._registry.resolve_tool(
            tool_name=call.tool_name,
            visible_tool_names=visible_tool_names,
        )
        if tool_info is None:
            message = f"Unknown tool requested: {call.tool_name}"
            error_payload = {
                "code": "unknown_tool",
                "message": message,
                "details": {},
            }
            return RuntimeToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                content_text=self._serialize_error(
                    code="unknown_tool",
                    message=message,
                ),
                error_code="UNKNOWN_TOOL",
                structured_payload={
                    "error_payload": error_payload,
                    "content_payload": {"error": error_payload},
                },
            )

        result = await self._mcp_manager.call_tool_by_qualified_name(
            qualified_name=tool_info.qualified_name,
            arguments=call.arguments,
        )
        success = bool(result.get("success"))
        content_text = str(result.get("content") or "")
        content_payload = self._parse_content_payload(content_text)
        structured_payload = {
            "server_id": tool_info.server_id,
            "tool_name": tool_info.name,
            "qualified_name": tool_info.qualified_name,
            "raw_qualified_name": tool_info.raw_qualified_name,
        }
        if content_payload is not None:
            structured_payload["content_payload"] = content_payload
        if success:
            if content_payload is not None:
                structured_payload["result_payload"] = content_payload
        else:
            structured_payload["error_payload"] = self._error_payload(
                error_code=str(result.get("error_code"))
                if result.get("error_code")
                else None,
                content_payload=content_payload,
                content_text=content_text,
            )
        return RuntimeToolResult(
            call_id=call.call_id,
            tool_name=tool_info.qualified_name,
            success=success,
            content_text=content_text,
            error_code=str(result.get("error_code"))
            if result.get("error_code")
            else None,
            structured_payload=structured_payload,
        )

    @staticmethod
    def _serialize_error(*, code: str, message: str) -> str:
        return json.dumps(
            {"error": {"code": code, "message": message}},
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _parse_content_payload(content_text: str) -> Any | None:
        if not content_text:
            return None
        try:
            return json.loads(content_text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _error_payload(
        *,
        error_code: str | None,
        content_payload: Any,
        content_text: str,
    ) -> dict[str, Any]:
        if isinstance(content_payload, dict):
            if isinstance(content_payload.get("error"), dict):
                nested = content_payload["error"]
                return {
                    "code": nested.get("code") or error_code,
                    "message": nested.get("message") or content_text,
                    "details": nested.get("details")
                    or content_payload.get("details")
                    or {},
                }
            return {
                "code": content_payload.get("code") or error_code,
                "message": content_payload.get("message") or content_text,
                "details": content_payload.get("details") or {},
            }
        return {
            "code": error_code,
            "message": content_text,
            "details": {},
        }
