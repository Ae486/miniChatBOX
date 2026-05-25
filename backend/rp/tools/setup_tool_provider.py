"""Backend-local provider exposing setup private tools to SetupAgent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from models.mcp_config import McpToolInfo
from pydantic import BaseModel, ValidationError
from rp.models.setup_handoff import SetupContextBuilderInput
from rp.services.memory_crud_serialization_service import MemoryCrudSerializationService
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_truth_index_service import SetupTruthIndexService
from rp.setup_agent_memory.service import SetupSessionMemoryService
from rp.tools.setup_tools import (
    AssetRegisterTool,
    MemoryOpenTool,
    MemoryReadRefsTool,
    MemorySearchTool,
    StageEntryDeleteTool,
    StageEntryEditTool,
    StageEntryListTool,
    StageEntryReadTool,
    StageEntryWriteTool,
)
from rp.tools.setup_tools.base import setup_tool_error_details
from rp.tools.setup_tools.read_draft_refs import ReadDraftRefsTool
from rp.services.setup_workspace_service import SetupWorkspaceService
from rp.tools.setup_tool_contracts import SetupToolContractError
from rp.tools.setup_tool_registry import (
    SETUP_TOOL_REGISTRY,
    build_setup_tool_infos,
    build_setup_tool_schema_map,
)


class _SetupContextBuilderLike(Protocol):
    def build(self, input_model: SetupContextBuilderInput) -> Any: ...


class SetupToolProvider:
    """Expose setup private contract tools for the SetupAgent execution layer."""

    provider_id = "rp_setup"
    server_name = "RP Setup"

    def __init__(
        self,
        *,
        workspace_service: SetupWorkspaceService,
        context_builder: _SetupContextBuilderLike,
        runtime_state_service: SetupAgentRuntimeStateService,
        serialization_service: MemoryCrudSerializationService | None = None,
        truth_index_service: SetupTruthIndexService | None = None,
    ) -> None:
        self._workspace_service = workspace_service
        self._context_builder = context_builder
        self._runtime_state_service = runtime_state_service
        self._serialization_service = (
            serialization_service or MemoryCrudSerializationService()
        )
        self._truth_index_service = truth_index_service or SetupTruthIndexService()
        family_kwargs = {
            "workspace_service": self._workspace_service,
            "context_builder": self._context_builder,
            "runtime_state_service": self._runtime_state_service,
            "truth_index_service": self._truth_index_service,
        }
        self._asset_register_tool = AssetRegisterTool(**family_kwargs)
        self._read_draft_refs_tool = ReadDraftRefsTool(**family_kwargs)
        self._setup_memory_service = SetupSessionMemoryService(
            draft_ref_reader=lambda input_model: self._read_draft_refs_tool._read_draft_refs(
                input_model=input_model
            ),
            truth_index_service=self._truth_index_service,
        )
        memory_family_kwargs = {
            **family_kwargs,
            "setup_memory_service": self._setup_memory_service,
        }
        self._memory_search_tool = MemorySearchTool(**memory_family_kwargs)
        self._memory_open_tool = MemoryOpenTool(**memory_family_kwargs)
        self._memory_read_refs_tool = MemoryReadRefsTool(**memory_family_kwargs)
        self._stage_entry_list_tool = StageEntryListTool(**family_kwargs)
        self._stage_entry_read_tool = StageEntryReadTool(**family_kwargs)
        self._stage_entry_write_tool = StageEntryWriteTool(**family_kwargs)
        self._stage_entry_edit_tool = StageEntryEditTool(**family_kwargs)
        self._stage_entry_delete_tool = StageEntryDeleteTool(**family_kwargs)
        self._schemas: dict[str, type[BaseModel]] = build_setup_tool_schema_map()
        self._dispatch_handlers: dict[str, Callable[[Any], Any]] = (
            self._build_dispatch_handlers()
        )

    def list_tools(self) -> list[McpToolInfo]:
        return build_setup_tool_infos(
            provider_id=self.provider_id,
            server_name=self.server_name,
        )

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        model = self._schemas.get(tool_name)
        if model is None:
            return {
                "success": False,
                "content": self._serialization_service.serialize_error(
                    code="unknown_tool",
                    message=f"Unknown setup tool: {tool_name}",
                ),
                "error_code": "UNKNOWN_TOOL",
            }
        try:
            input_model = model.model_validate(arguments)
            result = await self._dispatch(tool_name=tool_name, input_model=input_model)
            return {
                "success": True,
                "content": self._serialization_service.serialize_result(result),
                "error_code": None,
            }
        except SetupToolContractError as exc:
            return {
                "success": False,
                "content": self._serialization_service.serialize_error(
                    code=exc.code,
                    message=str(exc),
                    retryable=exc.retryable,
                    details=exc.details,
                ),
                "error_code": exc.error_code,
            }
        except ValidationError as exc:
            return {
                "success": False,
                "content": self._serialization_service.serialize_error(
                    code="schema_validation_failed",
                    message="Setup tool arguments failed validation",
                    details=self._validation_error_details(
                        tool_name=tool_name,
                        arguments=arguments,
                        exc=exc,
                    ),
                ),
                "error_code": "SCHEMA_VALIDATION_FAILED",
            }
        except ValueError as exc:
            return {
                "success": False,
                "content": self._serialization_service.serialize_error(
                    code="setup_tool_failed",
                    message=str(exc),
                    details=setup_tool_error_details(
                        tool_name=tool_name,
                        failure_origin="domain",
                        repair_strategy="continue_discussion",
                    ),
                ),
                "error_code": "SETUP_TOOL_FAILED",
            }
        except Exception as exc:  # pragma: no cover - defensive surface
            self._workspace_service.rollback()
            return {
                "success": False,
                "content": self._serialization_service.serialize_error(
                    code="execution_error",
                    message=f"Setup tool execution failed: {exc}",
                    retryable=True,
                    details=setup_tool_error_details(
                        tool_name=tool_name,
                        failure_origin="execution",
                        repair_strategy="continue_discussion",
                        transient_retry=True,
                    ),
                ),
                "error_code": "EXECUTION_ERROR",
            }

    async def _dispatch(self, *, tool_name: str, input_model: Any) -> Any:
        handler = self._dispatch_handlers.get(tool_name)
        if handler is not None:
            return handler(input_model)
        raise ValueError(f"Unknown setup tool: {tool_name}")

    def _build_dispatch_handlers(self) -> dict[str, Callable[[Any], Any]]:
        handlers: dict[str, Callable[[Any], Any]] = {}
        for registration in SETUP_TOOL_REGISTRY:
            owner = getattr(self, registration.handler_attr)
            handlers[registration.name] = getattr(owner, registration.dispatch_method)
        return handlers

    def _validation_error_details(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        exc: ValidationError,
    ) -> dict[str, Any]:
        errors = exc.errors()
        return setup_tool_error_details(
            tool_name=tool_name,
            failure_origin="validation",
            repair_strategy="auto_repair",
            required_fields=self._required_fields_from_errors(errors),
            extra={
                "errors": errors,
                "provided_fields": sorted(arguments.keys())
                if isinstance(arguments, dict)
                else [],
            },
        )

    @staticmethod
    def _required_fields_from_errors(errors: list[dict[str, Any]]) -> list[str]:
        required_fields: list[str] = []
        for item in errors:
            error_type = str(item.get("type") or "")
            if "missing" not in error_type:
                continue
            loc = item.get("loc")
            if isinstance(loc, (list, tuple)):
                field = ".".join(
                    str(part) for part in loc if part not in {"body", "arguments"}
                )
            elif loc:
                field = str(loc)
            else:
                field = ""
            if field and field not in required_fields:
                required_fields.append(field)
        return required_fields
