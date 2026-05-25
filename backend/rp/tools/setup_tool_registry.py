"""Registered provider-visible setup tool metadata."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from models.mcp_config import McpToolInfo
from rp.agent_runtime.prompts.setup_agent import tool_description_text
from rp.setup_agent_memory.contracts import (
    SetupSessionMemoryOpenInput,
    SetupSessionMemoryReadInput,
    SetupSessionMemorySearchInput,
)

from .setup_tool_contracts import (
    SetupRegisterAssetInput,
    SetupStageEntryDeleteInput,
    SetupStageEntryEditInput,
    SetupStageEntryListInput,
    SetupStageEntryReadInput,
    SetupStageEntryWriteInput,
)


@dataclass(frozen=True)
class SetupToolRegistration:
    name: str
    description: str
    input_model: type[BaseModel]
    handler_attr: str
    dispatch_method: str
    capability_group: str


SETUP_TOOL_REGISTRY: tuple[SetupToolRegistration, ...] = (
    SetupToolRegistration(
        name="setup.asset.register",
        description=tool_description_text("setup.asset.register"),
        input_model=SetupRegisterAssetInput,
        handler_attr="_asset_register_tool",
        dispatch_method="_dispatch_asset_register",
        capability_group="asset",
    ),
    SetupToolRegistration(
        name="setup.memory.search",
        description=tool_description_text("setup.memory.search"),
        input_model=SetupSessionMemorySearchInput,
        handler_attr="_memory_search_tool",
        dispatch_method="_dispatch_memory_search",
        capability_group="setup_memory",
    ),
    SetupToolRegistration(
        name="setup.memory.open",
        description=tool_description_text("setup.memory.open"),
        input_model=SetupSessionMemoryOpenInput,
        handler_attr="_memory_open_tool",
        dispatch_method="_dispatch_memory_open",
        capability_group="setup_memory",
    ),
    SetupToolRegistration(
        name="setup.memory.read_refs",
        description=tool_description_text("setup.memory.read_refs"),
        input_model=SetupSessionMemoryReadInput,
        handler_attr="_memory_read_refs_tool",
        dispatch_method="_dispatch_memory_read_refs",
        capability_group="setup_memory",
    ),
    SetupToolRegistration(
        name="setup.stage_entry.list",
        description=tool_description_text("setup.stage_entry.list"),
        input_model=SetupStageEntryListInput,
        handler_attr="_stage_entry_list_tool",
        dispatch_method="_dispatch_stage_entry_list",
        capability_group="stage_entry",
    ),
    SetupToolRegistration(
        name="setup.stage_entry.read",
        description=tool_description_text("setup.stage_entry.read"),
        input_model=SetupStageEntryReadInput,
        handler_attr="_stage_entry_read_tool",
        dispatch_method="_dispatch_stage_entry_read",
        capability_group="stage_entry",
    ),
    SetupToolRegistration(
        name="setup.stage_entry.write",
        description=tool_description_text("setup.stage_entry.write"),
        input_model=SetupStageEntryWriteInput,
        handler_attr="_stage_entry_write_tool",
        dispatch_method="_dispatch_stage_entry_write",
        capability_group="stage_entry",
    ),
    SetupToolRegistration(
        name="setup.stage_entry.edit",
        description=tool_description_text("setup.stage_entry.edit"),
        input_model=SetupStageEntryEditInput,
        handler_attr="_stage_entry_edit_tool",
        dispatch_method="_dispatch_stage_entry_edit",
        capability_group="stage_entry",
    ),
    SetupToolRegistration(
        name="setup.stage_entry.delete",
        description=tool_description_text("setup.stage_entry.delete"),
        input_model=SetupStageEntryDeleteInput,
        handler_attr="_stage_entry_delete_tool",
        dispatch_method="_dispatch_stage_entry_delete",
        capability_group="stage_entry",
    ),
)


def build_setup_tool_schema_map() -> dict[str, type[BaseModel]]:
    return {entry.name: entry.input_model for entry in SETUP_TOOL_REGISTRY}


def build_setup_tool_infos(
    *,
    provider_id: str,
    server_name: str,
) -> list[McpToolInfo]:
    return [
        McpToolInfo(
            server_id=provider_id,
            server_name=server_name,
            name=entry.name,
            description=entry.description,
            input_schema=entry.input_model.model_json_schema(),
        )
        for entry in SETUP_TOOL_REGISTRY
    ]
