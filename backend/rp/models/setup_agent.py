"""SetupAgent execution request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId


class SetupAgentDialogueMessage(BaseModel):
    """Minimal persisted-in-client dialogue message for setup turns."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class SetupAgentTurnRequest(BaseModel):
    """One setup agent turn request."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    model_id: str
    provider_id: str | None = None
    target_step: SetupStepId | None = None
    target_stage: SetupStageId | None = None
    history: list[SetupAgentDialogueMessage] = Field(default_factory=list)
    user_edit_delta_ids: list[str] = Field(default_factory=list)
    external_mcp_tool_allowlist: list[str] = Field(default_factory=list)
    user_prompt: str


class SetupAgentTurnResponse(BaseModel):
    """Non-stream turn response carrying only user-visible assistant text."""

    model_config = ConfigDict(extra="forbid")

    assistant_text: str
