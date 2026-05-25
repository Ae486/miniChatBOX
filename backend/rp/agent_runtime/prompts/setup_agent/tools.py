"""SetupAgent tool-facing prose.

Tool descriptions live here because they are model-visible prompt text attached
to tool schemas. Higher-level tool-use strategy stays separate as capability
guidance, so the main system prompt can talk about workflows without copying
full tool schema descriptions.
"""

from __future__ import annotations

CAPABILITY_GUIDANCE: dict[str, str] = {
    "no_active_tools": (
        "No setup tools are active for this turn; continue with visible text only."
    ),
    "stage_entry.write": (
        "For world_background, character_design, and plot_blueprint, use "
        "setup.stage_entry.write as the primary draft write tool. Provide only "
        "entry_type, title, summary, and text sections; the backend chooses the "
        "current stage, ids, semantic paths, and section shape."
    ),
    "stage_entry.read_list": (
        "Use setup.stage_entry.list and setup.stage_entry.read to inspect the "
        "current stage draft before editing or deleting entries."
    ),
    "stage_entry.edit_delete": (
        "Use setup.stage_entry.edit or setup.stage_entry.delete only with a "
        "current target_ref and basis_fingerprint from the current stage."
    ),
    "asset.register": (
        "Use setup.asset.register when the user provides a setup-scoped reference asset."
    ),
    "setup_session_memory.search": (
        "Use setup.memory.search to find setup fact refs from editable draft and "
        "accepted setup truth when the needed exact fact is not visible, especially "
        "before creative design that involves a named setup object whose concrete "
        "facts are not visible. Search results and navigation_summary are navigation "
        "only, not fact content. Use setup.memory.open on a chosen ref before relying "
        "on exact details. Opening a level-3 entry ref returns a level-4 section "
        "directory; opening a level-4 section ref returns clean fact content. After "
        "grounding the confirmed facts, keep new creative additions clearly framed "
        "as proposals until they are written or accepted."
    ),
    "legacy_patch": "Use {tool_name} only for its legacy step-specific draft family.",
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "setup.asset.register": (
        "Register a setup-scoped asset reference. Use when the user provides a relevant "
        "reference document or asset. Do not use for Memory OS mutation. Target object: "
        "ImportedAssetRaw. Important field: source_ref."
    ),
    "setup.memory.search": (
        "Search SetupAgent session-scoped setup fact index for small candidate refs from "
        "editable draft and accepted setup truth. Use it to locate named setup objects "
        "before creative design when their concrete facts are not visible. Returns "
        "navigation refs and navigation_summary only; this is not fact content. Use "
        "setup.memory.open on a chosen ref before relying on exact details. Read-only; "
        "not RP Memory OS and not long-term user memory."
    ),
    "setup.memory.open": (
        "Open one setup memory ref. Use it before relying on a named setup object whose "
        "concrete facts are not visible. Opening a level-3 entry ref returns its level-4 "
        "section directory, not content. Opening a level-4 section ref returns clean "
        "structured fact content. Read-only; setup fact sources are editable draft and "
        "accepted setup truth."
    ),
    "setup.memory.read_refs": (
        "Compatibility/internal readback for bounded payloads from current DB-backed setup "
        "sources. Agent-facing guidance should prefer setup.memory.search plus "
        "setup.memory.open."
    ),
    "setup.stage_entry.list": (
        "List editable entries from the current canonical setup stage draft block. The "
        "backend resolves the current stage from the workspace; the model must not pass "
        "stage_id. Use for world_background, character_design, and plot_blueprint draft "
        "review before read/edit/delete."
    ),
    "setup.stage_entry.read": (
        "Read one editable entry from the current canonical setup stage draft block by "
        "stage:<stage_id>:<entry_id> ref. The backend verifies the ref stage matches the "
        "workspace current stage; the model must not pass stage_id."
    ),
    "setup.stage_entry.write": (
        "Create one editable entry in the current canonical setup stage draft block. The "
        "model provides only content fields such as entry_type, title, summary, and text "
        "sections; the backend owns current_stage, entry_id, section_id, semantic_path, "
        "and internal section shape."
    ),
    "setup.stage_entry.edit": (
        "Edit one editable entry in the current canonical setup stage draft block using a "
        "current basis_fingerprint. The backend verifies the target ref stage matches the "
        "workspace current stage; the model must not pass stage_id."
    ),
    "setup.stage_entry.delete": (
        "Delete one editable entry from the current canonical setup stage draft block using "
        "a current basis_fingerprint. The backend verifies the target ref stage matches the "
        "workspace current stage; the model must not pass stage_id."
    ),
}


def capability_guidance_text(fragment_id: str, **values: object) -> str:
    return CAPABILITY_GUIDANCE[fragment_id].format(**values)


def tool_description_text(tool_name: str) -> str:
    return TOOL_DESCRIPTIONS[tool_name]
