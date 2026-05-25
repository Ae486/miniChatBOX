"""Setup context compaction prompt prose."""

from __future__ import annotations

COMPACT_SYSTEM_PROMPT = (
    "You are running SetupStageCompactPrompt. Produce a compact carry-forward "
    "summary for older current-step setup discussion. Do not call tools. Do not "
    "write drafts. Do not decide readiness or commit. Preserve only facts, decisions, "
    "open threads, draft refs, and unresolved blockers needed for the next SetupAgent "
    "turn in this same stage. When incremental_update is true, update "
    "previous_compact_summary using only newly_compacted_current_step_messages; the "
    "recent raw window is still prompt-visible elsewhere and must not be duplicated. "
    "Output JSON only."
)
