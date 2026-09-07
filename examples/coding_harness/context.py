"""Application-owned Jinja context; native protocol history stays structured."""

import json
from dataclasses import asdict

from slick import Prompt
from slick.turns import ModelTurn, ToolResult, UserMessage

SKILLS = {"python": "coding_harness/skills/python.j2"}


def render_instructions(workspace, config) -> str:
    unknown = set(config.skills) - SKILLS.keys()
    if unknown:
        raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
    return Prompt("coding_harness/system.j2")(
        root=str(workspace.root),
        checks=[check.model_dump() for check in config.checks],
        limits=config.limits.model_dump(),
        skills=[SKILLS[name] for name in config.skills],
    )


def context_size(history: list, *, instructions: str, tools: list) -> int:
    payload = {
        "history": [asdict(item) for item in history],
        "instructions": instructions,
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def history_views(history: list) -> list[dict]:
    """Select observable content only; never render provider reasoning payloads."""
    views = []
    for item in history:
        match item:
            case UserMessage(text=text):
                views.append({"role": "user", "text": text})
            case ModelTurn(text=text, tool_calls=calls):
                views.append(
                    {"role": "assistant", "text": text, "calls": [asdict(call) for call in calls]}
                )
            case ToolResult(call_id=call_id, content=content, is_error=error):
                views.append({"role": "tool", "id": call_id, "text": content, "error": error})
            case _:
                raise ValueError("Unknown history record")
    return views


def summary_prompt(state, config) -> str:
    from .state import ContextSummary

    views = history_views(state.history)
    omitted = 0
    while True:
        text = Prompt("coding_harness/compact.j2")(
            task=state.task,
            checks=[check.model_dump() for check in config.checks],
            ledger=state.edit_ledger,
            history=views,
            omitted=omitted,
            schema=ContextSummary.model_json_schema(),
        )
        if (
            context_size([UserMessage(text)], instructions="", tools=[])
            <= config.limits.context_hard_chars
        ):
            return text
        if not views:
            raise ValueError("Pinned summary context exceeds the input-size limit")
        # Summary views are not replayed protocol; remove an oldest whole exchange.
        views.pop(0)
        omitted += 1
        while views and views[0]["role"] != "user":
            views.pop(0)
            omitted += 1
