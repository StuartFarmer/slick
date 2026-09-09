"""Application-owned Jinja context and explicit input-size bounds."""

import json

from slick import Prompt

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


def context_size(context: str, *, tools: list, tool_results=None) -> int:
    payload = {
        "context": context,
        "tool_results": tool_results or [],
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def history_views(session, state, *, include_ready=False) -> list[dict]:
    """Project responses and app notes; never replay an already rendered prompt."""
    history = session.history
    notes = {}
    for note in state.context_notes:
        notes.setdefault(note["after"], []).append(
            {key: value for key, value in note.items() if key != "after"}
        )
    views = []
    for index in range(state.context_start, len(history) + 1):
        views.extend(notes.get(index, []))
        if index == len(history):
            break
        exchange = history[index]
        work = [item for item in exchange["work"] if include_ready or item["submitted"]]
        views.append(
            {
                "role": "assistant",
                "text": exchange["text"],
                "calls": [item["request"] for item in work],
            }
        )
        views.extend(
            {
                "role": "tool",
                "id": item["request"]["id"],
                "text": item["result"]["content"],
                "error": item["result"]["is_error"],
            }
            for item in work
            if item["result"] is not None
        )
    return views


def render_context(workspace, config, state, session) -> str:
    return Prompt("coding_harness/context.j2")(
        instructions=render_instructions(workspace, config),
        history=history_views(session, state),
    )


def summary_prompt(state, config, session) -> str:
    from .state import ContextSummary

    views = history_views(session, state)
    omitted = 0
    while True:
        text = Prompt("coding_harness/compact.j2")(
            task=state.task,
            checks=[check.model_dump() for check in config.checks],
            ledger=state.edit_ledger,
            history=views,
            tool_work={"requests": session.pending_requests, "results": session.ready_results},
            omitted=omitted,
            schema=ContextSummary.model_json_schema(),
        )
        if context_size(text, tools=[]) <= config.limits.context_hard_chars:
            return text
        if not views:
            raise ValueError("Pinned summary context exceeds the input-size limit")
        # Summary views are not replayed protocol; remove an oldest whole exchange.
        views.pop(0)
        omitted += 1
        while views and views[0]["role"] != "user":
            views.pop(0)
            omitted += 1
