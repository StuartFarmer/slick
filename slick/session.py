"""Automatic interaction bookkeeping around explicit provider and tool calls."""

import asyncio
import json
from contextlib import contextmanager
from copy import deepcopy
from threading import get_ident
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ConfigDict, TypeAdapter
from typing_extensions import NotRequired, TypedDict

from .prompts import Prompt, _render_prompt, parse
from .tools import Tool, ToolError, ToolRequest, ToolResult, prepare_tools

SNAPSHOT_CONFIG = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)


class _Work(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    request: ToolRequest
    result: ToolResult | None
    submitted: bool


class _Exchange(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    context: str | None
    text: str
    work: list[_Work]


class _Run(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    context: str
    status: Literal["start", "tools", "continue", "complete", "incomplete"]


class _SnapshotFields(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    history: list[_Exchange]
    continuation: NotRequired[dict[str, Any]]
    run: NotRequired[_Run]


def _valid_snapshot(snapshot):
    history = snapshot["history"]
    for index, exchange in enumerate(history):
        ids = [work["request"]["id"] for work in exchange["work"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"history[{index}]: Duplicate tool request IDs in snapshot")
        for work_index, work in enumerate(exchange["work"]):
            location = f"history[{index}].work[{work_index}]"
            result = work["result"]
            if result is None and work["submitted"]:
                raise ValueError(f"{location}: Submitted work requires a result")
            if result is not None and result["request"] != work["request"]:
                raise ValueError(f"{location}: Tool result does not match its request")
            if index < len(history) - 1 and not work["submitted"]:
                raise ValueError(f"{location}: Historical work must be submitted")
    return snapshot


SessionSnapshot = Annotated[_SnapshotFields, AfterValidator(_valid_snapshot)]


_snapshot = TypeAdapter(SessionSnapshot)


def _result(request, content, *, error=False):
    return {"request": deepcopy(request), "content": content, "is_error": error}


async def _invoke(request, tools):
    if request.get("argument_error"):
        return _result(request, request["argument_error"], error=True)
    tool = tools.get(request["name"])
    if tool is None:
        return _result(request, f"Unknown tool: {request['name']}", error=True)
    try:
        content = await tool.ainvoke(request["arguments"])
    except ToolError as error:
        return _result(request, f"{error}; effects may have occurred; inspect state", error=True)
    return _result(request, content)


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


class Session:
    """Own interaction history and tool work; the application supplies context.

    Operations are sequential per instance; prompt() supports synchronous calls.
    Supplied providers and Python functions remain application-owned resources.
    """

    def __init__(self, *, provider=None, tools=None):
        self._provider = provider
        self._tools = prepare_tools([] if tools is None else tools)
        self._history: list[dict] = []
        self._busy = False
        self._running = None
        self._continuation = None
        self._run = None

    @contextmanager
    def _operation(self):
        if self._busy or (self._running is not None and self._running != _owner()):
            raise RuntimeError("A Session operation is already active")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    @contextmanager
    def _run_guard(self):
        if self._busy or self._running is not None:
            raise RuntimeError("A Session operation is already active")
        self._running = _owner()
        try:
            yield
        finally:
            self._running = None

    def _start(self, prompt, output_type, max_turns, variables):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if prompt is None:
            if self._run is None:
                raise ValueError("Supply a prompt to start a conversation")
            if variables:
                raise ValueError("Cannot supply template variables when resuming")
            return
        if isinstance(prompt, Prompt):
            context = _render_prompt(prompt, output_type, variables)
        elif isinstance(prompt, str) and not variables:
            context = prompt
        else:
            raise TypeError("Supply a Prompt with variables or already rendered text")
        if self._run is not None and self._run["context"] != context:
            raise ValueError(
                "A different conversation is pending; resume it before starting another"
            )
        if self._run is None:
            self._run = {"context": context, "status": "start"}

    def _finish(self, output_type):
        result = parse(self._history[-1]["text"], str if output_type is None else output_type)
        self._run = None
        return result

    def _check_status(self):
        if self._run["status"] == "incomplete":
            raise RuntimeError(
                "Provider stopped before completing the conversation; inspect history"
            )

    def run(self, prompt=None, /, *, output_type=None, max_turns=20, **variables):
        """Run a bounded synchronous tool conversation and parse its final answer.

        Omit prompt to resume a paused run. max_turns limits additional provider
        calls; reaching it leaves requested tools unexecuted for explicit continuation.
        """
        with self._run_guard():
            self._start(prompt, output_type, max_turns, variables)
            if self._run["status"] == "complete":
                return self._finish(output_type)
            self._check_status()
            for _ in range(max_turns):
                for request in self.pending_requests:
                    self._resolve_sync(request)
                self._turn()
                self._check_status()
                if self._run["status"] == "complete":
                    return self._finish(output_type)
            raise RuntimeError(f"Session reached its {max_turns}-turn limit; resume to continue")

    async def arun(self, prompt=None, /, *, output_type=None, max_turns=20, **variables):
        """Like run(), awaiting providers and tools; checkpoint each operation in a workflow."""
        with self._run_guard():
            self._start(prompt, output_type, max_turns, variables)
            if self._run["status"] == "complete":
                return self._finish(output_type)
            self._check_status()
            for _ in range(max_turns):
                for request in self.pending_requests:
                    await self._astep("tool", self.resolve, request)
                await self._astep("turn", self._aturn)
                self._check_status()
                if self._run["status"] == "complete":
                    return self._finish(output_type)
            raise RuntimeError(f"Session reached its {max_turns}-turn limit; resume to continue")

    async def _astep(self, name, function, *args):
        from .workflow import _current, _invoke

        if _current.get() is not None:
            return await _invoke(f"session:{name}", function, args, {})
        return await function(*args)

    def _turn_input(self, native):
        context = self._run["context"] if self._run["status"] == "start" else ""
        if native:
            return context
        if self._continuation is not None:
            raise ValueError("A native conversation requires a compatible provider")
        # Custom call/acall providers get a portable transcript without recursively nested context.
        transcript = [{"text": item["text"], "work": item["work"]} for item in self._history]
        return self._run["context"] + (
            "\n\nConversation so far:\n" + json.dumps(transcript) if transcript else ""
        )

    def _accept_turn(self, context, response, native):
        if native:
            result = self._record_response(context, (response["text"], response["requests"]))
            self._continuation = deepcopy(response["continuation"])
            self._run["status"] = response["status"]
        else:
            result = self._record_response(context, response)
            self._run["status"] = "tools" if result[1] else "complete"
        return result

    async def _aturn(self) -> tuple[str, list[dict]]:
        with self._operation():
            provider = self._prepare_call(None)
            native = getattr(provider, "aturn", None)
            context = self._turn_input(native)
            kwargs = {"tools": self.tools, "tool_results": self.ready_results}
            response = (
                await native(context, continuation=deepcopy(self._continuation), **kwargs)
                if native
                else await provider.acall(context, **kwargs)
            )
            return self._accept_turn(context, response, native)

    def _turn(self):
        with self._operation():
            provider = self._prepare_call(None)
            native = getattr(provider, "turn", None)
            context = self._turn_input(native)
            kwargs = {"tools": self.tools, "tool_results": self.ready_results}
            response = (
                native(context, continuation=deepcopy(self._continuation), **kwargs)
                if native
                else provider.call(context, **kwargs)
            )
            return self._accept_turn(context, response, native)

    def _resolve_sync(self, request):
        with self._operation():
            work = next(work for work in self._work if work["request"] == request)
            if request.get("argument_error"):
                result = _result(request, request["argument_error"], error=True)
            elif request["name"] not in self._tools:
                result = _result(request, f"Unknown tool: {request['name']}", error=True)
            else:
                try:
                    result = _result(
                        request, self._tools[request["name"]].invoke(request["arguments"])
                    )
                except ToolError as error:
                    result = _result(
                        request, f"{error}; effects may have occurred; inspect state", error=True
                    )
            work["result"] = result

    @property
    def _work(self):
        return self._history[-1]["work"] if self._history else []

    @property
    def history(self) -> list[dict]:
        """A detached chronological list of successful exchanges and their work."""
        return deepcopy(self._history)

    @property
    def tools(self) -> list[Tool]:
        """The registered tools, in registration order, in a fresh list."""
        return list(self._tools.values())

    @property
    def pending_requests(self) -> list[dict]:
        """Unexecuted requests in the current exchange, in model order."""
        return deepcopy([work["request"] for work in self._work if work["result"] is None])

    @property
    def ready_results(self) -> list[ToolResult]:
        """Completed results awaiting submission to a provider."""
        return deepcopy(
            [
                work["result"]
                for work in self._work
                if work["result"] is not None and not work["submitted"]
            ]
        )

    async def acall(self, context: str, *, provider=None) -> tuple[str, list[dict]]:
        """Call once, submit ready results, and record a valid response atomically.

        Context is sent exactly as supplied. Tools are executed only by resolve
        or resolve_pending. A provider override applies to this call only.
        """
        with self._operation():
            selected = self._prepare_call(provider)
            response = await selected.acall(
                context, tools=self.tools, tool_results=self.ready_results
            )
            return self._record_response(context, response)

    async def aprompt(
        self, prompt: Prompt, /, *, output_type: Any = None, provider=None, **variables
    ) -> tuple[Any, list[dict]]:
        """Render, call asynchronously, and return (parsed output or raw text, requests).

        output_type supplies the template's schema variable and the parse type.
        None leaves text and template variables unchanged. The raw exchange is
        recorded before parsing; validation failures leave tool work accessible.
        Requests are never executed or cancelled automatically.
        """
        context = _render_prompt(prompt, output_type, variables)
        text, requests = await self.acall(context, provider=provider)
        return parse(text, str if output_type is None else output_type), requests

    def prompt(
        self, prompt: Prompt, /, *, output_type: Any = None, provider=None, **variables
    ) -> tuple[Any, list[dict]]:
        """Like aprompt(), using provider.call directly without an event loop."""
        context = _render_prompt(prompt, output_type, variables)
        with self._operation():
            selected = self._prepare_call(provider)
            response = selected.call(context, tools=self.tools, tool_results=self.ready_results)
            text, requests = self._record_response(context, response)
        return parse(text, str if output_type is None else output_type), requests

    def _prepare_call(self, provider):
        selected = self._provider if provider is None else provider
        if selected is None:
            raise ValueError("Supply a provider to Session or the call")
        if any(work["result"] is None for work in self._work):
            raise ValueError("Resolve or cancel pending requests before calling a provider")
        return selected

    def _record_response(self, context, response):
        text, requests = response
        requests = deepcopy(requests)
        if requests and not self._tools:
            raise ValueError("Provider requested tools when none were offered")
        for work in self._work:
            work["submitted"] = True
        self._history.append(
            {
                "context": context,
                "text": text,
                "work": [
                    {"request": request, "result": None, "submitted": False} for request in requests
                ],
            }
        )
        return text, deepcopy(requests)

    async def _resolve(self, work):
        if work["result"] is None:
            try:
                work["result"] = await _invoke(work["request"], self._tools)
            except BaseException:
                work["result"] = _result(
                    work["request"],
                    "Interrupted; effects may have occurred. Inspect current state.",
                    error=True,
                )
                raise
        return deepcopy(work["result"])

    async def resolve(self, request: ToolRequest | dict) -> ToolResult:
        """Execute current work once, or return its already recorded result."""
        with self._operation():
            for work in self._work:
                if work["request"] == request:
                    return await self._resolve(work)
            raise ValueError("Request does not match work in the current exchange")

    async def resolve_pending(self) -> list[ToolResult]:
        """Execute unstarted requests sequentially; ordinary tool errors are results.

        Cancellation records the interrupted request and propagates. Remaining
        requests stay pending for the application to resolve or cancel.
        """
        with self._operation():
            return [await self._resolve(work) for work in self._work if work["result"] is None]

    def cancel_pending(self, reason: str) -> list[ToolResult]:
        """Record unstarted work as stopped, without executing any function."""
        with self._operation():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Cancellation reason must be a nonempty string")
            results = []
            for work in self._work:
                if work["result"] is None:
                    work["result"] = _result(work["request"], f"Not started: {reason}", error=True)
                    results.append(deepcopy(work["result"]))
            return results

    def to_dict(self) -> dict:
        """Return an idle, JSON-compatible snapshot; no clients or functions."""
        with self._operation():
            snapshot = {"history": self.history}
            if self._continuation is not None:
                snapshot["continuation"] = deepcopy(self._continuation)
            if self._run is not None:
                snapshot["run"] = deepcopy(self._run)
            return snapshot

    def _restore(self, data):
        with self._operation():
            snapshot = _snapshot.dump_python(_snapshot.validate_python(data), exclude_unset=True)
            self._history = deepcopy(snapshot["history"])
            self._continuation = deepcopy(snapshot.get("continuation"))
            self._run = deepcopy(snapshot.get("run"))

    @classmethod
    def from_dict(cls, data: dict, *, provider=None, tools=None) -> "Session":
        """Restore validated data and bind explicitly supplied live resources.

        Loading performs no provider calls or tool execution. Pending work stays
        pending and completed work retains its recorded results.
        """
        session = cls(provider=provider, tools=tools)
        session._restore(data)
        return session
