"""Automatic interaction bookkeeping around explicit provider and tool calls."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field, TypeAdapter
from typing_extensions import TypedDict

from .tools import Tool, ToolError, ToolRequest, ToolResult, prepare_tools

SNAPSHOT_CONFIG = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)


class _WorkFields(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    request: ToolRequest
    result: ToolResult | None
    submitted: bool


def _valid_work(work):
    result = work["result"]
    if result is None:
        if work["submitted"]:
            raise ValueError("Submitted work requires a result")
    elif result["request"] != work["request"]:
        raise ValueError("Tool result does not match its request")
    return work


_Work = Annotated[_WorkFields, AfterValidator(_valid_work)]


class _ExchangeFields(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    context: str | None
    text: str
    work: list[_Work]


def _unique_work(exchange):
    ids = [item["request"]["id"] for item in exchange["work"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tool request IDs in snapshot")
    return exchange


_Exchange = Annotated[_ExchangeFields, AfterValidator(_unique_work)]


class _SnapshotFields(TypedDict):
    __pydantic_config__ = SNAPSHOT_CONFIG
    version: Annotated[int, Field(ge=1, le=1)]
    history: list[_Exchange]


def _valid_history(snapshot):
    for exchange in snapshot["history"][:-1]:
        if any(not item["submitted"] for item in exchange["work"]):
            raise ValueError("Only the current exchange can contain unsubmitted work")
    return snapshot


SessionSnapshot = Annotated[_SnapshotFields, AfterValidator(_valid_history)]


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


class Session:
    """Own interaction history and tool work; the application supplies context.

    Operations are asynchronous and sequential per instance. Supplied providers
    and Python functions remain application-owned resources.
    """

    def __init__(self, *, provider=None, tools=None):
        self._provider = provider
        self._tools = prepare_tools([] if tools is None else tools)
        self._history: list[dict] = []
        self._busy = False

    @contextmanager
    def _operation(self):
        if self._busy:
            raise RuntimeError("A Session operation is already active")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

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
            selected = self._provider if provider is None else provider
            if selected is None:
                raise ValueError("Supply a provider to Session or acall")
            if self.pending_requests:
                raise ValueError("Resolve or cancel pending requests before calling a provider")
            results = self.ready_results
            text, requests = await selected.acall(context, tools=self.tools, tool_results=results)
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
                        {"request": request, "result": None, "submitted": False}
                        for request in requests
                    ],
                }
            )
            return text, deepcopy(requests)

    def _find_work(self, request):
        for work in self._work:
            if work["request"] == request:
                return work
        raise ValueError("Request does not match work in the current exchange")

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
            return await self._resolve(self._find_work(request))

    async def resolve_pending(self) -> list[ToolResult]:
        """Execute unstarted requests sequentially; ordinary tool errors are results.

        Cancellation records the interrupted request and propagates. Remaining
        requests stay pending for the application to resolve or cancel.
        """
        with self._operation():
            pending = [work for work in self._work if work["result"] is None]
            return [await self._resolve(work) for work in pending]

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
            return {"version": 1, "history": self.history}

    @classmethod
    def from_dict(cls, data: dict, *, provider=None, tools=None) -> "Session":
        """Restore validated data and bind explicitly supplied live resources.

        Loading performs no provider calls or tool execution. Pending work stays
        pending and completed work retains its recorded results.
        """
        history = _snapshot.dump_python(_snapshot.validate_python(data), exclude_unset=True)[
            "history"
        ]
        session = cls(provider=provider, tools=tools)
        session._history = history
        return session
