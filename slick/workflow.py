"""Opt-in durable replay of ordinary Python async functions."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import sqlite3
import textwrap
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import FunctionType
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from .inbox import Inbox


class WorkflowError(ValueError):
    """A workflow cannot be compiled, owned, or safely replayed."""


@dataclass
class _Frame:
    run: Workflow
    path: tuple[str, ...]
    task: asyncio.Task
    counts: dict[str, int] = field(default_factory=dict)

    def next(self, name):
        if asyncio.current_task() is not self.task:
            raise WorkflowError("A workflow cannot execute concurrent instrumented calls")
        occurrence = self.counts.get(name, 0)
        self.counts[name] = occurrence + 1
        return (*self.path, f"{name}:{occurrence}")


_current: ContextVar[_Frame | None] = ContextVar("slick_workflow", default=None)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _name(function):
    module = function.__module__
    if module == "__main__":
        spec = inspect.unwrap(function).__globals__.get("__spec__")
        if spec is not None:
            module = spec.name
    return f"{module}.{function.__qualname__}"


class Workflow:
    """Journal a run in SQLite; use the same run_id and inputs to recover it.

    inputs holds stable business inputs, not providers or other runtime resources.
    Use a new run_id/version for changes to dependencies such as prompts or helpers.
    """

    def __init__(self, path, *, run_id: str, inputs=None, version: str = "1"):
        if not isinstance(run_id, str) or not run_id.strip():
            raise WorkflowError("run_id must be a nonblank string")
        if not isinstance(version, str) or not version.strip():
            raise WorkflowError("version must be a nonblank string")
        self.run_id = run_id
        self.version = version
        self.inputs = _json(inputs)
        self.inbox = Inbox(path)
        self._lock = None
        self._token = None
        with self.inbox._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY, version TEXT NOT NULL, inputs TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflow_functions (
                    run_id TEXT NOT NULL, name TEXT NOT NULL, code TEXT NOT NULL,
                    PRIMARY KEY (run_id, name)
                );
                CREATE TABLE IF NOT EXISTS workflow_results (
                    run_id TEXT NOT NULL, position TEXT NOT NULL,
                    target TEXT NOT NULL, schema TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (run_id, position)
                );
            """)

    async def __aenter__(self):
        if _current.get() is not None or self._lock is not None:
            raise WorkflowError("A Workflow context is already active")
        # A separate SQLite lock file leaves the inbox writable while a run waits.
        # SQLite releases the lock on process death; never unlink a live lock file.
        directory = self.inbox.path.with_name(self.inbox.path.name + ".locks")
        directory.mkdir(exist_ok=True)
        lock = sqlite3.connect(directory / _digest(self.run_id), timeout=0)
        try:
            lock.execute("BEGIN EXCLUSIVE")
            with self.inbox._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO workflow_runs (id, version, inputs) VALUES (?, ?, ?)",
                    (self.run_id, self.version, self.inputs),
                )
                existing = connection.execute(
                    "SELECT version, inputs FROM workflow_runs WHERE id = ?", (self.run_id,)
                ).fetchone()
                if existing != (self.version, self.inputs):
                    raise WorkflowError("Run already exists with different inputs or version")
        except sqlite3.OperationalError as error:
            lock.close()
            if str(error) == "database is locked":
                raise WorkflowError("Run is already active in another worker") from error
            raise
        except BaseException:
            lock.close()
            raise
        self._lock = lock
        self._token = _current.set(_Frame(self, (), asyncio.current_task()))
        return self

    async def __aexit__(self, *exc):
        _current.reset(self._token)
        self._lock.close()
        self._lock = self._token = None

    def _check_function(self, name, digest):
        with self.inbox._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO workflow_functions (run_id, name, code) VALUES (?, ?, ?)",
                (self.run_id, name, digest),
            )
            recorded = connection.execute(
                "SELECT code FROM workflow_functions WHERE run_id = ? AND name = ?",
                (self.run_id, name),
            ).fetchone()[0]
            if recorded != digest:
                raise WorkflowError(f"Workflow code changed for {name}; use a new run_id")


def _plain_json(value):
    if value is None or type(value) in (str, int, float, bool):
        return True
    if type(value) is list:
        return all(_plain_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json(item) for key, item in value.items())
    return False


def _prepare(site, function):
    if not inspect.iscoroutinefunction(function):
        raise WorkflowError("Awaited targets must be async functions; wrap other awaitables in one")

    async def call(*args, **kwargs):
        return await _invoke(site, function, args, kwargs)

    return call


async def _invoke(site, function, args, kwargs):
    frame = _current.get()
    position = frame.next(f"await:{site}")
    is_request = getattr(function, "__func__", None) is Inbox.request
    if is_request:
        key = "workflow:" + _json([str(frame.run.inbox.path), frame.run.run_id, position])
        kwargs.setdefault("key", key)
        result_type = kwargs.get("output_type") or Any
    else:
        result_type = get_type_hints(inspect.unwrap(function)).get("return", Any)
    adapter = TypeAdapter(result_type)
    schema = _json(adapter.json_schema())
    name = _name(function)
    target = name + ":" + getattr(function, "_workflow_digest", "")
    run = frame.run
    location = _json(position)
    with run.inbox._connect() as connection:
        row = connection.execute(
            "SELECT target, schema, result FROM workflow_results WHERE run_id = ? AND position = ?",
            (run.run_id, location),
        ).fetchone()
    if row is not None:
        if row[:2] != (target, schema):
            raise WorkflowError("Recorded call has a different function or output type")
        return adapter.validate_json(row[2])
    token = _current.set(_Frame(run, position, frame.task))
    try:
        result = await function(*args, **kwargs)
    finally:
        _current.reset(token)
    if result_type is Any and not _plain_json(result):
        raise WorkflowError(f"{name} needs a return annotation to restore its result type")
    if result_type is Any:
        _json(result)  # Reject nonfinite numbers rather than recording them as null.
    result = adapter.validate_python(result, strict=True)
    payload = adapter.dump_json(result).decode()
    restored = adapter.validate_json(payload)  # Never commit a result that cannot be restored.
    # No await between completion and commit: coroutine cancellation cannot split this boundary.
    with run.inbox._connect() as connection:
        connection.execute(
            "INSERT INTO workflow_results (run_id, position, target, schema, result) "
            "VALUES (?, ?, ?, ?, ?)",
            (run.run_id, location, target, schema, payload),
        )
    return restored


class _Instrument(ast.NodeTransformer):
    def __init__(self):
        self.site = 0

    def visit_Await(self, node):
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name | ast.Attribute):
            raise WorkflowError("Only direct awaited function calls are supported")
        if any(isinstance(child, ast.Await) for child in ast.walk(call)):
            raise WorkflowError("Nested await expressions are unsupported")
        site = self.site
        self.site += 1
        # Prepare the target before evaluating arguments, in their original Python scope.
        return ast.copy_location(
            ast.Await(
                value=ast.Call(
                    func=ast.Call(
                        func=ast.Name(id="__slick_await", ctx=ast.Load()),
                        args=[ast.Constant(site), call.func],
                        keywords=[],
                    ),
                    args=call.args,
                    keywords=call.keywords,
                )
            ),
            node,
        )


def _compile(function):
    if any(
        name.startswith("__slick_")
        for name in (
            function.__name__,
            *inspect.signature(function).parameters,
            *function.__code__.co_freevars,
        )
    ):
        raise WorkflowError("Names starting with __slick_ are reserved")
    try:
        source, first_line = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise WorkflowError("@workflow requires an async function with available source") from error
    node = ast.parse(textwrap.dedent("".join(source))).body[0]
    allowed = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Expr,
        ast.If,
        ast.For,
        ast.While,
        ast.Return,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Raise,
    )
    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.stmt) and not isinstance(child, allowed):
                raise WorkflowError(f"Unsupported workflow syntax: {type(child).__name__}")
            if isinstance(
                child,
                ast.Yield
                | ast.YieldFrom
                | ast.Lambda
                | ast.NamedExpr
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            ):
                raise WorkflowError(f"Unsupported workflow syntax: {type(child).__name__}")
            if isinstance(child, ast.Name) and child.id.startswith("__slick_"):
                raise WorkflowError("Names starting with __slick_ are reserved")
    node.decorator_list = []
    digest = _digest(ast.dump(node, include_attributes=False))
    node = _Instrument().visit(node)
    # Defaults and annotations already ran at definition time. Keep their original objects.
    node.args.defaults = []
    node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
    for arg in (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    ):
        if arg is not None:
            arg.annotation = None
    node.returns = None
    if getattr(node, "type_params", []):
        raise WorkflowError("Generic workflow definitions are unsupported")
    ast.increment_lineno(node, first_line - 1)
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))

    def cell(value):
        return (lambda: value).__closure__[0]

    cells["__slick_await"] = cell(_prepare)
    factory = ast.FunctionDef(
        name="__slick_factory",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[
            *(
                ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(None))
                for name in cells
            ),
            node,
            ast.Return(value=ast.Name(id=node.name, ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    if "type_params" in factory._fields:
        factory.type_params = []
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace = {}
    exec(compile(module, function.__code__.co_filename, "exec"), namespace)
    code = namespace["__slick_factory"]().__code__
    compiled = FunctionType(
        code,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        tuple(cells[name] for name in code.co_freevars),
    )
    compiled.__kwdefaults__ = function.__kwdefaults__
    return compiled, digest


def workflow(function):
    """Record supported awaited calls when a Workflow context is active.

    Plain Python between awaits reexecutes on recovery and must be deterministic.
    Without a context, call the original function with its normal Python behavior.
    """
    if not inspect.iscoroutinefunction(function):
        raise WorkflowError("@workflow requires an async function")
    if hasattr(function, "__wrapped__"):
        raise WorkflowError(
            "@workflow must directly decorate the function; put other decorators above it"
        )
    compiled, digest = _compile(function)
    name = _name(function)

    @wraps(function)
    async def call(*args, **kwargs):
        frame = _current.get()
        if frame is None:
            return await function(*args, **kwargs)
        position = frame.next(name)
        frame.run._check_function(name, digest)
        token = _current.set(_Frame(frame.run, position, frame.task))
        try:
            return await compiled(*args, **kwargs)
        finally:
            _current.reset(token)

    call._workflow_digest = digest
    return call
