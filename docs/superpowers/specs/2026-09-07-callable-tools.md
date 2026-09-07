# Plain Python tools: contract and acceptance specification

Status: implemented local tool contract, approved in the September 7 discussion.
The original 129-case acceptance suite has been promoted to
[tests/test_tools.py](../../../tests/test_tools.py), with additional regression
cases. Native backend tool transport and the model/tool loop remain separate.

## 1. Outcome and boundaries

Applications supply ordinary functions or bound methods. Slick derives metadata
and JSON Schema, validates model-selected arguments, invokes the original
callable, validates any declared return contract, and serializes the result.
There is no required decorator, registration service, agent base class, or tool
class on application objects.

```python
class Documents:
    def __init__(self, store):
        self.store = store

    def read(self, key: str) -> str:
        """Retrieve a document by its key."""
        return self.store.get(key)

documents = Documents(store)
tools = [documents.read]
```

The eventual execution interface can accept `tools=tools`. Designing whether a
backend performs one model turn or a complete tool loop is a separate milestone.
This spec does not introduce `backend.acall(..., tools=...)`, an Agent, a runner,
an OpenRouter adapter, or provider-specific tool transport. It defines the
callable contract those features will consume. Existing text-only backends and
`Prompt` remain unchanged.

The older [native-tools plan](../plans/2026-09-06-native-tools.md) is historical;
its Agent/Request/Response objects, backend package layout, and automatic thread
offloading are not requirements for this work.

Global constraints:

- Python >=3.10 and the repository's existing Pydantic >=2.0 declaration.
- No additional mandatory dependency; use `inspect`, `typing`, `json`, and Pydantic.
- No tool function runs during definition preparation.
- All automated acceptance tests run offline, without credentials or provider SDKs.
- All Jinja templates remain in application/example folders, never `slick/`.
- No implicit history, retries, disk writes, logging, caching, or thread scheduling.

## 2. Concrete interface for implementation and tests

Add `slick/tools.py`. `Tool` is an optional explicit wrapper, also used internally
when an application supplies a plain callable. No `@tool` decorator is required.

```python
Tool(function, *, name: str | None = None, description: str | None = None)

tool.name: str
tool.description: str
tool.parameters: dict                 # fresh JSON Schema dictionary on each access
tool.invoke(arguments: dict) -> str
await tool.ainvoke(arguments: dict) -> str

prepare_tools(functions: list) -> dict[str, Tool]
```

`prepare_tools` is a module-level integration helper, not a persistent registry.
It accepts a list containing functions, bound methods, and optional `Tool`
instances. It preserves list order, wraps plain callables, preserves already
prepared Tool instances, and rejects duplicate exposed names. An empty list
produces an empty dictionary. Preparation completes for the entire list before
the caller may make any request or execute any tool. On failure no partial
mapping is returned. No global registration or implicit method discovery occurs.

Export `Tool` and `ToolError` from `slick` for optional customization and error
handling. `prepare_tools` remains available from `slick.tools` only.

Normal application calls still use the original function or bound method. Slick
does not replace or modify that callable. `Tool.invoke` is explicitly the
validated, serialized model-call path, not a substitute for direct Python calls.

## 3. Preparation and metadata

**D01** Accept Python functions, coroutine functions, bound instance methods,
bound class methods, and functions accessed through `staticmethod`. Respect an
inspectable signature preserved by `functools.wraps`, but invoke the supplied
wrapper, not its `__wrapped__` target.

**D02** Infer the exposed name from `__name__`, unless explicitly overridden.
For this portable first version, require `[A-Za-z_][A-Za-z0-9_]{0,63}`. Do not
silently rename or truncate. An override can give an otherwise anonymous
function a valid name. Backend-specific names may be narrower in a later adapter.

**D03** Use `inspect.getdoc` for the description, or clean an explicit description
with `inspect.cleandoc`, then strip surrounding whitespace. Require nonempty text.
Preserve the full docstring body;
Google/NumPy/Sphinx parameter-docstring parsing is not part of this milestone.
`Annotated[T, Field(description=...)]` supplies parameter descriptions today.

**D04** Resolve type hints with extras. Missing or unresolvable input annotations
fail with the tool and parameter identified. Do not guess a type from a default
or from the first real call. Definitions referenced by string annotations must
be resolvable from their defining module/class; there is no caller-frame search.

**D05** Accept positional-or-keyword and keyword-only parameters, including
zero-argument functions. Initially reject positional-only parameters, `*args`,
`**kwargs`, unbound instance methods, generator/async-generator functions,
callable instances, and `functools.partial`. Their support requires a deliberate
extension of this contract. Bound `self`/`cls` never appears in the schema.

**D06** Preserve required/default/nullable semantics. No default means required.
`T | None` permits null but is still required without a default. Validate each
top-level parameter default against its declared type during preparation.
When an argument is absent, omit that keyword from the actual invocation so
Python applies its original default, including object identity and normal
mutable-default behavior. Nested Pydantic defaults follow the model's own config.
Python signature defaults take precedence over default/default_factory metadata
in a parameter's `Annotated` Field; that metadata cannot make a required
parameter optional.

**D07** Generate a root object schema with named properties, required fields,
and `additionalProperties: false`. A zero-argument schema has empty properties
and accepts only `{}`. Nested schemas may use `$defs`/`$ref`; neither exact schema
titles nor definition order is part of the contract. Backend adaptation must
not mutate this local schema. Mutating one `parameters` result must not affect
future schemas or validation.

## 4. Initial type boundary

This is an explicit initial Slick policy, not a list of everything Pydantic can
do. The same annotation allowlist applies recursively to inputs and annotated
returns. Unsupported annotations fail during preparation, even if Pydantic can
generate a schema for them.

| Annotation | Initial support |
| --- | --- |
| `str`, `int`, `float`, `bool`, `None` / `NoneType` | Yes; floats must be finite |
| `list[T]` | Yes, recursively |
| `dict[str, T]` | Yes locally; arbitrary-key inputs need backend compatibility checks |
| `Literal[...]` | Nonempty string/integer/boolean/null literal values |
| `Enum` | Homogeneous string-valued or integer-valued enums |
| `Union` / `T | U`, including nullable types | Yes, when every branch is supported |
| `Annotated[T, Field(...)]` | Supported base type plus Pydantic constraints/description |
| Pydantic `BaseModel` | Yes, supported declared fields, including nested models |
| `TypedDict` | Yes, supported declared fields; use `typing_extensions.TypedDict` on Python <3.12 |
| `Any`, `object`, bare `list`/`dict`, unresolved type variables | Reject: insufficient contract |
| Non-string dictionary keys | Reject: JSON keys cannot preserve their Python types |
| `tuple`, `set`, `frozenset`, `deque` | Deferred: conversion/ordering and schema differences |
| `bytes`, `bytearray`, `memoryview` | Deferred: no implicit binary encoding |
| `datetime`, `date`, `time`, `timedelta`, `UUID`, `Path`, `Decimal`, `complex` | Deferred: choose explicit representations first |
| Arbitrary classes, `Callable`, `type`, `range`, generators/iterators | Reject |
| Dataclasses, recursive/self-referential models, `RootModel` | Deferred in this first contract |

For a deferred type, use an explicitly typed text/list/model representation in
an application wrapper. Do not change the original application's data types.
For example, pass a document ID and let a bound method access `self.store`;
do not ask the model to supply a database connection.

BaseModel validators, serializers, aliases, and nested `extra` behavior follow
that model's declared Pydantic behavior. The top-level argument object always
forbids extra names. Models using arbitrary runtime-object fields remain
unsupported. Function-parameter `Field` aliases are rejected initially so the
exposed argument name remains the actual Python parameter name; model field
aliases are supported. Custom validators may impose rules absent from JSON
Schema, so local validation is always required.

## 5. Invocation and validation

**I01** `invoke`/`ainvoke` accept a dictionary representing the JSON argument
object. They reject strings, arrays, null, non-string keys, non-JSON Python
objects, cycles, and nonfinite numbers before calling the function. No fenced
JSON extraction, prose repair, `eval`, or Python-source execution is involved.
Parsing raw provider argument strings belongs to the future transport adapter.

**I02** Validate with Pydantic strict JSON semantics: reject numeric strings for
numbers, boolean values for numbers, floating-point values for integers, and
strings/integers for booleans. An integer can populate a float. Enum JSON values
become Enum instances. Nested object arguments become BaseModel instances;
TypedDict arguments remain dictionaries. Constrained `Field` values are checked.
Do not `model_dump()` the entire validated argument model before dispatch, since
that would turn the nested model instances back into dictionaries.

**I03** Missing required parameters, unknown root argument names, and invalid
values fail before the callable runs. Input dictionaries/lists supplied by the
caller must not be mutated by validation or by the tool; validate from a JSON
copy. This isolation does not apply to omitted Python defaults or `self` state.

**I04** The same bound method must use the same instance, and two independently
prepared instances must not share state accidentally. Only explicitly supplied
methods are eligible. A call may change its own instance state normally.

**I05** `invoke` accepts synchronous functions. An async-only tool fails with an
execution-phase error before its coroutine is created. `ainvoke` awaits async
functions and executes synchronous functions inline. There is no hidden thread,
event loop, or `asyncio.run`; blocking sync functions block their caller.
Applications can supply an explicit async wrapper using `asyncio.to_thread`.

**I06** A synchronous function unexpectedly returning an awaitable fails in the
result phase; it is not automatically awaited. Close a newly returned coroutine
to avoid an unawaited-coroutine leak; do not cancel an unrelated Future/Task.

**I07** Preserve `asyncio.CancelledError`, `KeyboardInterrupt`, and `SystemExit`.
Ordinary function exceptions become execution errors with their original cause.
Never retry a tool or roll back its side effects. Return-validation failure is
after execution and must not be mistaken for proof that no action occurred.

## 6. Returns and serialization

**R01** When a return annotation exists, validate the actual returned Python
value strictly against it before serialization. `-> None` means the function
must return None. A declared BaseModel accepts model instances or mappings that
Pydantic can validate into that model. Other conversions follow Pydantic strict
Python semantics; returning an enum value string is not returning an Enum instance.

**R02** Missing return annotations are allowed, because the model does not choose
the return value. Runtime results must then already be JSON values: strings,
booleans, integers, finite floats, null, lists, or string-keyed dictionaries of
those values. Untyped BaseModel instances, tuples, sets, bytes, generators, and
arbitrary objects fail; annotate a supported structured return to enable its
Pydantic serialization. Explicit `-> Any` is unsupported under the annotation
allowlist; it does not serve as an alternate spelling for missing metadata.

**R03** String results are returned unchanged, including empty strings and
whitespace. All other results become JSON text: numbers, booleans, null, arrays,
or objects. Structured results use Pydantic JSON-mode serialization with aliases;
do not fall back to `str(value)` or silently omit invalid fields. Check finite
numeric values before serialization can turn NaN/infinity into null, and verify
the serialized output is valid JSON. A custom serializer failure is a result error.
Output JSON whitespace/key order is not a contract.

**R04** The result annotation is a local tool-output contract. This milestone
does not transmit a return schema, enforce a final model-answer type, or implement
images/files/provider-native result blocks. The future backend wraps the returned
text with the correct call ID and role. Native return-schema support, if added,
must be an explicit backend capability rather than an assumption.

## 7. Errors and backend compatibility

```python
class ToolError(ValueError):
    name: str
    phase: str  # "definition", "arguments", "execution", or "result"
```

Errors identify the exposed tool name and, where applicable, the failing
parameter/field in their message. Pydantic, inspection, and callable exceptions
remain available as `__cause__`. Tests assert phase/name/path information rather
than exact prose or Pydantic's version-dependent full error message.

Preparation rejects invalid metadata, signatures, annotations, defaults, and
duplicates. Argument errors prove the callable was not invoked. Execution/result
errors make no such promise. No default error is converted into model-visible
prose or automatically retried. A future runner may implement an explicit policy.

A future backend must check its own tool/schema support before a request:

- Native definitions and executable handlers come from the same prepared tools.
- Provider conversion uses a fresh schema. It must not silently drop local rules
  while claiming provider-side enforcement, or change argument meanings.
- OpenAI strict mode requires all fields to be required and disallows arbitrary
  object keys. Do not silently treat null as omission to simulate Python defaults.
  Choose a compatible request mode or reject the schema with an actionable error.
- `dict[str, T]` can be a valid local input yet incompatible with a selected
  provider mode. Tool-result dictionaries are not governed by input-schema rules.
- CLI harnesses cannot execute an in-process Python callable merely because its
  schema is included in a prompt. Such transport is outside this milestone.

The executable suite below does not pretend to test these unimplemented
provider adapters. Their later offline tests must assert request envelopes,
unsupported-schema rejection before transport, correlation IDs, native
continuation preservation, and zero execution of unknown names.

## 8. Automated acceptance suite and requirement coverage

Run from the repository root:

```bash
python -m pytest tests/test_tools.py --collect-only -q
python -m pytest tests/test_tools.py -q
python -m pytest -q
python -m ruff check slick tests examples
```

The contract file uses the public/module interfaces above and contains
real pytest assertions, parameterized positive/negative types, and instrumented
functions. It must not use `importorskip`, `xfail`, a mock implementation of Tool,
or network calls. Before implementation the suite was confirmed red because
`slick.tools` did not exist. These cases now run as part of the ordinary suite.

| Requirement | Executable coverage |
| --- | --- |
| D01-D03 | Metadata, explicit overrides, invalid names/descriptions, wrappers, instance/class/static methods |
| D04-D05 | Missing/unresolved types, unsupported signatures/callables, zero arguments |
| D06-D07 | Required/null/default distinction, invalid defaults, default identity, schema isolation |
| Type boundary | Parameterized supported scalars/containers/enums/models/TypedDict and rejected annotations |
| I01-I03 | JSON-only input, cycles/nonfinite data, strict invalid arguments, call counters, input isolation |
| I04-I07 | Bound state, exact call counts, sync/async behavior, cancellation, exception causes, unexpected awaitable |
| R01-R04 | Exact text, JSON round trips, nested model types, invalid/missing/None returns, serializer failure |
| Preparation | Empty/mixed list, explicit wrappers, duplicate names, whole-list failure without invocation |

The contract was promoted to `tests/test_tools.py` with its assertions retained.
Add focused regressions for any newly discovered
edge cases; do not weaken this contract to match an implementation shortcut.
Validate on Python 3.10 with the lowest installable declared Pydantic version,
and on the current supported development environment. Do not silently rely on
an API introduced after the declared Pydantic floor; propose a dependency change
explicitly if it becomes necessary.

## 9. Implementation sequence

1. Implement D01-D07 and preparation; run definition/schema cases red then green.
2. Implement strict argument reconstruction and synchronous invocation; prove
   rejected values never increment the instrumented function counters.
3. Implement typed/untyped result validation and serialization; prove return
   failures execute once and never fall back to stringification.
4. Implement native asynchronous invocation/cancellation and bound-method tests.
5. Run the full contract and existing regression suite, document supported types
   and the explicit blocking behavior of sync tools in async calls.

Finish this milestone before designing native backend turns. A Jinja macro can
later consume `name`, `description`, and `parameters` in an example without owning
execution or replacing the provider-native tools field.

## References

- [Pydantic schemas](https://docs.pydantic.dev/latest/concepts/json_schema/): schema generation and validation/serialization modes.
- [Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/): JSON and Python modes differ.
- [Pydantic serialization](https://docs.pydantic.dev/latest/concepts/serialization/): typed JSON-mode serialization and errors.
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling): argument definitions and client execution.
- [OpenAI schema restrictions](https://developers.openai.com/api/docs/guides/structured-outputs): strict-mode requirements.

The project's installed Pydantic 2.13.4 was probed during specification. It can
generate schemas for tuples, sets, bytes, complex values, dates, and paths; their
exclusion above is a deliberate first-version policy. `bytearray` and `range`
failed schema generation, and `Callable` failed JSON Schema generation. These
observations are not a promise about every Pydantic version.
