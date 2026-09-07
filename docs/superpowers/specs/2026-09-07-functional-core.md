# Slick functional core

Approved in conversation on 2026-09-07. This supersedes the September 6 backend/toolbox implementation scope. Earlier tool and Agent designs remain research, not implementation requirements.

The core is Python arguments → Jinja text → backend text → Python value. Expose `render(template, /, **variables) -> str` and `parse(text, returns=str)`. Reuse both in `@prompt`; preserve includes/macros/inheritance, strict undefined variables, current template root and reload behavior. Standalone rendering does not inject instructions. Typed decorated functions render the existing output instructions and validate locally.

Backends support `call(text) -> str` and/or `acall(text) -> str`. Add optional OpenAI Responses and Anthropic Messages adapters with explicit model IDs, timeout and output limits, lazy SDK imports, per-call owned clients or application-owned injected clients. Disable automatic SDK retries by default; let callers configure them. Preserve native failure causes, reject incomplete/refused/tool outputs instead of treating them as successful text. No tools, conversation objects, native structured-output protocol or run framework.

Add native async execution to existing CLI models, preserving command configuration. Async subprocess timeout/cancellation must stop and reap the owned process. Keep the model API and CLI compatible.

`@prompt(backend=object)` is the new path. Plain `def` uses `call`, `async def` uses `acall`; async `.render()` is awaited. No event-loop detection or sync-to-thread fallback for backend execution. Reject model/backend ambiguity and unsupported execution modes clearly. Preserve signatures/docstrings and result typing. Existing computed-context bodies remain supported for compatibility.

Modern calls default to no cache, no logs, no automatic repair. Explicit `cache`, `log_dir`, and `max_repairs` opt into existing behavior; SDK retries are separate. Legacy `model=` and bare decorators retain existing sync defaults and overrides. Explicit modern disk persistence requires a stable backend identity; simple custom call/acall objects work without metadata when persistence is off. New async calls default to modern no-persistence/no-repair behavior even when resolving the legacy default model.

Global constraints: Python >=3.10; Jinja2 and Pydantic only mandatory third-party dependencies; ordinary tests are offline; no live paid calls, publishing or git commits required. Changes remain reviewable in the current checkout when git metadata is read-only.

Ship examples for a typed summary, a history rendered with Jinja, and externally retrieved documents. Application Python owns retrieval and history storage. Update README, changelog, optional extras and mark earlier plans as superseded.
