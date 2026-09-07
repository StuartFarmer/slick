# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Native async `OpenAI.aturn` and `Anthropic.aturn` with callable tool definitions,
  structured history, correlated results and preserved provider payloads. Each
  request returns one turn; applications own execution and history.
- Independent coding harness example with a Textual TUI, headless offline demo,
  workspace tools, bounded subprocesses, real verification/repair, Jinja skills,
  context compaction and explicit JSON sessions. Textual stays example-only.
- Offline native transport, workspace, repair, session and UI acceptance tests.
- Callable tools via `slick.tools.prepare_tools`, with optional `Tool` overrides,
  inferred schemas, bound methods, strict argument/return validation, sync/async
  invocation, and phased `ToolError` diagnostics. No decorator or provider loop.
- Automated callable-tool contract tests, including Python 3.10/Pydantic 2.0 checks.
- Rendering-only `Prompt("file.j2")` callable, independent of backends and state.
- `QuestionAnswerer` example showing ordinary class methods and instance-owned
  history with separate answer and critique templates.
- Public `render(template, **variables)` and `parse(text, returns=str)` functions,
  shared with the prompt decorator.
- `@prompt(backend=...)` and native async decorated calls/rendering; modern calls
  default to no logging, cache or automatic repair. Legacy sync defaults remain.
- Optional OpenAI Responses and Anthropic Messages adapters with sync/async calls,
  explicit limits/retries, lazy SDK loading and owned/injected client lifecycle.
- Native async CLI `acall`/`aexecute` with timeout/cancellation process cleanup.
- Offline examples for summaries, templated conversation history and retrieval.

### Changed
- Two prompting examples use string constraints compatible with Pydantic 2.0.
- Prompt validation errors expose the rejected text in `.response`.
- API SDKs are optional extras; the core still depends only on Jinja2 and Pydantic.

## [0.2.0]

Rewritten around one primitive: a prompt is a typed Python function whose
docstring is a Jinja template.

### Added
- `@prompt` (`slick.prompts`): parameters as template variables, a Jinja
  template file as the prompt, return annotation as the output contract, body as
  optional computed context. Structured returns via `pydantic.TypeAdapter` with
  a `{{ output_format }}` schema slot and a parse-and-repair loop.
- `@prompt(template="name.md.j2")` resolves against `TEMPLATE_ROOT` (default
  `prompts/`), so `{% include %}` and friends compose across files and the
  docstring is free to be documentation. Templates load per call, so editing one
  needs no restart. Omitting `template=` keeps the docstring as the template.
  Inspect either with `.source()` and `.template_name`.
- Content-addressed run logging under `LOG_DIR`, doubling as a cache.
- CLI-backed models (`slick.models`): `ClaudeModel`, `CodexModel`, and default
  resolution via `set_default()` / `$SLICK_BACKEND` / `$SLICK_MODEL`.
- `slick model` and `slick call` CLI commands.

### Removed
- `llm_step` / `llm_step_async` and the LangChain dependency. Both were broken:
  the provider factory's result was discarded and `ChatOpenAI` rebuilt
  unconditionally, so every call went to OpenAI; and Pydantic returns raised
  `TemplateSyntaxError` because `str.format` brace-escaping was applied to a
  Jinja template. Structured output, parsing, and repair now live in `@prompt`,
  which is what LangChain was there for.
- `slick.providers` (7 provider classes) — unreachable from the above bug, and
  superseded by `Model.call(prompt) -> str` as the whole model interface.
- `slick.config` — TOML/user/project config layers that were stub `return {}`.
- `slick models list|providers|set-default|show-default|test`. `set-default`
  could not work: it mutated in-memory state and the process then exited.
- Streaming. It defaulted to on and wrote to stdout from library code.

## [0.1.0] - Initial release
- Initial project scaffolding
