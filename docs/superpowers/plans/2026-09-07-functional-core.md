# Functional Core Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development task by task, with offline tests before implementation and review after changes.

**Goal:** deliver standalone rendering, optional typed prompt shorthand, and text-in/text-out API and CLI backends.

**Architecture:** shared rendering and parsing functions support plain Python and decorated functions. Sync and async execution use the backend's matching method. Native tools and conversation frameworks are deferred.

**Tech Stack:** Python >=3.10, Jinja2, Pydantic 2, optional provider SDKs, stdlib asyncio/subprocess, pytest and ruff.

**Spec:** [Approved functional core](../specs/2026-09-07-functional-core.md).

## Global constraints

- Python >=3.10; Jinja2 and Pydantic only mandatory dependencies.
- Offline validation; no paid calls or publishing.
- Preserve legacy public APIs; no Agent/tool/workflow framework.
- Work in the shared checkout if git metadata prevents isolation. Leave changes uncommitted and reviewable.

## Tasks

- [x] 1. Core (`slick/prompts.py`, `slick/__init__.py`, `tests/test_core.py`): first test standalone rendering with includes/macros/history and strict missing variables; verify `render` is absent. Add public render and parse, then test modern sync/async typed decorators, no default I/O/repair/cache, explicit compatibility options, preserved signatures and cancellation. Reuse existing template/parser helpers. Validate with `pytest tests/test_core.py tests/test_prompts.py`.
- [x] 2. API adapters (`slick/backends.py`, `tests/test_api_backends.py`, optional extras in `pyproject.toml`): first test exact OpenAI/Anthropic request envelopes and completed text using injected fake SDK clients. Implement `OpenAI`/`Anthropic` with call/acall and lazy client lifecycle; add incomplete/refusal/tool errors, missing-extra guidance and async cancellation tests. Owned clients use explicit timeout and retry settings. Run provider tests independently.
- [x] 3. CLI async (`slick/models.py`, `tests/test_models.py`): first test real Python subprocess echo/nonzero exit/timeout/cancellation through `acall`; implement native async process execution and cleanup without changing sync behavior. Run model tests independently.
- [x] 4. Integration/docs (`examples/`, README, changelog, previous planning docs): show summary/history/retrieval using shared core. Test examples using fake backends, run full pytest and ruff, build wheel and inspect optional extras/typing marker. Review the combined diff for compatibility and actual scope before completion.

## Execution record

- Baseline: 54 tests pass.
- No runtime changes existed before this plan; prior docs/outputs are untracked user-session artifacts to preserve.

- Implementation isolated in `/tmp/slick-core-worktree`; changes will be copied into the user checkout after verification.
- Core and provider tests pass; real SDK transport checks use simulated HTTP, with no paid calls.
- Review caught encoding after process spawn; regression fix assigned before final validation.

- Final review: encoding-before-spawn regression fixed and re-reviewed; empty workdir normalized consistently in sync/async execution.
- Validation: 162 tests with optional SDKs; 158 passed and one optional module skipped without extras. Ruff and mypy pass; wheel/sdist build and optional dependency metadata verified. Python 3.10 syntax checked; runtime tests used Python 3.14.
