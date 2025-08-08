I'll scan the key files to understand current structure and see if anything already touches configuration or LangChain, so the TODO doc can align with existing patterns. Then I'll draft the TODO with scope, API, config, CLI, and LangChain integration details.
Read file: /Users/stuart/Developer/AI/slick/slick/__init__.py
Read file: /Users/stuart/Developer/AI/slick/slick/decorators.py
Read file: /Users/stuart/Developer/AI/slick/slick/cli.py
Read file: /Users/stuart/Developer/AI/slick/pyproject.toml
Read file: /Users/stuart/Developer/AI/slick/README.md

Just checked your current `slick` code: the decorators directly instantiate `ChatOpenAI` with a hardcoded default, and there’s no model/provider registry or global config. I’ll draft a concise TODO/design doc in Markdown as requested [[memory:3243370]].

### TODO: Global Default Model + Multi‑Provider Model Management (via LangChain)

- **Objective**: Add a component that:
  - **Sets and reads a global default model**.
  - **Manages models and providers** (OpenAI, Anthropic, Google, Mistral, Groq, Cohere, Azure OpenAI).
  - **Creates LangChain chat models** from a resolved provider+model.
  - **Exposes CLI** for listing/selecting/testing models.
- **Non‑goals**: Fine‑grained rate limiting, key storage, provider account management.

### User stories
- As a dev, I can set a global default model once and use `@llm_step` without specifying `model=...` each time.
- As a dev, I can pick models by friendly aliases (e.g., `gpt-4o-mini`, `claude-sonnet`) and the system knows the provider.
- As a dev, I can list available providers/models and validate my env keys via CLI.
- As a dev, I can override the global model per call.

### Config resolution order
- Highest to lowest:
  - Explicit function arg `model=`/`provider=`
  - In‑memory runtime set (e.g., `slick.set_default_model(...)`)
  - Env vars: `SLICK_MODEL`, `SLICK_PROVIDER`
  - User config file: `~/.config/slick/config.toml` (mac: `~/Library/Application Support/slick/config.toml` also ok)
  - Project config: `pyproject.toml` under `[tool.slick]`
  - Built‑in library default: `openai:gpt-4o-mini` (or your preferred)
- Provider inference from model name via registry if `provider` omitted.

### Public API (library)
- `slick.models.set_default_model(model: str, provider: str | None = None) -> None`
- `slick.models.get_default_model() -> tuple[str, str]`
- `slick.models.create_chat_model(model: str | None = None, provider: str | None = None, **kwargs) -> BaseChatModel`
- `slick.models.list_providers() -> list[ProviderSpec]`
- `slick.models.list_models(provider: str | None = None) -> list[ModelSpec]`
- `slick.models.resolve(model_or_alias: str) -> ModelSpec`
- `slick.models.is_available(provider: str) -> bool` (checks installed package and key)

### Data model
- `ModelSpec` (pydantic): `id`, `provider`, `aliases`, `context_window`, `supports_json`, `supports_tools`, `supports_vision`, optional price fields.
- `ProviderSpec` (pydantic): `name`, `env_keys` (e.g., `OPENAI_API_KEY`), `pip_extras` (for optional install), `langchain_class` path, optional base URL/deployment fields (Azure).

### Providers and LangChain classes
- **OpenAI**: `langchain_openai.ChatOpenAI` (env: `OPENAI_API_KEY`)
- **Azure OpenAI**: `langchain_openai.AzureChatOpenAI` (env: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`)
- **Anthropic**: `langchain_anthropic.ChatAnthropic` (`ANTHROPIC_API_KEY`)
- **Google**: `langchain_google_genai.ChatGoogleGenerativeAI` (`GOOGLE_API_KEY`)
- **Mistral**: `langchain_mistralai.ChatMistralAI` (`MISTRAL_API_KEY`)
- **Cohere**: `langchain_cohere.ChatCohere` (`COHERE_API_KEY`)
- **Groq**: `langchain_groq.ChatGroq` (`GROQ_API_KEY`)
- Dynamic import with clear error if missing; suggest installing provider extra.

### Registry and aliases
- Ship a curated registry with common models and aliases:
  - `openai:gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `o4-mini-2025-04-16`
  - `anthropic:claude-3-5-sonnet`, `claude-3-7-sonnet`
  - `google:gemini-1.5-pro`, `gemini-2.0-flash`
  - `mistral:mistral-large`, `open-mixtral-8x7b`
  - `cohere:command-r-plus`
  - `groq:llama-3.1-70b`, `llama-3.3-70b`
- Allow users to extend/override via config file.

### Config files
- `~/.config/slick/config.toml` (or macOS Application Support) and/or project `pyproject.toml`:
  - `[tool.slick] default_provider="openai" default_model="gpt-4o-mini"`
  - `[tool.slick.registry]` sections to add aliases/custom endpoints.
- Do not store API keys; only read from env.

### CLI
- `slick models list [--provider <name>]`
- `slick models providers`
- `slick models set-default <model_or_alias> [--provider <name>]`
- `slick models show-default`
- `slick models test [--model <m>] [--provider <p>] [--prompt "hi"]`
- Friendly errors if provider package not installed or env key missing; print install hints.

### Decorator integration
- `llm_step` / `llm_step_async`:
  - Backward compat: `model=` still accepted.
  - If absent, call `create_chat_model()` using global default.
  - Keep `stream` behavior; pass through `**llm_kwargs`.
  - If `return_type` parsing used today, leave unchanged.

### Implementation plan (edits)
- Add `slick/models.py`:
  - Registry + provider map + dynamic import factory.
  - Config loader (env, user, project) with precedence + in‑memory store.
  - API functions listed above.
- Update `slick/decorators.py`:
  - Replace direct `ChatOpenAI` construction with `create_chat_model()`.
  - Default `model=None`; let resolver handle fallback.
- Add CLI subcommands in `slick/cli.py` under `models`.
- Add optional extras in `pyproject.toml`:
  - `[tool.poetry.extras]` e.g., `anthropic=["langchain-anthropic"]`, `google=["langchain-google-genai"]`, etc.
- Add helpful errors and install hints on missing provider packages.
- Update `README.md` with config examples and CLI usage.

### Acceptance criteria
- Without specifying `model=`, `@llm_step` uses the configured global default.
- `slick models set-default` persists default and `show-default` reflects it.
- `slick models providers` lists providers with availability (installed/key present).
- `slick models list` shows curated models and any user registry entries.
- `create_chat_model()` returns a functional LangChain chat model for each supported provider when configured.
- Missing provider module or key produces actionable errors.
- Existing decorator features (streaming, dict/Pydantic parsing, n>1) still work.

### Tests
- Unit tests for resolver precedence, alias mapping, config reading.
- Tests per provider behind marks that skip if package/env missing.
- CLI tests for set-default/list/providers/test flows.
- Decorator tests verifying model factory invoked and outputs unchanged.

### Edge cases
- Azure deployment vs model names; support `deployment` param resolution.
- Model names that overlap across providers; alias registry disambiguates.
- Users provide base URLs (e.g., OpenAI compatible APIs); allow config override for `base_url`.
- Silence streaming when `stream=False` still respected.

### Security
- Keys only via env; never write to config. Surface which env is missing, not the value.

### Versioning & docs
- Minor version bump.
- README: Quickstart for picking models, env setup table, provider matrix.

### Quick examples
- Code:
  ```python
  from slick.models import set_default_model, create_chat_model
  set_default_model("gpt-4o-mini")  # provider inferred as openai
  llm = create_chat_model()  # uses global
  ```
- CLI:
  ```bash
  slick models set-default gpt-4o-mini
  slick models show-default
  slick models providers
  slick models list --provider anthropic
  slick models test --prompt "Hello"
  ```

- I scanned `slick/__init__.py`, `slick/decorators.py`, `slick/cli.py`, and `pyproject.toml`, then drafted a concise design/TODO.
- The doc includes API, registry, config precedence, CLI, provider classes, and acceptance criteria aligned with LangChain.