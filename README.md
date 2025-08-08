# slick

Slick: a Python library and CLI.

## Installation

```bash
pip install slick-ai
```

## Usage

- Library:

```python
import slick
print(slick.__version__)
```

- CLI:

```bash
slick --help
```

## LLM decorators

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

Use `llm_step` to turn a function docstring into a prompt. The function body is not executed; the decorator returns the LLM output (optionally parsed to dict or a Pydantic model):

```python
from pydantic import BaseModel
from slick import llm_step

class Answer(BaseModel):
    summary: str

@llm_step(model="gpt-4o-mini")
def summarize(text: str) -> Answer:
    """
    Summarize the following text in one sentence.
    Text: {{ text }}
    """

result = summarize("Slick makes LLM steps easy.")
print(result.summary)
```

- For async code, use `llm_step_async`.
- For `dict` returns, the decorator emits a `dict` via LangChain's `StructuredOutputParser`.
- For `List[...]` returns with `n>1`, multiple generations are returned.

## Development

```bash
# install dependencies (creates/uses Poetry-managed venv)
poetry install --with dev

# run commands within the venv
poetry run pytest
poetry run ruff check .
poetry run mypy slick
```

## Releasing to PyPI

1) Update the version in `pyproject.toml` under `[tool.poetry]`.

2) Build and publish (TestPyPI first is recommended):

```bash
# build artifacts
poetry build

# publish to TestPyPI
poetry publish --repository testpypi

# when ready, publish to PyPI
poetry publish
```

Optionally configure credentials once:

```bash
poetry config pypi-token.pypi <YOUR_PYPI_TOKEN>
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi <YOUR_TEST_PYPI_TOKEN>
```
