import asyncio

import pytest

from examples.coding_harness import __main__ as cli
from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.config import HarnessConfig
from tests.coding_harness.test_agent import Script


@pytest.mark.parametrize(
    "override", [["--workspace", "/tmp"], ["--config", "config.json"], ["--dry-run"]]
)
def test_resume_rejects_overrides_before_reading_session(tmp_path, override):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--resume", str(tmp_path / "missing"), *override])
    assert caught.value.code == 2


def test_resume_selects_provider_and_model(workspace, tmp_path, monkeypatch, capsys):
    agent = CodingAgent(Script(("Earlier", [])), workspace, HarnessConfig())
    asyncio.run(agent.run("First task"))
    path = tmp_path.parent / (tmp_path.name + ".json")
    agent.save(path)
    selected = []

    def provider(*, model):
        selected.append(model)
        result = Script(("Continued", []))
        result.identity = lambda: {"provider": "anthropic", "model": model}
        return result

    monkeypatch.setattr(cli.providers, "AnthropicAPI", provider)
    assert (
        cli.main(
            [
                "--resume",
                str(path),
                "--provider",
                "anthropic",
                "--model",
                "test-model",
                "--headless",
                "--task",
                "Continue",
            ]
        )
        == 2
    )
    assert selected == ["test-model"]
    assert "Continued" in capsys.readouterr().out


def test_changing_resume_provider_requires_model(workspace, tmp_path):
    path = tmp_path.parent / (tmp_path.name + ".json")
    CodingAgent(Script(), workspace, HarnessConfig()).save(path)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--resume", str(path), "--provider", "anthropic"])
    assert caught.value.code == 2
