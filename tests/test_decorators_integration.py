from __future__ import annotations

"""Integration tests around decorators using global default model (skeleton).

These tests will be enabled once the model factory is implemented and mocking is added.
"""

import pytest


@pytest.mark.skip(reason="Networked integration; enable after model factory is implemented")
def test_llm_step_uses_global_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set default via models API; ensure decorator uses it when model= is omitted."""

    # from slick import llm_step
    # from slick.models import set_default_model
    # set_default_model("gpt-4o-mini")
    #
    # @llm_step()
    # def demo(x: str) -> str:
    #     """
    #     Echo: {{ x }}
    #     """
    #
    # assert demo("hi")
    pass


