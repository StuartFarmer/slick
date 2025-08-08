from __future__ import annotations

"""Tests for config precedence and env reading (skeleton)."""

import pytest


@pytest.mark.skip(reason="Pending config/models implementation")
def test_env_vars_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env should take effect if in-memory default not set."""

    # from slick import config
    # monkeypatch.setenv("SLICK_MODEL", "gpt-4o-mini")
    # m, p = config.resolve_defaults()
    # assert m == "gpt-4o-mini"
    pass


