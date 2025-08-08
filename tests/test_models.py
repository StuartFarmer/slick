from __future__ import annotations

"""Tests for provider listing, defaults, and factory skeleton.

What this file will cover once implemented:
- Registry resolution by id and alias
- Provider listing and basic availability reporting
- Default model set/get (in-memory only initially)
- Factory error behavior before implementation
"""

import pytest

from slick import __version__  # sanity import


def test_imports_sanity() -> None:
    """Ensure the package imports cleanly."""

    assert isinstance(__version__, str)


@pytest.mark.skip(reason="Pending models implementation")
def test_set_and_get_default_model() -> None:
    """Set default and read it back; ensure tuple shape (model, provider)."""

    # from slick import models
    # models.set_default_model("gpt-4o-mini")
    # m, p = models.get_default_model()
    # assert m == "gpt-4o-mini"
    pass


@pytest.mark.skip(reason="Pending models implementation")
def test_list_providers_and_models() -> None:
    """List providers and models (names only) for a given provider."""

    # from slick import models
    # providers = models.list_providers()
    # assert any(p.name == "openai" for p in providers)
    # model_names = models.list_models(provider="openai")
    # assert isinstance(model_names, list)
    pass


@pytest.mark.skip(reason="Pending models implementation")
def test_create_chat_model_not_implemented() -> None:
    """Until implemented, factory should raise NotImplementedError."""

    # from slick import models
    # with pytest.raises(NotImplementedError):
    #     models.create_chat_model()
    pass


