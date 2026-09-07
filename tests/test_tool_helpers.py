"""Exercise tool conversion independently of callable execution."""

import inspect
import json
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from slick import tools


class Request(BaseModel):
    query: str


class Arguments(BaseModel):
    argument_0: Request = Field(alias="request")
    argument_1: int = Field(default=5, alias="limit")


def test_argument_conversion_preserves_models_and_omits_python_defaults():
    arguments = {"request": {"query": "hello"}}
    kwargs = tools._validate_arguments(arguments, Arguments)

    assert kwargs == {"request": Request(query="hello")}
    assert type(kwargs["request"]) is Request
    assert arguments == {"request": {"query": "hello"}}


def test_parameter_field_does_not_mutate_shared_annotation_metadata():
    metadata = Field(default_factory=lambda: 7, ge=1)
    annotation = Annotated[int, metadata]
    parameter = inspect.Parameter("count", inspect.Parameter.KEYWORD_ONLY)

    field = tools._parameter_field(parameter, annotation)

    assert field.is_required()
    assert field.alias == "count"
    assert metadata.default_factory() == 7
    assert metadata.alias is None


def test_result_serialization_uses_model_aliases():
    class Document(BaseModel):
        document_id: str = Field(alias="id")

    result = tools._serialize_result(Document(id="intro"), TypeAdapter(Document))

    assert json.loads(result) == {"id": "intro"}


def test_result_serialization_rejects_nonfinite_model_fields():
    class Score(BaseModel):
        value: float

    with pytest.raises(ValueError, match="finite"):
        tools._serialize_result(Score(value=float("inf")), TypeAdapter(Score))
