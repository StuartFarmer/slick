# llm_script.py
import os, inspect
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser, ResponseSchema
from pydantic import BaseModel
from typing import get_origin
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage

## llm_step: use function docstring as template and inject raw LLM output into the function
def llm_step(fn=None, *, model="o4-mini-2025-04-16", **llm_kwargs):
    """
    Decorator that uses the function's docstring as the LLM prompt.
    Runs the LLM call (with JSON/Pydantic parsing or multi-output support) and returns
    the parsed LLM result directly, without invoking the wrapped function body.
    Supports multi-output for List[...] return types with n>1 via the `n` parameter.
    """
    # support usage with or without args
    if fn is None:
        return lambda f: llm_step(f, model=model, **llm_kwargs)

    sig = inspect.signature(fn)
    return_type = sig.return_annotation
    # pick parser + format_instructions for JSON/Pydantic
    output_parser = None
    format_instructions = ""
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        output_parser = PydanticOutputParser(pydantic_object=return_type)
        format_instructions = output_parser.get_format_instructions()
    elif return_type is dict:
        schema = ResponseSchema(name="output", description="Valid JSON object")
        output_parser = StructuredOutputParser.from_response_schemas([schema])
        format_instructions = output_parser.get_format_instructions()

    # build full prompt from docstring
    template = inspect.cleandoc(fn.__doc__ or "")
    if format_instructions:
        escaped = format_instructions.replace("{", "{{").replace("}", "}}")
        template = template.rstrip() + "\n\n" + escaped

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Extract 'stream' flag to control streaming output
        stream = kwargs.pop('stream', True)
        bound = sig.bind_partial(*args, **kwargs).arguments
        # init LLM
        # Use factory for dynamic provider selection
        try:
            from .models import create_chat_model

            llm = create_chat_model(
                model=model,
                streaming=stream,
                callbacks=[StreamingStdOutCallbackHandler()],
                **llm_kwargs,
            )
        except Exception:
            # Fallback: Configure ChatOpenAI with streaming if requested
            # (kept for backward compatibility if factory not usable)
            llm_init_kwargs = {
                'model': model,
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                **llm_kwargs
            }
            if stream:
                llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], **llm_init_kwargs)
            else:
                llm = ChatOpenAI(**llm_init_kwargs)
        llm_init_kwargs = {
            'model': model,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            **llm_kwargs
        }
        if stream:
            llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], **llm_init_kwargs)
        else:
            llm = ChatOpenAI(**llm_init_kwargs)
        origin = get_origin(return_type)
        # multi-output support
        if origin == list and llm_kwargs.get('n', 1) > 1:
            prompt_str = PromptTemplate(input_variables=list(bound.keys()), template=template, template_format="jinja2").format(**bound)
            messages = [HumanMessage(content=prompt_str)]
            gens = llm.generate([messages]).generations[0]
            raw = [(g.message.content if hasattr(g, 'message') else g.text) for g in gens]
        else:
            # single-output or JSON
            seq = PromptTemplate(input_variables=list(bound.keys()), template=template, template_format="jinja2") | llm
            if output_parser:
                seq = seq | output_parser
            raw = seq.invoke(bound)
            # If the raw output is a BaseMessage (e.g., AIMessage), extract its content
            if isinstance(raw, BaseMessage):
                raw = raw.content
            if return_type is dict:
                raw = raw['output']
        # return the LLM output directly
        return raw
    return wrapper


## llm_step_async: async version of llm_step
def llm_step_async(fn=None, *, model="o4-mini-2025-04-16", **llm_kwargs):
    """
    Async decorator that uses the function's docstring as the LLM prompt.
    Same as llm_step but for async functions.
    """
    # support usage with or without args
    if fn is None:
        return lambda f: llm_step_async(f, model=model, **llm_kwargs)

    sig = inspect.signature(fn)
    return_type = sig.return_annotation
    # pick parser + format_instructions for JSON/Pydantic
    output_parser = None
    format_instructions = ""
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        output_parser = PydanticOutputParser(pydantic_object=return_type)
        format_instructions = output_parser.get_format_instructions()
    elif return_type is dict:
        schema = ResponseSchema(name="output", description="Valid JSON object")
        output_parser = StructuredOutputParser.from_response_schemas([schema])
        format_instructions = output_parser.get_format_instructions()

    # build full prompt from docstring
    template = inspect.cleandoc(fn.__doc__ or "")
    if format_instructions:
        escaped = format_instructions.replace("{", "{{").replace("}", "}}")
        template = template.rstrip() + "\n\n" + escaped

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        # Extract 'stream' flag to control streaming output
        stream = kwargs.pop('stream', True)
        bound = sig.bind_partial(*args, **kwargs).arguments
        # init LLM
        # Use factory for dynamic provider selection
        try:
            from .models import create_chat_model

            llm = create_chat_model(
                model=model,
                streaming=stream,
                callbacks=[StreamingStdOutCallbackHandler()],
                **llm_kwargs,
            )
        except Exception:
            # Fallback: Configure ChatOpenAI with streaming if requested
            llm_init_kwargs = {
                'model': model,
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                **llm_kwargs
            }
            if stream:
                llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], **llm_init_kwargs)
            else:
                llm = ChatOpenAI(**llm_init_kwargs)
        llm_init_kwargs = {
            'model': model,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            **llm_kwargs
        }
        if stream:
            llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], **llm_init_kwargs)
        else:
            llm = ChatOpenAI(**llm_init_kwargs)
        origin = get_origin(return_type)
        # multi-output support
        if origin == list and llm_kwargs.get('n', 1) > 1:
            prompt_str = PromptTemplate(input_variables=list(bound.keys()), template=template, template_format="jinja2").format(**bound)
            messages = [HumanMessage(content=prompt_str)]
            gens = await llm.agenerate([messages])
            raw = [(g.message.content if hasattr(g, 'message') else g.text) for g in gens.generations[0]]
        else:
            # single-output or JSON
            seq = PromptTemplate(input_variables=list(bound.keys()), template=template, template_format="jinja2") | llm
            if output_parser:
                seq = seq | output_parser
            raw = await seq.ainvoke(bound)
            # If the raw output is a BaseMessage (e.g., AIMessage), extract its content
            if isinstance(raw, BaseMessage):
                raw = raw.content
            if return_type is dict:
                raw = raw['output']
        # return the LLM output directly
        return raw
    return wrapper
