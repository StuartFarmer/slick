"""HTTP providers. Optional SDKs load when a request is sent."""

import json
import os

from ..tools import make_request
from .base import APIProvider


class OpenAIAPI(APIProvider):
    """OpenAI API with provider-owned messages and client lifecycle."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        timeout: float = 60,
        max_output_tokens: int = 2048,
        max_retries: int = 0,
    ):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def _send(self, request):
        import openai

        with openai.OpenAI(timeout=self.timeout, max_retries=self.max_retries) as client:
            return client.responses.create(**request)

    async def _asend(self, request):
        import openai

        async with openai.AsyncOpenAI(timeout=self.timeout, max_retries=self.max_retries) as client:
            return await client.responses.create(**request)

    def _encode(self, context, prepared, results):
        items = [{"role": "user", "content": context}] if context else []
        for result in results:
            call = result["request"]
            arguments = call["arguments"]
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                }
            )
        items.extend(
            {
                "type": "function_call_output",
                "call_id": result["request"]["id"],
                "output": result["content"],
            }
            for result in results
        )
        payload = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "input": items if results else context,
        }
        if prepared:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": False,
                }
                for tool in prepared.values()
            ]
            payload["parallel_tool_calls"] = False
        return payload

    def _decode(self, response):
        text, requests = [], []
        for item in response.output:
            if item.type == "message":
                text.extend(block.text for block in item.content if block.type == "output_text")
            elif item.type == "function_call":
                requests.append(make_request(item.call_id, item.name, item.arguments))
        return "".join(text), requests


class AnthropicAPI(APIProvider):
    """Anthropic API with provider-owned messages and client lifecycle."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        timeout: float = 60,
        max_output_tokens: int = 2048,
        max_retries: int = 0,
    ):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def _send(self, request):
        import anthropic

        with anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries) as client:
            return client.messages.create(**request)

    async def _asend(self, request):
        import anthropic

        async with anthropic.AsyncAnthropic(
            timeout=self.timeout, max_retries=self.max_retries
        ) as client:
            return await client.messages.create(**request)

    def _encode(self, context, prepared, results):
        messages = [{"role": "user", "content": context}] if context else []
        calls, outputs = [], []
        for result in results:
            call = result["request"]
            calls.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                }
            )
            outputs.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result["content"],
                    "is_error": result.get("is_error", False),
                }
            )
        if results:
            messages.extend(
                [{"role": "assistant", "content": calls}, {"role": "user", "content": outputs}]
            )
        payload = {"model": self.model, "max_tokens": self.max_output_tokens, "messages": messages}
        if prepared:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in prepared.values()
            ]
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return payload

    def _decode(self, response):
        text, requests = [], []
        for item in response.content:
            if item.type == "text":
                text.append(item.text)
            elif item.type == "tool_use":
                requests.append(make_request(item.id, item.name, item.input))
        return "".join(text), requests


class ChatCompletionsProvider(APIProvider):
    """Shared wire format for OpenRouter and LiteLLM."""

    def _encode(self, context, prepared, results):
        messages = [{"role": "user", "content": context}] if context else []
        calls = []
        for result in results:
            call = result["request"]
            arguments = call["arguments"]
            calls.append(
                {
                    "type": "function",
                    "id": call["id"],
                    "function": {
                        "name": call["name"],
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments),
                    },
                }
            )
        if results:
            messages.append({"role": "assistant", "tool_calls": calls})
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result["request"]["id"],
                    "content": result["content"],
                }
                for result in results
            )
        payload = {"messages": messages}
        if prepared:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in prepared.values()
            ]
        return payload

    def _decode(self, response):
        message = response.choices[0].message
        requests = [
            make_request(item.id, item.function.name, item.function.arguments)
            for item in (getattr(message, "tool_calls", None) or [])
        ]
        return message.content or "", requests


class OpenRouterAPI(ChatCompletionsProvider):
    """Direct OpenRouter requests through the optional OpenAI SDK."""

    provider = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60,
        max_output_tokens: int = 2048,
        max_retries: int = 0,
    ):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.api_key = api_key

    def _send(self, request):
        import openai

        with openai.OpenAI(
            api_key=self.api_key if self.api_key is not None else os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as client:
            return client.chat.completions.create(**request)

    async def _asend(self, request):
        import openai

        async with openai.AsyncOpenAI(
            api_key=self.api_key if self.api_key is not None else os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as client:
            return await client.chat.completions.create(**request)

    def _encode(self, context, prepared, results):
        return {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "n": 1,
            **super()._encode(context, prepared, results),
        }


class LiteLLMAPI(ChatCompletionsProvider):
    """Call LiteLLM with the supplied model, context, tools, and options."""

    provider = "litellm"

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 60,
        max_retries: int = 0,
        options: dict | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.options = {} if options is None else options

    def _encode(self, context, prepared, results):
        request = {
            "model": self.model,
            **super()._encode(context, prepared, results),
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            "drop_params": False,
            "stream": False,
            "n": 1,
            **self.options,
        }
        if self.api_base is not None:
            request["api_base"] = self.api_base
        if self.api_key is not None:
            request["api_key"] = self.api_key
        return request

    def _send(self, request):
        import litellm

        return litellm.completion(**request)

    async def _asend(self, request):
        import litellm

        return await litellm.acompletion(**request)
