"""Generic client for any OpenAI-compatible inference endpoint.

OpenAI-compatible providers expose the Chat Completions API, so a single
parameterized client covers them. Per-provider quirks are handled via
constructor flags.
"""

import time
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from .base import BaseLLMClient, Prompt, StreamChunk


class OpenAICompatibleClient(BaseLLMClient):
    """Client for any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        service_name: str,
        base_url: str,
        timeout: float = 180.0,
        uses_max_completion_tokens: bool = False,
        extra_body: Optional[dict] = None,
    ):
        """Initialize the client.

        Args:
            api_key: API key for the provider
            model: Provider-specific model ID
            service_name: Provider display name (used in metrics/reports)
            base_url: OpenAI-compatible base URL
            timeout: Per-request timeout in seconds
            uses_max_completion_tokens: Send `max_completion_tokens` instead of
                `max_tokens` (required by some providers)
            extra_body: Extra request-body fields merged into every request
                (e.g. OpenRouter `provider` routing controls)
        """
        super().__init__(api_key, model, service_name)
        self.uses_max_completion_tokens = uses_max_completion_tokens
        self.extra_body = extra_body or {}
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def stream_completion(
        self,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion, yielding timestamped chunks.

        TTFT is taken at the first emitted token of EITHER visible content or
        reasoning content, so reasoning models are measured consistently across
        providers. Token usage is accumulated and emitted in a single final
        chunk after the stream closes (handles providers that send usage in a
        trailing choices-empty chunk).
        """
        token_param = "max_completion_tokens" if self.uses_max_completion_tokens else "max_tokens"
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": (
                [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            ),
            token_param: max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.extra_body:
            request_params["extra_body"] = self.extra_body

        stream = await self.client.chat.completions.create(**request_params)

        is_first = True
        completion_tokens = None
        prompt_tokens = None

        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            visible = delta.content or ""
            # Reasoning models stream thinking via `reasoning_content` before
            # any visible content — count it for an honest TTFT.
            reasoning = getattr(delta, "reasoning_content", None) or ""
            piece = visible or reasoning

            if piece:
                yield StreamChunk(
                    content=visible,
                    timestamp=time.perf_counter(),
                    is_first=is_first,
                )
                is_first = False

        # Single authoritative final chunk with usage.
        yield StreamChunk(
            content="",
            timestamp=time.perf_counter(),
            is_final=True,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    async def close(self) -> None:
        """Close the underlying client."""
        await self.client.close()
