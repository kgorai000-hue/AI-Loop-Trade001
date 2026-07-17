"""Anthropic Messages API client with exponential backoff and prompt caching."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AnthropicClientError(RuntimeError):
    pass


class AnthropicClient:
    """
    Thin wrapper around the official `anthropic` SDK.
    API key: ANTHROPIC_API_KEY environment variable only.
    """

    def __init__(
        self,
        max_retries: int = 5,
        enable_prompt_cache: bool = True,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 60.0,
    ) -> None:
        self.max_retries = max(1, int(max_retries))
        self.enable_prompt_cache = bool(enable_prompt_cache)
        self.base_delay_sec = float(base_delay_sec)
        self.max_delay_sec = float(max_delay_sec)
        self._client = None

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AnthropicClientError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise AnthropicClientError(
                "anthropic package not installed; pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _system_blocks(
        self,
        system: str,
        cached_context: Optional[str] = None,
    ) -> list[dict[str, Any]] | str:
        if not self.enable_prompt_cache or not cached_context:
            if cached_context:
                return f"{system}\n\n---\nCACHED CONTEXT\n---\n{cached_context}"
            return system

        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": system,
            },
            {
                "type": "text",
                "text": cached_context,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        return blocks

    def _is_retryable(self, exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "rate" in name or "rate_limit" in msg or "429" in msg:
            return True
        if "overloaded" in msg or "timeout" in msg or "timed out" in msg:
            return True
        if "500" in msg or "502" in msg or "503" in msg or "529" in msg:
            return True
        # anthropic SDK exception attributes
        status = getattr(exc, "status_code", None)
        if status in (408, 429, 500, 502, 503, 529):
            return True
        return False

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        user: str,
        cached_context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        client = self._get_client()
        system_payload = self._system_blocks(system, cached_context=cached_context)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user}],
                }
                if isinstance(system_payload, list):
                    kwargs["system"] = system_payload
                else:
                    kwargs["system"] = system_payload

                response = client.messages.create(**kwargs)
                parts: list[str] = []
                for block in response.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                return "\n".join(parts).strip()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise AnthropicClientError(str(exc)) from exc
                delay = min(
                    self.max_delay_sec,
                    self.base_delay_sec * (2 ** (attempt - 1)),
                )
                delay *= 0.5 + random.random()  # jitter
                logger.warning(
                    "Anthropic call failed (attempt %d/%d): %s; sleep %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise AnthropicClientError(str(last_exc) if last_exc else "unknown error")

    @staticmethod
    def extract_json(text: str) -> Any:
        """Parse JSON from a model reply; tolerate fenced blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first fence and optional trailing fence
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # find first { or [
            start_obj = text.find("{")
            start_arr = text.find("[")
            starts = [i for i in (start_obj, start_arr) if i >= 0]
            if not starts:
                raise
            start = min(starts)
            snippet = text[start:]
            # balance braces roughly
            for end in range(len(snippet), 0, -1):
                try:
                    return json.loads(snippet[:end])
                except json.JSONDecodeError:
                    continue
            raise
