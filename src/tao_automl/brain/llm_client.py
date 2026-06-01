# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM Client abstraction for AutoML brain capabilities.

Supports OpenAI-compatible APIs (OpenAI, NVIDIA NIM, vLLM, Ollama, etc.)
with configurable endpoint, model, API key, retry logic, and token tracking.
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client, resolved from params -> env -> defaults."""

    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]] = None) -> "LLMConfig":
        """Build config from algorithm_specific_params, falling back to env vars."""
        params = params or {}
        return cls(
            endpoint=params.get(
                "llm_endpoint",
                os.getenv("AUTOML_LLM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
            ),
            model=params.get(
                "llm_model",
                os.getenv("AUTOML_LLM_MODEL", "meta/llama-3.1-70b-instruct")
            ),
            api_key=params.get(
                "llm_api_key",
                os.getenv("AUTOML_LLM_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
            ),
            temperature=float(params.get(
                "llm_temperature",
                os.getenv("AUTOML_LLM_TEMPERATURE", "0.7")
            )),
            max_tokens=int(params.get(
                "llm_max_tokens",
                os.getenv("AUTOML_LLM_MAX_TOKENS", "4096")
            )),
            timeout=int(params.get(
                "llm_timeout",
                os.getenv("AUTOML_LLM_TIMEOUT", "120")
            )),
            max_retries=int(params.get(
                "llm_max_retries",
                os.getenv("AUTOML_LLM_MAX_RETRIES", "3")
            )),
            retry_delay=float(params.get(
                "llm_retry_delay",
                os.getenv("AUTOML_LLM_RETRY_DELAY", "2.0")
            )),
        )


@dataclass
class LLMUsage:
    """Tracks cumulative token usage across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    def record(self, prompt_tok: int, completion_tok: int, latency_ms: float):
        """Record token usage from an LLM call."""
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += prompt_tok + completion_tok
        self.num_calls += 1
        self.total_latency_ms += latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize usage stats to dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": self.num_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "errors": self.errors,
        }


@dataclass
class LLMResponse:
    """Parsed LLM response."""

    content: str = ""
    json_content: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Return True if no error occurred."""
        return self.error is None


class LLMClient:
    """OpenAI-compatible LLM client with retry logic, JSON extraction, and usage tracking."""

    def __init__(self, config: Optional[LLMConfig] = None, params: Optional[Dict[str, Any]] = None):
        """Initialize the LLMClient."""
        if config is None:
            config = LLMConfig.from_params(params)
        self.config = config
        self.usage = LLMUsage()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                logger.info(
                    "Initializing OpenAI client: endpoint=%s, model=%s, api_key=%s, timeout=%d",
                    self.config.endpoint,
                    self.config.model,
                    ("***" + self.config.api_key[-6:]) if len(self.config.api_key) > 6 else "(unset)",
                    self.config.timeout,
                )
                self._client = OpenAI(
                    base_url=self.config.endpoint,
                    api_key=self.config.api_key or "not-set",
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.warning(
                    "openai package not installed. Install with: pip install openai"
                )
                raise
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request with retry logic.

        Args:
            messages: List of {role, content} dicts.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            json_mode: If True, request JSON output and attempt to parse it.

        Returns:
            LLMResponse with content and optional parsed json_content.
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                client = self._get_client()
                t0 = time.monotonic()
                response = client.chat.completions.create(**kwargs)
                latency = (time.monotonic() - t0) * 1000

                content = response.choices[0].message.content or ""
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                    }
                    self.usage.record(
                        usage["prompt_tokens"],
                        usage["completion_tokens"],
                        latency,
                    )

                result = LLMResponse(
                    content=content,
                    usage=usage,
                    latency_ms=latency,
                )

                if json_mode:
                    result.json_content = self._extract_json(content)

                logger.info(
                    "LLM call succeeded (attempt %d/%d, %.0fms, %d tokens)",
                    attempt, self.config.max_retries, latency,
                    usage.get("completion_tokens", 0),
                )
                return result

            except Exception as e:
                last_error = str(e)
                self.usage.errors += 1
                cause_chain = []
                exc = e
                while exc is not None:
                    cause_chain.append(f"{type(exc).__name__}: {exc}")
                    exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s | cause chain: %s",
                    attempt, self.config.max_retries, last_error,
                    " -> ".join(cause_chain),
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        return LLMResponse(error=f"All {self.config.max_retries} attempts failed: {last_error}")

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Best-effort JSON extraction from LLM output."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        for open_char, close_char in [("{", "}"), ("[", "]")]:
            start = text.find(open_char)
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == open_char:
                        depth += 1
                    elif text[i] == close_char:
                        depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning("Could not extract JSON from LLM response")
        return None

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return cumulative usage stats."""
        return self.usage.to_dict()
