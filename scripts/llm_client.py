from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-20b"},
    "aionlabs": {"base_url": "https://api.aionlabs.ai/v1", "key_env": "AIONLABS_API_KEY", "model": "aion-labs/aion-2.0"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "model": "gpt-4.1-mini"},
}


@dataclass
class LLMResult:
    text: str
    raw: dict[str, Any]


class LLMError(RuntimeError):
    """A provider request failed."""


def resolve_model(provider: str, model: str | None = None) -> str:
    """Resolve a model without requiring credentials or contacting the provider."""
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}. Choose from {', '.join(PROVIDERS)}")
    return model or os.getenv(f"{provider.upper()}_MODEL") or PROVIDERS[provider]["model"]


def provider_config(provider: str, model: str | None = None) -> tuple[OpenAI, str]:
    """Create a configured client using the shared SDK."""
    provider = provider.strip().lower()
    selected_model = resolve_model(provider, model)
    settings = PROVIDERS[provider]
    api_key = os.getenv(settings["key_env"], "").strip()
    if not api_key:
        raise ValueError(f"Missing {settings['key_env']} in the project .env or environment")
    client = OpenAI(api_key=api_key, base_url=settings["base_url"], timeout=120.0, max_retries=3)
    return client, selected_model


def response_text(raw: dict[str, Any]) -> str:
    """Extract readable text from a serialized chat completion."""
    try:
        content = raw["choices"][0]["message"]["content"]
    except KeyError, IndexError, TypeError:
        content = None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat completion contains no readable text")
    return content.strip()


def describe_api_error(error: APIError) -> str:
    # Provider response bodies can include sensitive data.
    if isinstance(error, APITimeoutError):
        return "request timed out"
    if isinstance(error, APIConnectionError):
        return "could not connect to the provider"
    if isinstance(error, APIStatusError):
        return f"HTTP {error.status_code} ({type(error).__name__})"
    return type(error).__name__


def ask_ai(
    prompt: str,
    provider: str = "groq",
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """Return the model's answer as plain text. Extra kwargs are optional provider-specific parameters."""
    return ask_ai_with_response(prompt, provider, system_prompt, model, **kwargs).text


def ask_ai_with_response(
    prompt: str,
    provider: str = "groq",
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    **kwargs: Any,
) -> LLMResult:
    """Return text and full response (useful for caching/debugging)."""
    client, selected_model = provider_config(provider, model)
    try:
        with client:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                **kwargs,
            )
    except APIError as error:
        raise LLMError(f"{provider} model {selected_model}: {describe_api_error(error)}") from None
    raw = response.model_dump(mode="json")
    try:
        return LLMResult(text=response_text(raw), raw=raw)
    except ValueError as error:
        raise LLMError(f"{provider} model {selected_model}: {error}") from None


def list_models(provider: str = "groq") -> list[str]:
    client, _ = provider_config(provider)
    try:
        with client:
            return sorted({model.id for model in client.models.list()})
    except APIError as error:
        raise LLMError(f"{provider} model listing: {describe_api_error(error)}") from None
