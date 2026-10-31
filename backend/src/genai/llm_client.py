"""LLM client shared by POST /forecast/explain and POST /chat.

Reads credentials from environment variables (loaded from .env locally via
python-dotenv). Supports Azure OpenAI and Ollama's OpenAI-compatible API.
See .env.example for configuration.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessage

# Load .env from the project root explicitly, rather than relying on the
# process's current working directory (which differs depending on how/where
# the server is launched, e.g. from an IDE run configuration). This file
# lives at backend/src/genai/llm_client.py -- .env is 3 levels up.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

_client = None


class LLMConfigError(RuntimeError):
    """Raised when the selected LLM provider is unsupported or unconfigured."""


def get_client() -> OpenAI | AzureOpenAI:
    """Return a cached client for the selected provider."""

    global _client

    if _client is None:
        if LLM_PROVIDER == "ollama":
            if not OLLAMA_MODEL:
                raise LLMConfigError("Missing OLLAMA_MODEL environment variable.")
            _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        elif LLM_PROVIDER == "azure":
            required = {
                "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
                "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
                "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise LLMConfigError(f"Missing Azure OpenAI environment variable(s): {', '.join(missing)}")

            _client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        else:
            raise LLMConfigError(f"Unsupported LLM_PROVIDER {LLM_PROVIDER!r}; use 'ollama' or 'azure'.")

    return _client


def _model_name() -> str:
    if LLM_PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            raise LLMConfigError("Missing OLLAMA_MODEL environment variable.")
        return OLLAMA_MODEL
    if LLM_PROVIDER == "azure":
        if not AZURE_OPENAI_DEPLOYMENT_NAME:
            raise LLMConfigError("Missing AZURE_OPENAI_DEPLOYMENT_NAME environment variable.")
        return AZURE_OPENAI_DEPLOYMENT_NAME
    raise LLMConfigError(f"Unsupported LLM_PROVIDER {LLM_PROVIDER!r}; use 'ollama' or 'azure'.")


def call_chat_json(system_prompt: str, user_message: str) -> str:
    """Call the configured model in JSON mode.

    Returns the raw response content (a JSON string, per response_format).
    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures so the caller
    can translate those into a clean HTTP error.
    """

    client = get_client()

    response = client.chat.completions.create(
        model=_model_name(),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def call_chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
) -> ChatCompletionMessage:
    """General-purpose chat completion call, optionally with tool definitions
    and/or JSON-mode output. Used by the /chat tool-calling flow, where the
    caller needs the raw assistant message (to inspect .tool_calls), not
    just its .content -- unlike call_chat_json, which is JSON-mode only and
    used exclusively by /forecast/explain.

    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures.
    """

    client = get_client()

    kwargs: dict[str, Any] = {"model": _model_name(), "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick manual test for the configured LLM provider.")
    parser.add_argument("--name", default="World", help="Name to greet in the test prompt")
    args = parser.parse_args()

    reply = call_chat_json(
        'You are a helpful assistant. Respond with JSON in this exact shape: {"message": "..."}.',
        f"Say hello to {args.name} in one short sentence.",
    )
    print(reply)
