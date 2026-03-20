import ollama
import os
from openai import OpenAI

from config import get_llm_provider, get_ollama_base_url, get_openrouter_api_key, get_openrouter_model

_selected_model: str | None = None


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def _openrouter_client() -> OpenAI:
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("openrouter_api_key not found in config.json")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def list_models() -> list[str]:
    """
    Lists all models available on the local Ollama server.
    OpenRouter models are not listed dynamically to avoid large payload requests.

    Returns:
        models (list[str]): Sorted list of model names.
    """
    provider = get_llm_provider()
    if provider == "openrouter":
        default_model = get_openrouter_model() or "openrouter/free"
        return [_selected_model] if _selected_model else [default_model]
    
    response = _ollama_client().list()
    return sorted(m.model for m in response.models)


def select_model(model: str) -> None:
    """
    Sets the model to use for all subsequent generate_text calls.

    Args:
        model (str): A model name.
    """
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    """
    Returns the currently selected model, or None if none has been selected.
    """
    return _selected_model


def generate_text(prompt: str, model_name: str = None) -> str:
    """
    Generates text using the configured provider.

    Args:
        prompt (str): User prompt
        model_name (str): Optional model name override

    Returns:
        response (str): Generated text
    """
    provider = get_llm_provider()
    model = model_name or _selected_model
    if not model:
        raise RuntimeError(
            "No LLM model selected. Call select_model() first or pass model_name."
        )

    if provider == "openrouter":
        client = _openrouter_client()
        model = model or get_openrouter_model()
        if not model:
            raise RuntimeError("No OpenRouter model selected or configured.")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # Default to Ollama
    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()
