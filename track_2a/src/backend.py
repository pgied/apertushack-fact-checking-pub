"""The connection to the backend that serves the model: an OpenAI-compatible client, and the .env file with its key."""

import os
from pathlib import Path

from omegaconf import DictConfig
from openai import OpenAI


def make_client(backend: DictConfig) -> OpenAI:
    """An OpenAI-compatible client for the backend.

    Its API key is the first of the backend's `api_key_env` variables that is set, in the environment
    or in .env. Its URL is in the configuration, where the CSCS backend takes it from BASE_URL.
    """
    api_key = "unused"  # the client needs a key, Ollama ignores it
    if backend.api_key_env:
        names = [backend.api_key_env] if isinstance(backend.api_key_env, str) else list(backend.api_key_env)
        api_key = next((os.environ[name] for name in names if os.environ.get(name)), None)
        if not api_key:
            raise SystemExit(f"{' or '.join(names)} is not set: put it in .env (see .env.example)")
    return OpenAI(base_url=backend.url, api_key=api_key, timeout=300, max_retries=3)


def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key and not key.startswith("#"):
            value = value.split(" #", 1)[0]  # drop a trailing comment
            os.environ.setdefault(key, value.strip().strip("'\""))
