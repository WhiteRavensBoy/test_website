import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_web_main(monkeypatch, *, ollama_host=None, docker_env=False):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    if ollama_host is not None:
        monkeypatch.setenv("OLLAMA_HOST", ollama_host)

    def fake_exists(path):
        return docker_env and path == "/.dockerenv"

    monkeypatch.setattr(os.path, "exists", fake_exists)
    sys.modules.pop("web_main", None)

    with patch("ollama.Client") as client_cls:
        module = importlib.import_module("web_main")

    return module, client_cls


def test_ollama_host_uses_environment_variable(monkeypatch):
    module, client_cls = import_web_main(monkeypatch, ollama_host="http://custom-host:11434")

    assert module.OLLAMA_HOST == "http://custom-host:11434"
    assert client_cls.call_args.kwargs["host"] == "http://custom-host:11434"


def test_ollama_host_uses_local_default_when_not_in_docker(monkeypatch):
    module, client_cls = import_web_main(monkeypatch, docker_env=False)

    assert module.OLLAMA_HOST == "http://127.0.0.1:11434"
    assert client_cls.call_args.kwargs["host"] == "http://127.0.0.1:11434"


def test_ollama_host_uses_docker_default(monkeypatch):
    module, client_cls = import_web_main(monkeypatch, docker_env=True)

    assert module.OLLAMA_HOST == "http://host.docker.internal:11434"
    assert client_cls.call_args.kwargs["host"] == "http://host.docker.internal:11434"


@pytest.mark.asyncio
async def test_chat_ai_returns_message_content(monkeypatch):
    module, _ = import_web_main(monkeypatch, ollama_host="http://custom-host:11434")
    module.client.chat = MagicMock(return_value={"message": {"content": "hello from llm"}})

    result = await module.chat_ai("hello")

    assert result == "hello from llm"
    module.client.chat.assert_called_once_with(
        model=module.model,
        messages=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.asyncio
async def test_calculate_time_decorator(monkeypatch):
    module, _ = import_web_main(monkeypatch)

    @module.calculate_time
    async def sample():
        return "ok"

    assert await sample() == "ok"


@pytest.mark.asyncio
async def test_calculate_user_count_decorator(monkeypatch):
    module, _ = import_web_main(monkeypatch)
    module.count = 0

    @module.calculate_user_count
    async def sample(query: str):
        return f"answer:{query}"

    result = await sample("abc")

    assert result == "answer:abc"
    assert module.count == 1


def test_greeting_endpoint(monkeypatch):
    module, _ = import_web_main(monkeypatch)
    client = TestClient(module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Hi, arun...."}


def test_ask_ai_endpoint(monkeypatch):
    module, _ = import_web_main(monkeypatch)
    module.chat_ai = AsyncMock(return_value="llm answer")
    module.count = 0
    client = TestClient(module.app)

    response = client.get("/ask/hello")

    assert response.status_code == 200
    assert response.text.startswith("User : hello")
    assert "llm answer" in response.text
    assert module.count == 1


def test_ask_ai_agent_endpoint(monkeypatch):
    module, _ = import_web_main(monkeypatch)
    module.ask_agent = MagicMock(return_value={"reply": "agent response"})
    client = TestClient(module.app)

    response = client.get("/ask_agent/test")

    assert response.status_code == 200
    assert response.json() == {"reply": "agent response"}
    module.ask_agent.assert_called_once_with(query="test")


@pytest.mark.asyncio
async def test_generate_with_vllm(monkeypatch):
    module, _ = import_web_main(monkeypatch)
    module.llm = MagicMock()
    module.llm.generate.return_value = "vllm-output"
    module.sampling_params = object()

    result = await module.generate_with_vllm("prompt")

    assert result == "vllm-output"
    module.llm.generate.assert_called_once_with("prompt", sampling_params=module.sampling_params)
