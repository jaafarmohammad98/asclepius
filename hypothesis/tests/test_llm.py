"""The LLM is reached with the researcher's own key, or a local model; a settings change switches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from hypothesis_engine.build import main
from hypothesis_engine.llm import (
    ENV,
    LLMError,
    LLMNotConfigured,
    Message,
    OutgoingRequest,
    Settings,
    connect,
    load_settings,
)

KEY = "sk-ant-api03-SECRET-marker-0123456789"


def test_no_key_and_no_local_model_is_a_clear_error(tmp_path: Path):
    with pytest.raises(LLMNotConfigured, match="ANTHROPIC_API_KEY") as error:
        load_settings(env={}, path=tmp_path / "llm.toml")
    assert "local" in str(error.value)


def test_a_key_in_the_environment_is_enough_for_the_default_cloud_model(tmp_path: Path):
    settings = load_settings(env={"ANTHROPIC_API_KEY": KEY}, path=tmp_path / "llm.toml")
    assert (settings.backend, settings.model) == ("anthropic", "claude-opus-5-5")


def test_the_environment_overrides_the_settings_file(tmp_path: Path):
    path = tmp_path / "llm.toml"
    path.write_text(f'backend = "anthropic"\napi_key = "{KEY}"\nmodel = "claude-sonnet-5-5"\n')
    settings = load_settings(env={"ASCLEPIUS_LLM_MODEL": "claude-haiku-5-5"}, path=path)
    assert (settings.model, settings.api_key) == ("claude-haiku-5-5", KEY)


def test_a_local_backend_must_name_its_model(tmp_path: Path):
    with pytest.raises(LLMNotConfigured, match="no model"):
        load_settings(env={"ASCLEPIUS_LLM_BACKEND": "local"}, path=tmp_path / "llm.toml")


class FakeLocalServer:
    """Answers chat-completion requests as Ollama would; keeps what it was sent."""

    def __init__(self, reply: str = "A mechanism.") -> None:
        self.reply = reply
        self.received: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        self.received.append((url, body))
        return {
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": self.reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }


LOCAL = Settings(backend="local", model="llama3.1:8b", url="http://localhost:11434/v1")
MESSAGES: list[Message] = [
    {"role": "system", "content": "You propose mechanisms."},
    {"role": "user", "content": "Why is allopregnanolone lower?"},
]


def test_a_local_model_answers_and_the_call_records_model_and_settings():
    server = FakeLocalServer()
    completion = connect(LOCAL, local_transport=server).complete(MESSAGES)
    assert completion.text == "A mechanism."
    assert completion.record() == {
        "backend": "local",
        "model": "llama3.1:8b",
        "settings": {
            "backend": "local",
            "model": "llama3.1:8b",
            "max_tokens": 16000,
            "url": "http://localhost:11434/v1",
        },
        "stop_reason": "stop",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    ((url, body),) = server.received
    assert url == "http://localhost:11434/v1/chat/completions"
    assert body["messages"] == MESSAGES


def test_the_hook_sees_every_request_before_it_is_sent():
    seen: list[OutgoingRequest] = []
    llm = connect(LOCAL, on_request=seen.append, local_transport=FakeLocalServer())
    llm.complete(MESSAGES)
    llm.complete([{"role": "user", "content": "And DHT?"}])
    assert [r.url for r in seen] == ["http://localhost:11434/v1/chat/completions"] * 2
    assert seen[1].body["messages"] == [{"role": "user", "content": "And DHT?"}]
    assert "And DHT?" in seen[1].text()


def test_a_hook_that_raises_stops_the_request():
    server = FakeLocalServer()

    def refuse(request: OutgoingRequest) -> None:
        raise RuntimeError("not this one")

    with pytest.raises(RuntimeError):
        connect(LOCAL, on_request=refuse, local_transport=server).complete(MESSAGES)
    assert server.received == []


class FakeAnthropic:
    """Answers Messages API requests as the Anthropic API would; keeps the raw requests."""

    def __init__(self, status: int = 200, answer: dict[str, Any] | None = None) -> None:
        self.status = status
        self.answer = answer
        self.received: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.received.append(request)
        if self.status != 200:
            error = {"type": "authentication_error", "message": "invalid x-api-key"}
            return httpx2.Response(self.status, json={"type": "error", "error": error})
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json=self.answer
            or {
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "A mechanism."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        )

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key=KEY, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(self))
        )


CLOUD = Settings(backend="anthropic", model="claude-opus-5-5", effort="high", api_key=KEY)


def test_claude_answers_with_the_researchers_key_and_system_messages_become_instructions():
    api = FakeAnthropic()
    completion = connect(CLOUD, anthropic_client=api.client()).complete(MESSAGES)
    assert completion.text == "A mechanism."
    assert completion.record() == {
        "backend": "anthropic",
        "model": "claude-opus-5-5",
        "settings": {
            "backend": "anthropic",
            "model": "claude-opus-5-5",
            "max_tokens": 16000,
            "effort": "high",
        },
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 4},
    }
    (sent,) = api.received
    assert KEY in sent.headers.get("authorization", "") + sent.headers.get("x-api-key", "")
    body = json.loads(sent.content)
    assert body["system"] == "You propose mechanisms."
    assert body["messages"] == [{"role": "user", "content": "Why is allopregnanolone lower?"}]
    assert body["output_config"] == {"effort": "high"}


def test_switching_back_ends_is_a_settings_change_only(tmp_path: Path):
    path = tmp_path / "llm.toml"
    api, server = FakeAnthropic(), FakeLocalServer()

    def answer() -> str:
        llm = connect(load_settings({}, path), anthropic_client=api.client(), local_transport=server)
        return llm.complete(MESSAGES).backend

    path.write_text(f'api_key = "{KEY}"\n')
    assert answer() == "anthropic"
    path.write_text(f'backend = "local"\nmodel = "llama3.1:8b"\napi_key = "{KEY}"\n')
    assert answer() == "local"
    assert (len(api.received), len(server.received)) == (1, 1)


def test_the_hook_sees_the_cloud_request_too_and_never_the_key():
    seen: list[OutgoingRequest] = []
    api = FakeAnthropic()
    connect(CLOUD, on_request=seen.append, anthropic_client=api.client()).complete(MESSAGES)
    (request,) = seen
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.body == json.loads(api.received[0].content)
    assert KEY not in request.text() and KEY not in repr(request)


def test_the_key_is_in_no_repr_record_or_error():
    llm = connect(CLOUD, anthropic_client=FakeAnthropic(status=401).client())
    assert KEY not in repr(CLOUD) and KEY not in repr(llm) and KEY not in str(CLOUD.public())
    with pytest.raises(LLMError, match="401") as error:
        llm.complete(MESSAGES)
    assert KEY not in str(error.value) and KEY not in repr(error.value)
    assert error.value.__cause__ is None and error.value.__context__ is None


def test_a_refusal_is_an_error_not_an_answer():
    refusal = {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": "refusal",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 0},
    }
    with pytest.raises(LLMError, match="declined"):
        connect(CLOUD, anthropic_client=FakeAnthropic(answer=refusal).client()).complete(MESSAGES)


def test_llm_check_without_settings_says_how_to_set_them_up(tmp_path, monkeypatch, capsys):
    for variable in ENV:
        monkeypatch.delenv(variable, raising=False)
    assert main(["llm-check", "--settings", str(tmp_path / "llm.toml")]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
