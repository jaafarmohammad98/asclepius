"""The settings screen (M4-09, spec H4): the researcher's own API key or a local model, stored in the
hypothesis engine's settings file, the key never sent back to the page."""

import json
import stat
from functools import partial

import pytest
from hypothesis_engine.llm import connect, load_settings

from asclepius.web import llm_settings
from asclepius.web.llm_settings import SettingsError

KEY = "sk-ant-test-0123456789abcdef"
CLOUD = {"backend": "anthropic", "anthropic": {"api_key": KEY, "model": "claude-opus-5-5", "effort": "high"}}


@pytest.fixture
def path(tmp_path):
    return tmp_path / "llm.toml"


def test_a_saved_key_and_model_are_what_the_engine_reads(path):
    llm_settings.save(path, CLOUD, env={})
    settings = load_settings({}, path)
    assert (settings.backend, settings.model, settings.effort, settings.api_key) == (
        "anthropic",
        "claude-opus-5-5",
        "high",
        KEY,
    )


def test_the_key_is_never_sent_back(path):
    shown = llm_settings.save(path, CLOUD, env={})
    assert KEY not in json.dumps(shown) and KEY not in json.dumps(llm_settings.read(path, env={}))
    assert shown["anthropic"]["key_set"] is True
    assert shown["active"] == {
        "backend": "anthropic",
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "effort": "high",
    }


def test_only_the_researcher_can_read_the_file(path):
    llm_settings.save(path, CLOUD, env={})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_blank_key_keeps_the_stored_one(path):
    llm_settings.save(path, CLOUD, env={})
    llm_settings.save(
        path, {"backend": "anthropic", "anthropic": {"api_key": "", "model": "claude-sonnet-5-5"}}, env={}
    )
    settings = load_settings({}, path)
    assert (settings.api_key, settings.model) == (KEY, "claude-sonnet-5-5")


def test_switching_to_a_local_model_keeps_the_key_for_later(path):
    llm_settings.save(path, CLOUD, env={})
    local = {"backend": "local", "local": {"model": "llama3.1:8b", "url": "http://localhost:11434/v1"}}
    shown = llm_settings.save(path, local, env={})
    assert shown["active"]["backend"] == "local" and shown["anthropic"]["key_set"] is True
    assert load_settings({"ASCLEPIUS_LLM_BACKEND": "anthropic"}, path).api_key == KEY


def test_the_key_can_be_forgotten(path):
    llm_settings.save(path, CLOUD, env={})
    local = {"backend": "local", "local": {"model": "llama3.1:8b"}, "forget_key": True}
    assert llm_settings.save(path, local, env={})["anthropic"]["key_set"] is False
    assert KEY not in path.read_text()


def test_settings_the_engine_would_refuse_are_not_saved(path):
    llm_settings.save(path, CLOUD, env={})
    before = path.read_text()
    with pytest.raises(SettingsError, match="no model is named"):
        llm_settings.save(path, {"backend": "local", "local": {"model": ""}}, env={})
    with pytest.raises(SettingsError, match="effort"):
        llm_settings.save(path, {"backend": "anthropic", "anthropic": {"effort": "huge"}}, env={})
    with pytest.raises(SettingsError, match="http"):
        llm_settings.save(path, {"backend": "local", "local": {"model": "m", "url": "file:///etc"}}, env={})
    assert path.read_text() == before


def test_quotes_and_new_lines_in_a_value_cannot_add_settings(path):
    sneaky = 'm"\nbackend = "anthropic'
    llm_settings.save(path, {"backend": "local", "local": {"model": sneaky}}, env={})
    assert load_settings({}, path).model == sneaky


def test_nothing_set_up_yet_says_how(path):
    shown = llm_settings.read(path, env={})
    assert shown["active"] is None and "No LLM is configured" in shown["problem"]


def test_environment_variables_that_override_the_file_are_named(path):
    llm_settings.save(path, CLOUD, env={})
    shown = llm_settings.read(path, env={"ASCLEPIUS_LLM_MODEL": "other", "ANTHROPIC_API_KEY": KEY})
    assert shown["overridden_by"] == ["ASCLEPIUS_LLM_MODEL", "ANTHROPIC_API_KEY"]
    assert shown["active"]["model"] == "other" and KEY not in json.dumps(shown)


def answering(url, body):
    return {"model": "llama3.1:8b-q4", "choices": [{"message": {"content": "ready"}}]}


def test_the_check_says_which_model_answered(path):
    llm_settings.save(path, {"backend": "local", "local": {"model": "llama3.1:8b"}}, env={})
    said = llm_settings.check(path, env={}, connect_llm=partial(connect, local_transport=answering))
    assert said == {"backend": "local", "model": "llama3.1:8b-q4", "answer": "ready"}


def test_a_check_that_reaches_no_model_says_why(path):
    llm_settings.save(path, {"backend": "local", "local": {"model": "llama3.1:8b"}}, env={})
    with pytest.raises(SettingsError, match="no local model server answered"):
        llm_settings.check(path, env={}, connect_llm=partial(connect, local_transport=_down))


def _down(url, body):
    from hypothesis_engine.llm import LLMError

    raise LLMError(f"no local model server answered at {url}")


def test_a_settings_file_that_cannot_be_read_is_not_overwritten(path):
    path.write_text("max_tokens = 4000\n[local\nmodel = 'x'\n")
    before = path.read_text()
    assert "not valid TOML" in llm_settings.read(path, env={})["problem"]
    with pytest.raises(SettingsError, match="not valid TOML"):
        llm_settings.save(path, {"backend": "local", "local": {"model": "m"}}, env={})
    assert path.read_text() == before
