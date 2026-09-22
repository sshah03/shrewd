"""The system prompt is shared across every document of a run; providers that discount
cached prefix reads are asked to cache it."""
from shrewd import teacher


def test_claude_system_prompt_is_marked_for_caching(monkeypatch):
    teacher._cacheable_models.clear()
    monkeypatch.setattr(
        "shrewd.teacher.litellm.get_model_info", lambda m: {"cache_read_input_token_cost": 2.5e-7}
    )
    out = teacher.mark_prefix_cache("anthropic/x", [
        {"role": "system", "content": "prompt"}, {"role": "user", "content": "doc"},
    ])
    assert out[0]["content"] == [
        {"type": "text", "text": "prompt", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1] == {"role": "user", "content": "doc"}


def test_unknown_or_undiscounted_models_are_left_alone(monkeypatch):
    teacher._cacheable_models.clear()
    messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "doc"}]

    def unknown(m):
        raise ValueError("no price data")

    monkeypatch.setattr("shrewd.teacher.litellm.get_model_info", unknown)
    assert teacher.mark_prefix_cache("test/fake", messages) is messages
    teacher._cacheable_models.clear()
    monkeypatch.setattr(
        "shrewd.teacher.litellm.get_model_info", lambda m: {"input_cost_per_token": 1e-6}
    )
    assert teacher.mark_prefix_cache("other/x", messages) is messages


def test_no_system_turn_means_no_marking(monkeypatch):
    teacher._cacheable_models.clear()
    monkeypatch.setattr(
        "shrewd.teacher.litellm.get_model_info", lambda m: {"cache_read_input_token_cost": 1e-7}
    )
    messages = [{"role": "user", "content": "just a question"}]
    assert teacher.mark_prefix_cache("anthropic/x", messages) is messages
