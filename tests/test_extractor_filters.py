"""Unit tests for the pre/post filters in server/extractor.py (no LLM calls)."""

from extractor import post_filter_facts, pre_filter
from models import ExtractedFact


# ── pre_filter ───────────────────────────────────────────────────────────────

def test_pre_filter_strips_tagged_blocks():
    text = "Keep this line.\n<mem0-context>injected recall block</mem0-context>\nAnd this one."
    out = pre_filter(text)
    assert "injected recall block" not in out
    assert "Keep this line." in out
    assert "And this one." in out


def test_pre_filter_drops_noise_prefixed_lines():
    text = "loaded: some unit file\nUser prefers morning meetings\nmain pid: 1234"
    out = pre_filter(text)
    assert out == "User prefers morning meetings"


def test_pre_filter_keeps_ordinary_text():
    text = "The quarterly numbers were reviewed on Tuesday."
    assert pre_filter(text) == text


# ── post_filter_facts ────────────────────────────────────────────────────────

def test_post_filter_drops_too_short_facts():
    facts = [ExtractedFact(text="too short"), ExtractedFact(text="User runs a bakery in Alexandria")]
    kept = post_filter_facts(facts)
    assert [f.text for f in kept] == ["User runs a bakery in Alexandria"]


def test_post_filter_drops_self_description_noise():
    facts = [
        ExtractedFact(text="The assistant can summarize long documents"),
        ExtractedFact(text="User moved the office to the new building"),
    ]
    kept = post_filter_facts(facts)
    assert [f.text for f in kept] == ["User moved the office to the new building"]
