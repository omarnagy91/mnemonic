"""Unit tests for server/categorizer.py (pure functions, no network)."""

from datetime import datetime, timedelta, timezone

from categorizer import (
    categorize_text,
    compute_weighted_score,
    estimate_importance,
    group_by_category,
)
from models import MemoryCategory


# ── categorize_text ──────────────────────────────────────────────────────────

def test_categorize_business_text():
    assert categorize_text("Signed a new client contract with monthly revenue") == MemoryCategory.business


def test_categorize_technical_text():
    assert categorize_text("Deployed the FastAPI server with Docker and fixed a config bug") == MemoryCategory.technical


def test_categorize_personal_text():
    assert categorize_text("My favorite hobby is photography and my sister paints") == MemoryCategory.personal


def test_categorize_unmatched_text_falls_back():
    assert categorize_text("Blue sky above the quiet field") == MemoryCategory.uncategorized


def test_categorize_picks_highest_scoring_category():
    # Two technical keywords vs one business keyword
    text = "The api server config changed after the client call"
    assert categorize_text(text) == MemoryCategory.technical


# ── estimate_importance ──────────────────────────────────────────────────────

def test_importance_identity_signal_boosts_to_nine():
    assert estimate_importance("My name is Lina and I was born in Alexandria") >= 9


def test_importance_short_text_is_lowered():
    long_text = "We will plan to migrate the database deployment next quarter"
    short_text = "fix bug"
    assert estimate_importance(short_text) < estimate_importance(long_text)


def test_importance_always_within_bounds():
    samples = ["", "x", "My name is Omar, born in Egypt, decided to commit always"]
    for s in samples:
        assert 1 <= estimate_importance(s) <= 10


# ── compute_weighted_score ───────────────────────────────────────────────────

def test_high_importance_skips_recency_decay():
    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    score = compute_weighted_score(similarity=1.0, importance=9, created_at=old_date)
    # importance 9 -> weight 0.9, recency factor must stay 1.0
    assert abs(score - 0.9) < 1e-9


def test_low_importance_decays_with_age():
    old_date = (datetime.now(timezone.utc) - timedelta(days=80)).isoformat()
    fresh_date = datetime.now(timezone.utc).isoformat()
    old_score = compute_weighted_score(similarity=1.0, importance=5, created_at=old_date)
    fresh_score = compute_weighted_score(similarity=1.0, importance=5, created_at=fresh_date)
    assert old_score < fresh_score


def test_recency_factor_has_floor():
    ancient = (datetime.now(timezone.utc) - timedelta(days=2000)).isoformat()
    score = compute_weighted_score(similarity=1.0, importance=5, created_at=ancient)
    # floor is 0.3 -> score >= 0.5 * 0.3
    assert score >= 0.15 - 1e-9


def test_access_boost_is_capped():
    base = compute_weighted_score(similarity=1.0, importance=10, access_count=0)
    boosted = compute_weighted_score(similarity=1.0, importance=10, access_count=1000)
    assert boosted <= base * 1.2 + 1e-9
    assert boosted > base


def test_unparseable_date_uses_fallback_factor():
    score = compute_weighted_score(similarity=1.0, importance=5, created_at="not-a-date")
    assert abs(score - 0.5 * 0.8) < 1e-9


# ── group_by_category ────────────────────────────────────────────────────────

def test_group_by_category_reads_top_level_and_metadata():
    memories = [
        {"category": "business", "memory": "a"},
        {"metadata": {"category": "business"}, "memory": "b"},
        {"memory": "c"},
    ]
    groups = group_by_category(memories)
    assert len(groups["business"]) == 2
    assert len(groups["uncategorized"]) == 1
