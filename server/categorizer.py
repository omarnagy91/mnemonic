"""
Mnemonic v2 — Memory Categorizer
Assigns categories and importance to memories, provides weighted scoring.
"""

import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from models import MemoryCategory


# ─── Category detection (lightweight, no LLM needed) ─────────────────────────

CATEGORY_KEYWORDS = {
    MemoryCategory.personal: [
        "name is", "lives in", "born", "age", "birthday", "favorite", "favourite",
        "likes", "dislikes", "hobby", "preference", "family", "wife", "husband",
        "brother", "sister", "mother", "father", "son", "daughter", "home",
        "personality", "diet", "health", "fitness", "religion", "language",
    ],
    MemoryCategory.business: [
        "company", "revenue", "client", "customer", "startup", "business",
        "pricing", "invoice", "contract", "partnership", "funding", "investor",
        "sales", "pipeline", "outreach", "roi", "profit", "loss", "market",
        "freelance", "consulting", "retainer",
    ],
    MemoryCategory.technical: [
        "api", "server", "docker", "database", "code", "deploy", "git",
        "python", "javascript", "typescript", "react", "next.js", "fastapi",
        "infrastructure", "aws", "qdrant", "nginx", "ssh", "port",
        "config", "env", "bug", "fix", "error", "debug", "install",
    ],
    MemoryCategory.decision: [
        "decided", "chose", "going to", "will", "plan to", "strategy",
        "pivot", "switch", "migrate", "stop", "start", "kill", "pause",
        "approve", "reject", "priority", "focus on",
    ],
    MemoryCategory.relationship: [
        "met with", "talked to", "contact", "email", "call", "meeting with",
        "works at", "introduced", "colleague", "friend", "mentor", "advisor",
        "team", "hire", "fired",
    ],
    MemoryCategory.temporal: [
        "deadline", "due", "scheduled", "appointment", "event", "tomorrow",
        "next week", "next month", "yesterday", "today", "date", "time",
        "calendar", "reminder", "expires",
    ],
}


def categorize_text(text: str) -> MemoryCategory:
    """Categorize a memory by keyword matching. Fast fallback when LLM isn't needed."""
    text_lower = text.lower()
    scores: Dict[MemoryCategory, int] = {}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return MemoryCategory.uncategorized

    return max(scores, key=scores.get)


def estimate_importance(text: str, category: Optional[MemoryCategory] = None) -> int:
    """Estimate importance 1-10 based on category and content signals."""
    if category is None:
        category = categorize_text(text)

    base = {
        MemoryCategory.personal: 8,
        MemoryCategory.business: 7,
        MemoryCategory.technical: 5,
        MemoryCategory.decision: 7,
        MemoryCategory.relationship: 6,
        MemoryCategory.temporal: 5,
        MemoryCategory.uncategorized: 4,
    }.get(category, 5)

    text_lower = text.lower()

    # Boost for identity signals
    if any(w in text_lower for w in ["name is", "born in", "i am", "my identity"]):
        base = max(base, 9)

    # Boost for strong decisions
    if any(w in text_lower for w in ["decided to", "will never", "always", "committed"]):
        base = min(base + 1, 10)

    # Lower for very short/generic facts
    if len(text) < 20:
        base = max(base - 1, 1)

    return max(1, min(10, base))


def compute_weighted_score(
    similarity: float,
    importance: int = 5,
    created_at: Optional[str] = None,
    access_count: int = 0,
) -> float:
    """
    Compute weighted score for search ranking.
    score = similarity * importance_weight * recency_factor * access_boost
    """
    importance_weight = importance / 10.0

    # Recency factor
    recency_factor = 1.0
    if importance < 8 and created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_old = (now - created).total_seconds() / 86400
            recency_factor = max(0.3, 1.0 - (days_old / 90.0))
        except (ValueError, TypeError):
            recency_factor = 0.8

    # Slight boost for frequently accessed memories
    access_boost = 1.0 + min(0.2, access_count * 0.02) if access_count > 0 else 1.0

    return similarity * importance_weight * recency_factor * access_boost


def group_by_category(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a list of memory dicts by their category field."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for mem in memories:
        cat = mem.get("category", mem.get("metadata", {}).get("category", "uncategorized"))
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(mem)
    return groups
