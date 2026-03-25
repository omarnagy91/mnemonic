"""
Mnemonic v3 — ContradictionDetector
Before storing a new fact, checks existing memories for contradictions and
resolves them via LLM (keep old, keep new, or merge).
"""

import json
import logging
from typing import Optional, Any

from openai import OpenAI

logger = logging.getLogger("mnemonic.contradiction")

_SYSTEM_PROMPT = (
    "You are a fact deduplication assistant. "
    "Determine if two memory facts about the same person contradict each other."
)

_USER_TEMPLATE = """Fact A (existing): {old}
Fact B (new): {new}

Do these two facts contradict each other?
Consider them contradictory only if they make conflicting claims about the same subject \
(e.g. different cities, different job titles, opposite preferences).
Non-conflicting elaborations or additions are NOT contradictions.

Return ONLY this JSON (no prose):
{{"contradicts": true|false, "keep": "old"|"new"|"merge", "merged_text": "..."}}

Rules:
- If contradicts=false, set keep="new" and merged_text="".
- If keep="merge", merged_text must be a single sentence combining the accurate parts of both facts.
- If keep="old" or keep="new", merged_text can be empty.
"""


class ContradictionDetector:
    """
    Checks new facts against existing memories and resolves contradictions.

    Usage::
        decision = detector.check_and_handle(new_fact_text, user_id, agent_id)
        # decision: {"action": "store"|"skip"|"merge", "merged_text": str|None}
    """

    def __init__(self, api_key: str, model: str, memory_instance: Any):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory = memory_instance

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_and_handle(
        self,
        new_fact_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> dict:
        """
        Search for high-similarity existing memories and check for contradictions.

        Returns a dict:
          {"action": "store"|"skip"|"merge", "merged_text": str|None}
        - "store"  → store new_fact_text as-is
        - "skip"   → existing fact is better; do not store new
        - "merge"  → delete old fact, store merged_text instead
        """
        try:
            search_kwargs = {"query": new_fact_text, "user_id": user_id, "limit": 3}
            if agent_id:
                search_kwargs["agent_id"] = agent_id

            similar = self.memory.search(**search_kwargs)
            items = similar.get("results", similar) if isinstance(similar, dict) else similar
            if not isinstance(items, list):
                return {"action": "store", "merged_text": None}

            for item in items:
                score = item.get("score", 0)
                if score < 0.85:
                    continue

                old_text = (item.get("memory") or "").strip()
                old_id = item.get("id", "")
                if not old_text or not old_id:
                    continue

                decision = self._llm_check(old_text, new_fact_text)

                if not decision.get("contradicts"):
                    continue

                keep = decision.get("keep", "new")
                merged = (decision.get("merged_text") or "").strip()

                if keep == "old":
                    logger.info(f"Contradiction resolved: keeping old [{old_id[:8]}], skipping new")
                    return {"action": "skip", "merged_text": None}

                # For "new" or "merge": delete the old memory first
                try:
                    self.memory.delete(old_id)
                    logger.info(f"Contradiction resolved: deleted old [{old_id[:8]}], keep={keep}")
                except Exception as e:
                    logger.warning(f"Failed to delete contradicted memory {old_id}: {e}")

                if keep == "merge" and merged:
                    return {"action": "merge", "merged_text": merged}

                return {"action": "store", "merged_text": None}

        except Exception as e:
            logger.error(f"ContradictionDetector error (non-fatal): {e}")

        return {"action": "store", "merged_text": None}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _llm_check(self, old_text: str, new_text: str) -> dict:
        """Ask the LLM whether two facts contradict each other."""
        prompt = _USER_TEMPLATE.format(old=old_text, new=new_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            logger.error(f"Contradiction LLM call failed: {e}")
            return {"contradicts": False}
