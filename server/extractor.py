"""
Mnemonic v4 — SmartExtractor
Enhanced fact extraction with comprehensive conversation analysis for compaction,
entity/relationship support, pre/post filters, temporal awareness, confidence scoring.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Optional

from openai import OpenAI

from models import (
    ExtractedFact, ExtractedEntity, ExtractedRelationship,
    ExtractionResult, MemoryCategory,
)

logger = logging.getLogger("mnemonic.extractor")


# ─── Pre-filter config ────────────────────────────────────────────────────────

# Regex patterns to strip entire blocks before LLM
_BLOCK_PATTERNS = [
    re.compile(r'<mem0-context>.*?</mem0-context>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<[a-z_-]{3,}>.*?</[a-z_-]{3,}>', re.DOTALL | re.IGNORECASE),
]

# Line prefixes that signal noise content (case-insensitive match on stripped line)
_NOISE_PREFIXES = (
    "i am ", "my role is", "i will ", "this agent", "the agent",
    "the assistant", "as an ai", "as your assistant",
    "● ", "active (running)", "inactive (dead)",
    "loaded:", "main pid:", "tasks:", "cgroup:",
    "systemd[", "dockerd", "containerd",
)

# Post-filter: reject facts whose lowercased text contains any of these substrings
_POST_FILTER_SUBSTRINGS = [
    "assistant", "agent is", "agent can", "agent will",
    "generates ", "provides ", "this session",
    "current date is", "today is ", "i can help",
    "language model", "the ai", "my purpose is",
    "i'm an ai", "i am an ai",
]


# ─── Prompts ──────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a Personal Memory Organizer. Extract ONLY facts about the USER from the conversation below.

CRITICAL RULES — DO NOT EXTRACT:
- Facts about the AI assistant: "Assistant is...", "I will help...", "I can..."
- Agent capabilities or behaviors: "Agent generates...", "This system provides..."
- Current date/time or session metadata: "Current date is...", "This session..."
- Infrastructure or service status: systemd output, docker status, ports, health checks
- Generic behavioral instructions or system prompts
- Any statement where the subject is the AI, model, or assistant

DO EXTRACT:
- Personal details about the user (name, location, family, preferences, health, hobbies)
- Business details (companies, clients, revenue, decisions, strategies)
- Professional details (job, role, colleagues, career moves, projects)
- Relationships (people they know — capture name, role, company, contact info if mentioned)
- Technical work (projects, tools, configs the user is personally working on)
- Plans and goals (upcoming events, deadlines, intentions, commitments)
- Experiences and outcomes (what happened, lessons learned, decisions made)
- Temporal events (scheduled things, past events with dates)

For each fact:
- "text": concise factual statement about the user (not the AI)
- "category": one of "personal"|"business"|"technical"|"decision"|"relationship"|"temporal"|"uncategorized"
- "importance": 1-10 scale:
    9-10 = core identity, non-decaying (name, nationality, core values)
    7-8  = major decisions, key relationships, business milestones
    5-6  = events, meetings, medium-term plans
    3-4  = observations, transient preferences
    1-2  = trivial or ephemeral
- "confidence": 0.95 for direct explicit statements, 0.70 for inferences

For each entity (person / company / project) explicitly mentioned:
- "name": entity name
- "type": "person" | "company" | "project"
- "attributes": dict — e.g. {{"role": "CTO", "company": "Acme", "email": "...", "phone": "...", "relationship_to_user": "client"}}

For each relationship between two entities:
- "from": source entity name
- "to": target entity name
- "type": relationship type — e.g. "works_at", "reports_to", "owns", "founded", "partnered_with", "client_of", "investor_in"
- "context": one sentence of context

Return ONLY this JSON (no prose, no markdown):
{{
  "facts": [{{"text": "...", "category": "...", "importance": 7, "confidence": 0.95}}],
  "entities": [{{"name": "...", "type": "person", "attributes": {{}}}}],
  "relationships": [{{"from": "...", "to": "...", "type": "...", "context": "..."}}]
}}

If nothing user-relevant is present, return: {{"facts": [], "entities": [], "relationships": []}}

Today is {date}. Extract from USER messages only — ignore assistant/system messages.
"""

DIGEST_PROMPT = """You are a memory extraction system for cron job / automated agent outputs.
Extract ONLY facts relevant to the user, their business, contacts, or notable real-world events.

IGNORE completely:
- Agent capabilities, process descriptions, tool usage
- System/service health: systemd, docker, ports, infrastructure status
- Statements about what the agent does or provides

For each fact: {{"text": "...", "category": "...", "importance": N, "confidence": 0.9}}
Categories: personal, business, technical, decision, relationship, temporal
Importance: 1-10 (same scale as above)
Confidence: 0.95 for direct facts, 0.70 for inferences

Return JSON: {{"facts": [...], "entities": [], "relationships": []}}
If nothing user-relevant, return: {{"facts": [], "entities": [], "relationships": []}}

Today is {date}.
"""


# ─── Filter functions ─────────────────────────────────────────────────────────

def pre_filter(text: str) -> str:
    """Strip XML blocks and noisy lines before sending to LLM."""
    for pattern in _BLOCK_PATTERNS:
        text = pattern.sub("", text)

    filtered_lines = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if any(stripped.startswith(prefix) for prefix in _NOISE_PREFIXES):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def post_filter_facts(facts: List[ExtractedFact]) -> List[ExtractedFact]:
    """Reject agent-description facts and facts that are too short."""
    kept = []
    for fact in facts:
        if len(fact.text.strip()) < 15:
            logger.debug(f"Post-filter: too short — {fact.text!r}")
            continue
        text_lower = fact.text.lower()
        if any(sub in text_lower for sub in _POST_FILTER_SUBSTRINGS):
            logger.debug(f"Post-filter: noise pattern — {fact.text[:60]!r}")
            continue
        kept.append(fact)
    return kept


# ─── SmartExtractor ───────────────────────────────────────────────────────────

class SmartExtractor:
    """
    v3 fact extraction: entity/relationship extraction, pre/post filters,
    confidence scoring, temporal awareness.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("MEM0_EXTRACTOR_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=self.api_key)

    def _call_llm(self, system_prompt: str, user_content: str) -> dict:
        """Call OpenAI with json_object response format."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from {self.model}: {e}")
            return {"facts": [], "entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"SmartExtractor LLM call failed: {e}")
            return {"facts": [], "entities": [], "relationships": []}

    def extract_from_messages(self, messages: List[dict]) -> ExtractionResult:
        """Extract facts + entities + relationships from a conversation thread."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = pre_filter(msg.get("content", ""))
            if content:
                lines.append(f"{role.capitalize()}: {content}")
        conversation = "\n".join(lines)

        if len(conversation.strip()) < 50:
            return ExtractionResult()

        prompt = EXTRACTION_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        raw = self._call_llm(prompt, conversation)
        return self._parse_result(raw)

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract from plain text (imports, ingested files)."""
        filtered = pre_filter(text)
        if len(filtered.strip()) < 50:
            return ExtractionResult()

        prompt = EXTRACTION_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        raw = self._call_llm(prompt, filtered)
        return self._parse_result(raw)

    def extract_digest(self, content: str, source: str = "") -> ExtractionResult:
        """Extract user-relevant facts from cron/agent output."""
        filtered = pre_filter(content)
        if len(filtered.strip()) < 50:
            return ExtractionResult()

        prompt = DIGEST_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        user_content = f"Source: {source}\n\n{filtered}"
        raw = self._call_llm(prompt, user_content)
        return self._parse_result(raw)

    def _parse_result(self, result: dict) -> ExtractionResult:
        """Normalize LLM output into ExtractionResult, applying post-filters."""
        facts: List[ExtractedFact] = []
        raw_facts: List[str] = []
        entities: List[ExtractedEntity] = []
        relationships: List[ExtractedRelationship] = []

        # ── Facts ──────────────────────────────────────────────────────────────
        for item in (result.get("facts") or []):
            if isinstance(item, str):
                raw_facts.append(item)
                facts.append(ExtractedFact(text=item))
                continue
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            raw_facts.append(text)

            cat_str = item.get("category", "uncategorized")
            try:
                category = MemoryCategory(cat_str)
            except ValueError:
                category = MemoryCategory.uncategorized

            importance = item.get("importance", 5)
            try:
                importance = max(1, min(10, int(importance)))
            except (ValueError, TypeError):
                importance = 5

            confidence = item.get("confidence", 0.95)
            try:
                confidence = max(0.1, min(1.0, float(confidence)))
            except (ValueError, TypeError):
                confidence = 0.95

            facts.append(ExtractedFact(
                text=text,
                category=category,
                importance=importance,
                confidence=confidence,
            ))

        facts = post_filter_facts(facts)

        # ── Entities ───────────────────────────────────────────────────────────
        for item in (result.get("entities") or []):
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if not name:
                continue
            entities.append(ExtractedEntity(
                name=name,
                type=item.get("type", "person"),
                attributes=item.get("attributes") or {},
            ))

        # ── Relationships ──────────────────────────────────────────────────────
        for item in (result.get("relationships") or []):
            if not isinstance(item, dict):
                continue
            from_e = (item.get("from") or "").strip()
            to_e = (item.get("to") or "").strip()
            if not from_e or not to_e:
                continue
            relationships.append(ExtractedRelationship(
                from_entity=from_e,
                to_entity=to_e,
                type=item.get("type", "related_to"),
                context=item.get("context", ""),
            ))

        return ExtractionResult(
            facts=facts,
            entities=entities,
            relationships=relationships,
            raw_facts=raw_facts,
        )

    def extract_comprehensive(
        self,
        conversation_text: str,
        extract_decisions: bool = True,
        extract_facts: bool = True,
        extract_preferences: bool = True,
        extract_actions: bool = True,
        extract_temporal: bool = True,
    ) -> ExtractionResult:
        """
        NEW v4: Comprehensive extraction for compaction workflow.
        Analyzes full conversations to extract different types of information.
        """
        if not self.client:
            return ExtractionResult()
        
        # Prepare specialized prompts for different extraction types
        extraction_prompts = []
        
        if extract_facts:
            extraction_prompts.append("- Facts and information learned")
        if extract_decisions:
            extraction_prompts.append("- Decisions made or conclusions reached")
        if extract_preferences:
            extraction_prompts.append("- User preferences or opinions expressed")
        if extract_actions:
            extraction_prompts.append("- Action items, tasks, or plans mentioned")
        if extract_temporal:
            extraction_prompts.append("- Time-sensitive events or deadlines")
        
        extraction_types = "\n".join(extraction_prompts)
        
        system_prompt = f"""You are a comprehensive information extractor. Analyze the conversation and extract:
{extraction_types}

For each extracted item, determine:
- text: The specific fact/decision/preference (concise but complete)
- category: One of: personal, business, technical, decision, relationship, temporal, uncategorized
- importance: 1-10 (10 = critical, 5 = normal, 1 = minor)
- confidence: 0.1-1.0 (confidence in extraction accuracy)
- type: fact, decision, preference, action, temporal (for classification)

Focus on extracting valuable, actionable information. Ignore meta-commentary about the conversation itself.

Return JSON: {{"items": [{{"text": "...", "category": "...", "importance": 5, "confidence": 0.95, "type": "fact"}}]}}"""

        user_prompt = f"""Conversation to analyze:
{conversation_text[:8000]}  # Limit context size

Extract all valuable information according to the criteria above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse JSON response
            try:
                result = json.loads(content)
                items = result.get("items", [])
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        items = result.get("items", [])
                    except json.JSONDecodeError:
                        items = []
                else:
                    items = []
            
            # Convert to ExtractedFact objects
            facts = []
            raw_facts = []
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                text = (item.get("text") or "").strip()
                if not text or len(text) < 10:
                    continue
                
                # Apply post-filtering
                if any(substring in text.lower() for substring in _POST_FILTER_SUBSTRINGS):
                    continue
                
                raw_facts.append(text)
                
                # Parse category
                cat_str = item.get("category", "uncategorized")
                try:
                    category = MemoryCategory(cat_str)
                except ValueError:
                    category = MemoryCategory.uncategorized
                
                # Parse importance
                importance = item.get("importance", 5)
                try:
                    importance = max(1, min(10, int(importance)))
                except (ValueError, TypeError):
                    importance = 5
                
                # Parse confidence
                confidence = item.get("confidence", 0.95)
                try:
                    confidence = max(0.1, min(1.0, float(confidence)))
                except (ValueError, TypeError):
                    confidence = 0.95
                
                facts.append(ExtractedFact(
                    text=text,
                    category=category,
                    importance=importance,
                    confidence=confidence,
                ))
            
            # Apply final post-filtering
            facts = post_filter_facts(facts)
            
            return ExtractionResult(
                facts=facts,
                entities=[],  # Simplified for compaction workflow
                relationships=[],  # Simplified for compaction workflow
                raw_facts=raw_facts,
            )
            
        except Exception as e:
            logger.error(f"Comprehensive extraction failed: {e}")
            return ExtractionResult()
