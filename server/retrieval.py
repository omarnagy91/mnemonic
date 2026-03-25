import os
"""
Mnemonic v4 — MultiStrategyRetrieval with Context Tree Architecture
Combines vector similarity search with hierarchical memory organization (L0/L1/L2),
applies weighted scoring, and provides context assembly for better retrieval.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from openai import OpenAI

from categorizer import categorize_text, estimate_importance, compute_weighted_score
from models import MemoryCategory, CategorySummary, ContextTreeResponse

logger = logging.getLogger("mnemonic.retrieval")


_REFLECT_SYSTEM = (
    "You are a personal knowledge assistant. "
    "Synthesize the provided memories into a direct, specific answer. "
    "Connect information across memories. Cite names, dates, and numbers exactly as they appear."
)

_REFLECT_USER = """Query: {query}

Relevant memories ({count} total):
{memories_text}

Synthesize a comprehensive answer to the query based on these memories. \
Be specific. If the memories do not contain enough information to answer fully, say so."""

_CATEGORY_SUMMARY_SYSTEM = (
    "You are a knowledge organizer. Create concise summaries of memory categories. "
    "Be specific about key facts, decisions, and relationships mentioned."
)

_L0_SUMMARY_PROMPT = """Memories in the {category} category:
{memories}

Create a brief L0 summary (max 50 tokens) covering the key themes and entities in this category."""

_L1_SUMMARY_PROMPT = """Memories in the {category} category:
{memories}

Create a detailed L1 summary (max 200 tokens) covering:
- Key facts and decisions
- Important people and relationships
- Current status and trends
- Notable changes or developments"""

_CONTEXT_ASSEMBLY_SYSTEM = (
    "You are a context assembler. Given a query and hierarchical memory summaries, "
    "create a coherent context that answers the query using the provided information. "
    "Start with high-level insights from category summaries, then drill into specific details."
)

_CONTEXT_ASSEMBLY_PROMPT = """Query: {query}

Category summaries:
{category_summaries}

Specific relevant memories:
{specific_memories}

Assemble a comprehensive context that answers the query by:
1. Starting with relevant high-level insights
2. Adding specific details from memories
3. Connecting related information across categories
4. Being concise but complete"""


class MultiStrategyRetrieval:
    """
    Enhanced retrieval that merges vector search results with context tree architecture,
    providing hierarchical memory organization and intelligent context assembly.
    """

    def __init__(self, memory_instance: Any, openai_client: OpenAI, model: str = "gpt-4.1-mini"):
        self.memory = memory_instance
        self.client = openai_client
        self.model = model
        self._category_summaries_cache: Dict[str, CategorySummary] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Multi-strategy retrieval with enhanced scoring and categorization.
        """
        merged: Dict[str, Dict[str, Any]] = {}

        # 1. Vector search via mem0
        try:
            kwargs: dict = {"query": query, "user_id": user_id, "limit": limit}
            if agent_id:
                kwargs["agent_id"] = agent_id
            raw = self.memory.search(**kwargs)
            items = raw.get("results", raw) if isinstance(raw, dict) else raw
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        merged[item["id"]] = item
        except Exception as e:
            logger.warning(f"Vector search error: {e}")

        # 2. Score and sort with enhanced categorization
        scored: List[Dict[str, Any]] = []
        for mem in merged.values():
            meta = (mem.get("metadata") or {})
            text = mem.get("memory", "")
            similarity = float(mem.get("score") or 0.5)

            importance = meta.get("importance", estimate_importance(text))
            try:
                importance = max(1, min(10, int(importance)))
            except (ValueError, TypeError):
                importance = 5

            confidence = float(meta.get("confidence") or 1.0)
            created_at = mem.get("created_at", "")
            access_count = int(meta.get("access_count") or 0)

            weighted = compute_weighted_score(
                similarity=similarity,
                importance=importance,
                created_at=created_at,
                access_count=access_count,
            ) * confidence

            mem["weighted_score"] = round(weighted, 4)

            # Ensure categorization
            if not meta.get("category"):
                meta["category"] = categorize_text(text).value
                mem["metadata"] = meta

            scored.append(mem)

        scored.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        return scored[:limit]

    def context_tree_search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        max_depth: int = 2,
        include_summaries: bool = True,
    ) -> ContextTreeResponse:
        """
        Hierarchical context assembly using the context tree architecture.
        
        L0: Category summaries (always loaded)
        L1: Detailed category summaries (loaded for relevant categories)
        L2: Individual memories (loaded for specific details)
        """
        start_time = datetime.now()
        
        # Step 1: Get all memories and categorize them
        all_memories = self.search(query=query, user_id=user_id, agent_id=agent_id, limit=50)
        
        # Step 2: Group by category and identify relevant categories
        category_groups = {}
        relevant_categories = []
        
        for mem in all_memories:
            category = mem.get("metadata", {}).get("category", "uncategorized")
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(mem)
            
            # Consider category relevant if it has high-scoring memories
            if mem.get("weighted_score", 0) > 0.5 and category not in relevant_categories:
                try:
                    relevant_categories.append(MemoryCategory(category))
                except ValueError:
                    relevant_categories.append(MemoryCategory.uncategorized)
        
        # Step 3: Generate/retrieve category summaries
        category_summaries = []
        if include_summaries:
            for category in relevant_categories:
                summary = self._get_or_generate_category_summary(
                    category, category_groups.get(category.value, []), user_id, max_depth
                )
                if summary:
                    category_summaries.append(summary)
        
        # Step 4: Assemble context hierarchically
        context = self._assemble_hierarchical_context(
            query, category_summaries, all_memories[:10], max_depth
        )
        
        # Step 5: Calculate token estimate
        context_text = context + "\n".join([s.l0_summary for s in category_summaries])
        total_tokens = len(context_text.split()) * 1.3  # Rough token estimate
        
        return ContextTreeResponse(
            query=query,
            relevant_categories=relevant_categories,
            category_summaries=category_summaries,
            context=context,
            memories_used=[mem for mem in all_memories[:10]],
            total_tokens=int(total_tokens)
        )

    def reflect(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str],
        client: OpenAI,
        model: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Retrieve top memories then ask LLM to synthesize a comprehensive answer.
        Returns {"answer": str, "sources": [memory_id, ...]}
        """
        memories = self.search(query=query, user_id=user_id, agent_id=agent_id, limit=limit)

        if not memories:
            return {"answer": "No relevant memories found.", "sources": []}

        lines = []
        source_ids = []
        for i, mem in enumerate(memories, 1):
            text = mem.get("memory", "")
            meta = mem.get("metadata") or {}
            cat = meta.get("category", "")
            date_str = (mem.get("created_at") or "")[:10]
            suffix = f" ({date_str})" if date_str else ""
            prefix = f"[{cat}] " if cat else ""
            lines.append(f"{i}. {prefix}{text}{suffix}")
            source_ids.append(mem.get("id", ""))

        memories_text = "\n".join(lines)
        user_prompt = _REFLECT_USER.format(
            query=query,
            count=len(memories),
            memories_text=memories_text,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REFLECT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Reflect LLM call failed: {e}")
            answer = f"Synthesis failed: {e}"

        return {"answer": answer, "sources": source_ids}

    def get_graph_data(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
        max_nodes: int = 100,
        category: Optional[MemoryCategory] = None,
    ) -> Dict[str, Any]:
        """
        Generate graph visualization data using Qdrant similarity search.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qclient = QdrantClient(host="localhost", port=6333)
        collection = os.environ.get("MEM0_COLLECTION", "openclaw_memories_v4")

        must_conds = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]

        all_points = []
        offset = None
        while len(all_points) < max_nodes:
            result = qclient.scroll(
                collection_name=collection,
                limit=min(100, max_nodes - len(all_points)),
                offset=offset, with_vectors=True, with_payload=True,
                scroll_filter=Filter(must=must_conds),
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None or not points:
                break
            offset = next_offset

        memories = []
        point_vectors = {}
        for p in all_points:
            pid = str(p.id)
            meta = p.payload.get("metadata", {})
            data = p.payload.get("data", "")
            cat = meta.get("category", "uncategorized")
            if category and cat != category.value:
                continue
            point_vectors[pid] = p.vector
            memories.append({"id": pid, "memory": data, "metadata": meta,
                           "created_at": p.payload.get("created_at", ""),
                           "category": cat})
        
        if len(memories) < 2:
            return {"nodes": [], "edges": [], "categories": [], "total_memories": 0}

        nodes = []
        categories_set = set()
        for mem in memories:
            cat = mem["category"]
            importance = mem["metadata"].get("importance", 5)
            categories_set.add(cat)
            nodes.append({
                "id": mem["id"],
                "memory": mem["memory"][:200],
                "category": cat,
                "importance": importance,
                "created_at": mem["created_at"],
                "label": mem["memory"][:50],
                "value": importance,
                "color": self._get_category_color(cat),
            })

        # Real similarity edges via Qdrant
        edges = []
        seen_pairs = set()
        for node in nodes[:50]:  # limit neighbor searches
            vec = point_vectors.get(node["id"])
            if vec is None:
                continue
            try:
                similar = qclient.query_points(
                    collection_name=collection, query=vec, limit=6,
                    query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
                    with_payload=False,
                ).points
                for s in similar:
                    sid = str(s.id)
                    if sid == node["id"] or s.score < similarity_threshold:
                        continue
                    pair = tuple(sorted([node["id"], sid]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    edges.append({"from": node["id"], "to": sid,
                                 "similarity": round(s.score, 3),
                                 "weight": max(1, int(s.score * 5))})
            except Exception as e:
                logger.warning(f"Similarity error: {e}")

        return {
            "nodes": nodes, "edges": edges,
            "categories": sorted(list(categories_set)),
            "total_memories": len(nodes),
        }

    # ── Internal Methods ───────────────────────────────────────────────────────

    def _get_or_generate_category_summary(
        self,
        category: MemoryCategory,
        memories: List[Dict[str, Any]],
        user_id: str,
        max_depth: int
    ) -> Optional[CategorySummary]:
        """Generate or retrieve cached category summary."""
        if not memories:
            return None
            
        cache_key = f"{user_id}_{category.value}"
        
        # Check cache (in production, this would be persistent)
        if cache_key in self._category_summaries_cache:
            cached = self._category_summaries_cache[cache_key]
            # Simple cache invalidation - refresh if older than 1 hour
            try:
                last_updated = datetime.fromisoformat(cached.last_updated.replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - last_updated).total_seconds() < 3600:
                    return cached
            except (AttributeError, ValueError):
                pass
        
        # Generate new summary
        try:
            memories_text = "\n".join([
                f"- {mem.get('memory', '')}"
                for mem in memories[:20]  # Limit for token efficiency
            ])
            
            # Generate L0 summary
            l0_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CATEGORY_SUMMARY_SYSTEM},
                    {"role": "user", "content": _L0_SUMMARY_PROMPT.format(
                        category=category.value, memories=memories_text
                    )}
                ],
                temperature=0.3,
                max_tokens=60
            )
            l0_summary = l0_response.choices[0].message.content or ""
            
            # Generate L1 summary if max_depth allows
            l1_summary = None
            if max_depth >= 1:
                l1_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _CATEGORY_SUMMARY_SYSTEM},
                        {"role": "user", "content": _L1_SUMMARY_PROMPT.format(
                            category=category.value, memories=memories_text
                        )}
                    ],
                    temperature=0.3,
                    max_tokens=250
                )
                l1_summary = l1_response.choices[0].message.content or ""
            
            # Calculate statistics
            importance_scores = [
                mem.get("metadata", {}).get("importance", 5)
                for mem in memories
            ]
            importance_avg = sum(importance_scores) / len(importance_scores) if importance_scores else 5.0
            
            summary = CategorySummary(
                category=category,
                l0_summary=l0_summary,
                l1_summary=l1_summary,
                memory_count=len(memories),
                last_updated=datetime.now(timezone.utc).isoformat(),
                importance_avg=importance_avg
            )
            
            # Cache it
            self._category_summaries_cache[cache_key] = summary
            return summary
            
        except Exception as e:
            logger.error(f"Category summary generation failed for {category.value}: {e}")
            return None

    def _assemble_hierarchical_context(
        self,
        query: str,
        category_summaries: List[CategorySummary],
        specific_memories: List[Dict[str, Any]],
        max_depth: int
    ) -> str:
        """Assemble hierarchical context from summaries and specific memories."""
        try:
            # Prepare category summaries text
            summaries_text = "\n".join([
                f"**{s.category.value}** (L0): {s.l0_summary}" +
                (f"\n  (L1): {s.l1_summary}" if s.l1_summary and max_depth >= 1 else "")
                for s in category_summaries
            ])
            
            # Prepare specific memories text
            memories_text = "\n".join([
                f"- [{mem.get('metadata', {}).get('category', 'uncategorized')}] {mem.get('memory', '')}"
                for mem in specific_memories[:8]  # Limit for context size
            ])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CONTEXT_ASSEMBLY_SYSTEM},
                    {"role": "user", "content": _CONTEXT_ASSEMBLY_PROMPT.format(
                        query=query,
                        category_summaries=summaries_text,
                        specific_memories=memories_text
                    )}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content or "Context assembly failed."
            
        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            return f"Error assembling context: {e}"

    def _get_category_color(self, category: str) -> str:
        """Return a color code for graph visualization."""
        colors = {
            "personal": "#4CAF50",
            "business": "#2196F3", 
            "technical": "#FF9800",
            "decision": "#9C27B0",
            "relationship": "#E91E63",
            "temporal": "#607D8B",
            "uncategorized": "#795548"
        }
        return colors.get(category, "#CCCCCC")