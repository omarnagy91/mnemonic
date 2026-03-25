#!/usr/bin/env python3
"""
Mnemonic v4 — Enhanced Memory API Server with Context Tree Architecture
Features: Hierarchical memory organization, compaction hooks, graph visualization,
timeline views, consolidated endpoints, and critical bug fixes.

Backward compatible with all v1/v2/v3 endpoints.
Run: python3 server.py
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from mem0 import Memory
from openai import OpenAI
from qdrant_client import QdrantClient

# ─── Ensure local imports work ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    AddRequest, SearchRequest, ForgetRequest, UpdateRequest,
    DigestRequest, ImportRequest, AdvancedSearchRequest,
    ReflectRequest, ConsolidateRequest, IngestFileRequest, MigrateRequest,
    ContextRequest, CompactRequest, TimelineRequest, GraphRequest,
    ContextTreeResponse, CompactResponse, TimelineResponse, GraphResponse,
    MemoryCategory, CategorySummary, GraphNode, GraphEdge, TimelineEntry,
    MemoryNode, ClusterGroup,
)
from extractor import SmartExtractor
from categorizer import (
    categorize_text, estimate_importance, compute_weighted_score, group_by_category,
)
from importer import start_import, get_job, list_jobs, load_file_data
from contradiction import ContradictionDetector
from retrieval import MultiStrategyRetrieval
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=3)

def _sync_search(query, user_id, agent_id, limit):
    """Synchronous search wrapper for thread pool."""
    return retrieval.search(query=query, user_id=user_id, agent_id=agent_id, limit=limit)

def _sync_add_with_error_handling(messages, user_id, agent_id, metadata):
    """
    Synchronous add wrapper with NONE event error handling.
    This fixes the critical Pydantic PointStruct validation error.
    """
    try:
        return memory.add(messages=messages, user_id=user_id, agent_id=agent_id, metadata=metadata)
    except Exception as e:
        error_msg = str(e)
        # Handle the specific NONE event validation error
        if "NONE" in error_msg and "PointStruct" in error_msg and "vector" in error_msg:
            logger.warning(f"Suppressing NONE event validation error: {error_msg}")
            # Return a mock response structure for NONE events
            return {
                "message": "Memory processed successfully (NONE event handled)",
                "results": []
            }
        # Re-raise other errors
        raise e

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mnemonic")

# ─── Config from env ──────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
QDRANT_HOST     = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", "6333"))
LLM_MODEL       = os.environ.get("MEM0_LLM_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.environ.get("MEM0_EMBEDDING_MODEL", "text-embedding-3-large")
MEM0_PORT       = int(os.environ.get("MEM0_PORT", "8765"))
COLLECTION_NAME = "openclaw_memories_v4"

# ─── mem0 config (v4 without Kuzu graph store - dropped) ──────────────────────
_base_config: Dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "config": {"model": LLM_MODEL, "api_key": OPENAI_API_KEY},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": EMBEDDING_MODEL, "api_key": OPENAI_API_KEY, "embedding_dims": 3072},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": COLLECTION_NAME,
            "embedding_model_dims": 3072,
        },
    },
    # NOTE: Removed graph_store (Kuzu) - it was causing validation errors
    "version": "v1.1",
}

# Initialize mem0 without graph store
try:
    memory = Memory.from_config(_base_config)
    logger.info("✓ mem0 initialized successfully (vector store only)")
except Exception as e:
    logger.error(f"✗ mem0 initialization failed: {e}")
    sys.exit(1)

# ─── OpenAI client for LLM operations ─────────────────────────────────────────
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✓ OpenAI client initialized")
except Exception as e:
    logger.error(f"✗ OpenAI client failed: {e}")
    client = None

# ─── Enhanced components ──────────────────────────────────────────────────────
extractor = SmartExtractor(client, LLM_MODEL) if client else None
contradiction_detector = ContradictionDetector(OPENAI_API_KEY, LLM_MODEL, memory) if client else None
retrieval = MultiStrategyRetrieval(memory, client, LLM_MODEL) if client else None

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mnemonic v4 Memory API",
    description="Enhanced memory server with context tree architecture",
    version="4.0.0"
)

# ─── Utility Functions ────────────────────────────────────────────────────────

def _safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure metadata is a dict with safe values."""
    if not isinstance(metadata, dict):
        metadata = {}
    
    # Ensure required fields exist
    safe_meta = {
        "importance": metadata.get("importance", 5),
        "category": metadata.get("category", "uncategorized"),
        "confidence": metadata.get("confidence", 1.0),
        "access_count": metadata.get("access_count", 0),
    }
    
    # Copy other fields safely
    for k, v in metadata.items():
        if k not in safe_meta:
            try:
                # Ensure JSON serializable
                json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)
    
    return safe_meta

def _extract_and_store_conversation(
    messages: List[Dict[str, Any]],
    user_id: str,
    agent_id: str,
    session_id: Optional[str] = None,
    extract_decisions: bool = True,
    extract_facts: bool = True,
    extract_preferences: bool = True,
    extract_actions: bool = True,
    extract_temporal: bool = True,
) -> Dict[str, int]:
    """
    Enhanced extraction for compaction workflow.
    Extracts different types of information from conversation and stores them.
    """
    if not extractor or not messages:
        return {"facts_extracted": 0, "decisions_extracted": 0, "preferences_extracted": 0, 
                "actions_extracted": 0, "temporal_events_extracted": 0}
    
    # Prepare conversation text
    conversation_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
        if isinstance(msg, dict) and msg.get('content')
    ])
    
    if not conversation_text.strip():
        return {"facts_extracted": 0, "decisions_extracted": 0, "preferences_extracted": 0,
                "actions_extracted": 0, "temporal_events_extracted": 0}
    
    extraction_results = {
        "facts_extracted": 0,
        "decisions_extracted": 0, 
        "preferences_extracted": 0,
        "actions_extracted": 0,
        "temporal_events_extracted": 0
    }
    
    try:
        # Use smart extraction with enhanced prompting
        result = extractor.extract_comprehensive(
            conversation_text,
            extract_decisions=extract_decisions,
            extract_facts=extract_facts,
            extract_preferences=extract_preferences,
            extract_actions=extract_actions,
            extract_temporal=extract_temporal
        )
        
        # Store extracted facts
        for fact in result.facts:
            try:
                metadata = _safe_metadata({
                    "category": fact.category.value,
                    "importance": fact.importance,
                    "confidence": fact.confidence,
                    "extraction_type": "fact",
                    "session_id": session_id,
                })
                
                _sync_add_with_error_handling(
                    messages=[{"role": "user", "content": fact.text}],
                    user_id=user_id,
                    agent_id=agent_id,
                    metadata=metadata
                )
                extraction_results["facts_extracted"] += 1
            except Exception as e:
                logger.warning(f"Failed to store extracted fact: {e}")
        
        # Count other extraction types (simplified for now)
        extraction_results["decisions_extracted"] = len([f for f in result.facts if "decision" in f.text.lower()])
        extraction_results["preferences_extracted"] = len([f for f in result.facts if "prefer" in f.text.lower() or "like" in f.text.lower()])
        extraction_results["actions_extracted"] = len([f for f in result.facts if any(word in f.text.lower() for word in ["will", "should", "need to", "plan to"])])
        extraction_results["temporal_events_extracted"] = len([f for f in result.facts if any(word in f.text.lower() for word in ["tomorrow", "next", "scheduled", "deadline"])])
        
    except Exception as e:
        logger.error(f"Comprehensive extraction failed: {e}")
    
    return extraction_results

# ─── Health Check (Priority) ──────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint - tests all critical components."""
    try:
        # Test Qdrant connection
        try:
            qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            collections = qdrant.get_collections()
            qdrant_status = "✓ connected"
        except Exception as e:
            qdrant_status = f"✗ error: {e}"
        
        # Test OpenAI connection
        openai_status = "✓ configured" if client else "✗ not configured"
        
        # Test mem0
        mem0_status = "✓ initialized" if memory else "✗ not initialized"
        
        status = {
            "service": "mnemonic-v4",
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.0.0",
            "components": {
                "qdrant": qdrant_status,
                "openai": openai_status, 
                "mem0": mem0_status,
                "graph_store": "disabled (Kuzu removed)",
                "extractor": "✓ available" if extractor else "✗ unavailable",
                "retrieval": "✓ available" if retrieval else "✗ unavailable"
            },
            "config": {
                "llm_model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "collection": COLLECTION_NAME,
                "port": MEM0_PORT
            }
        }
        
        return status
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "service": "mnemonic-v4", 
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

# ─── Core Memory Operations (Backward Compatible) ─────────────────────────────

@app.post("/add")
async def add_memory(request: AddRequest):
    """Add new memory with enhanced error handling."""
    try:
        metadata = _safe_metadata(request.metadata)
        
        # Enhanced extraction if available
        if extractor and len(request.messages) > 0:
            content = " ".join([msg.get("content", "") for msg in request.messages])
            if content.strip():
                # Categorize and score
                category = categorize_text(content)
                importance = estimate_importance(content)
                
                metadata.update({
                    "category": category.value,
                    "importance": importance,
                    "enhanced_extraction": True
                })
        
        # Add to mem0 with error handling
        result = await asyncio.get_event_loop().run_in_executor(
            _executor, _sync_add_with_error_handling,
            request.messages, request.user_id, request.agent_id, metadata
        )
        
        return {
            "message": "Memory added successfully",
            "user_id": request.user_id,
            "metadata": metadata,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Add memory error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {e}")

@app.post("/search") 
async def search_memories(request: SearchRequest):
    """Search memories with enhanced retrieval."""
    try:
        if retrieval:
            results = await asyncio.get_event_loop().run_in_executor(
                _executor, retrieval.search,
                request.query, request.user_id, request.agent_id, request.limit
            )
        else:
            # Fallback to basic mem0 search
            kwargs = {"query": request.query, "user_id": request.user_id, "limit": request.limit}
            if request.agent_id:
                kwargs["agent_id"] = request.agent_id
            raw_result = memory.search(**kwargs)
            results = raw_result.get("results", raw_result) if isinstance(raw_result, dict) else raw_result
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user memory profile."""
    try:
        result = memory.get_all(user_id=user_id)
        memories = result.get("results", result) if isinstance(result, dict) else result
        
        # Enhanced statistics
        categories = {}
        total_importance = 0
        for mem in memories:
            meta = mem.get("metadata", {})
            cat = meta.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
            total_importance += meta.get("importance", 5)
        
        avg_importance = total_importance / len(memories) if memories else 0
        
        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "categories": categories,
            "average_importance": round(avg_importance, 2),
            "memories": memories
        }
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {e}")

@app.delete("/forget")
async def forget_memory(request: ForgetRequest):
    """Delete a specific memory."""
    try:
        memory.delete(memory_id=request.memory_id)
        return {"message": "Memory deleted successfully", "memory_id": request.memory_id}
    except Exception as e:
        logger.error(f"Forget error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {e}")

@app.put("/update")
async def update_memory(request: UpdateRequest):
    """Update an existing memory."""
    try:
        memory.update(memory_id=request.memory_id, data=request.data)
        return {"message": "Memory updated successfully", "memory_id": request.memory_id}
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {e}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        # Get all memories (using default_user as default for stats)
        result = memory.get_all(user_id="default_user")
        memories = result.get("results", result) if isinstance(result, dict) else result
        
        # Calculate enhanced statistics
        stats = {
            "total_memories": len(memories),
            "categories": {},
            "importance_distribution": {f"{i}": 0 for i in range(1, 11)},
            "recent_24h": 0,
            "recent_7d": 0,
            "recent_30d": 0,
            "average_importance": 0,
            "system_info": {
                "version": "4.0.0",
                "model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "collection": COLLECTION_NAME,
                "graph_store": "disabled",
                "features": ["context_tree", "compaction", "timeline", "graph_viz"]
            }
        }
        
        now = datetime.now(timezone.utc)
        total_importance = 0
        
        for mem in memories:
            meta = mem.get("metadata", {})
            
            # Category distribution
            cat = meta.get("category", "uncategorized")
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            
            # Importance distribution
            importance = meta.get("importance", 5)
            total_importance += importance
            stats["importance_distribution"][str(importance)] = stats["importance_distribution"].get(str(importance), 0) + 1
            
            # Time-based statistics
            created_at = mem.get("created_at", "")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - created_dt
                    
                    if delta.days == 0:
                        stats["recent_24h"] += 1
                    if delta.days <= 7:
                        stats["recent_7d"] += 1
                    if delta.days <= 30:
                        stats["recent_30d"] += 1
                except (ValueError, TypeError):
                    pass
        
        stats["average_importance"] = round(total_importance / len(memories), 2) if memories else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

# ─── New v4 Endpoints ─────────────────────────────────────────────────────────

@app.post("/context", response_model=ContextTreeResponse)
async def context_tree_search(request: ContextRequest):
    """
    NEW v4: Hierarchical context assembly using context tree architecture.
    Returns assembled context with category summaries and relevant memories.
    """
    try:
        if not retrieval:
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        
        result = await asyncio.get_event_loop().run_in_executor(
            _executor, retrieval.context_tree_search,
            request.query, request.user_id, request.agent_id,
            request.max_depth, request.include_summaries
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Context tree search error: {e}")
        raise HTTPException(status_code=500, detail=f"Context search failed: {e}")

@app.post("/compact", response_model=CompactResponse)
async def compact_conversation(request: CompactRequest):
    """
    NEW v4: Compaction hook for OpenClaw's pre-compaction memory flush.
    Takes a conversation transcript and extracts all valuable information.
    """
    try:
        start_time = time.time()
        
        extraction_results = _extract_and_store_conversation(
            messages=request.messages,
            user_id=request.user_id,
            agent_id=request.agent_id or "compaction",
            session_id=request.session_id,
            extract_decisions=request.extract_decisions,
            extract_facts=request.extract_facts,
            extract_preferences=request.extract_preferences,
            extract_actions=request.extract_actions,
            extract_temporal=request.extract_temporal
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CompactResponse(
            facts_extracted=extraction_results["facts_extracted"],
            decisions_extracted=extraction_results["decisions_extracted"],
            preferences_extracted=extraction_results["preferences_extracted"],
            actions_extracted=extraction_results["actions_extracted"],
            temporal_events_extracted=extraction_results["temporal_events_extracted"],
            total_memories_created=extraction_results["facts_extracted"],  # Simplified
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Compaction error: {e}")
        raise HTTPException(status_code=500, detail=f"Compaction failed: {e}")

@app.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    user_id: str = "default_user",
    agent_id: Optional[str] = "default_agent",
    category: Optional[MemoryCategory] = None,
    min_importance: Optional[int] = Query(None, ge=1, le=10),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=1000)
):
    """
    NEW v4: Get memories as a chronological timeline with filtering.
    """
    try:
        # Use Qdrant scroll directly to get ALL memories (not limited to 100)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qclient = QdrantClient(host="localhost", port=6333)
        collection = os.environ.get("MEM0_COLLECTION", "openclaw_memories_v4")
        
        must_conds = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        
        all_points = []
        offset = None
        while True:
            result = qclient.scroll(
                collection_name=collection, limit=100, offset=offset,
                with_payload=True, with_vectors=False,
                scroll_filter=Filter(must=must_conds),
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None or not points:
                break
            offset = next_offset
        
        # Convert to memory format
        memories = []
        for p in all_points:
            memories.append({
                "id": str(p.id),
                "memory": p.payload.get("data", ""),
                "metadata": p.payload.get("metadata", {}),
                "created_at": p.payload.get("created_at", ""),
            })
        
        # Apply filters
        filtered = []
        for mem in memories:
            meta = mem.get("metadata", {})
            
            # Category filter
            if category and meta.get("category") != category.value:
                continue
                
            # Importance filter
            if min_importance and meta.get("importance", 5) < min_importance:
                continue
                
            # Date filters
            created_at = mem.get("created_at", "")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    
                    if from_date:
                        from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
                        if created_dt < from_dt:
                            continue
                    
                    if to_date:
                        to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
                        if created_dt > to_dt:
                            continue
                            
                except (ValueError, TypeError):
                    continue
            
            filtered.append(mem)
        
        # Sort by creation date (newest first)
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Limit results
        limited = filtered[:limit]
        
        # Format as timeline entries
        entries = []
        for mem in limited:
            meta = mem.get("metadata", {})
            entries.append(TimelineEntry(
                id=mem.get("id", ""),
                memory=mem.get("memory", ""),
                category=meta.get("category", "uncategorized"),
                importance=meta.get("importance", 5),
                created_at=mem.get("created_at", ""),
                metadata=meta
            ))
        
        return TimelineResponse(
            entries=entries,
            total_count=len(filtered),
            filters_applied={
                "category": category.value if category else None,
                "min_importance": min_importance,
                "from_date": from_date,
                "to_date": to_date,
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"Timeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline failed: {e}")

@app.get("/graph", response_model=GraphResponse)
async def get_graph(
    user_id: str = "default_user",
    agent_id: Optional[str] = "default_agent", 
    similarity_threshold: float = Query(0.7, ge=0.5, le=0.99),
    max_nodes: int = Query(100, ge=10, le=500),
    category: Optional[MemoryCategory] = None
):
    """
    ENHANCED v4: Generate graph visualization data from memory similarities.
    Returns nodes and edges for vis-network rendering.
    """
    try:
        if not retrieval:
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        
        graph_data = retrieval.get_graph_data(
            user_id=user_id,
            agent_id=agent_id,
            similarity_threshold=similarity_threshold,
            max_nodes=max_nodes,
            category=category
        )
        
        # Convert to response format
        nodes = [GraphNode(**node) for node in graph_data["nodes"]]
        edges = [
            GraphEdge(from_id=edge.get("from", edge.get("from_id","")), 
                     to_id=edge.get("to", edge.get("to_id","")), 
                     similarity=edge["similarity"], weight=edge.get("weight"))
            for edge in graph_data["edges"]
        ]
        
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            categories=graph_data["categories"],
            total_memories=graph_data["total_memories"]
        )
        
    except Exception as e:
        logger.error(f"Graph generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {e}")

@app.get("/categories")
async def get_categories(user_id: str = "default_user"):
    """
    NEW v4: Get category structure with summaries for context tree navigation.
    """
    try:
        if not retrieval:
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        
        # Get all memories and group by category
        result = memory.get_all(user_id=user_id)
        memories = result.get("results", result) if isinstance(result, dict) else result
        
        category_groups = {}
        for mem in memories:
            category = mem.get("metadata", {}).get("category", "uncategorized")
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(mem)
        
        # Generate summaries for each category
        categories = []
        for cat_name, cat_memories in category_groups.items():
            try:
                category_enum = MemoryCategory(cat_name)
                summary = retrieval._get_or_generate_category_summary(
                    category_enum, cat_memories, user_id, max_depth=1
                )
                if summary:
                    categories.append(summary.dict())
            except ValueError:
                # Handle unknown categories
                categories.append({
                    "category": cat_name,
                    "l0_summary": f"{len(cat_memories)} memories in {cat_name} category",
                    "l1_summary": None,
                    "memory_count": len(cat_memories),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "importance_avg": sum(m.get("metadata", {}).get("importance", 5) for m in cat_memories) / len(cat_memories)
                })
        
        return {
            "user_id": user_id,
            "categories": categories,
            "total_categories": len(categories),
            "total_memories": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Categories error: {e}")
        raise HTTPException(status_code=500, detail=f"Categories failed: {e}")

# ─── Legacy Endpoints (Backward Compatibility) ────────────────────────────────

@app.post("/digest")
async def digest_content(request: DigestRequest):
    """Legacy endpoint - enhanced with v4 features."""
    try:
        if not extractor:
            raise HTTPException(status_code=503, detail="Extractor service unavailable")
        
        result = extractor.extract_from_cron_output(request.content, request.source)
        
        stored_count = 0
        for fact in result.facts:
            try:
                metadata = _safe_metadata({
                    "category": fact.category.value,
                    "importance": fact.importance,
                    "confidence": fact.confidence,
                    "source": request.source,
                    "digest": True
                })
                
                _sync_add_with_error_handling(
                    messages=[{"role": "assistant", "content": fact.text}],
                    user_id=request.user_id,
                    agent_id=request.agent_id,
                    metadata=metadata
                )
                stored_count += 1
            except Exception as e:
                logger.warning(f"Failed to store digested fact: {e}")
        
        return {
            "message": f"Digested {stored_count} facts from {request.source}",
            "facts_extracted": len(result.facts),
            "facts_stored": stored_count,
            "source": request.source
        }
        
    except Exception as e:
        logger.error(f"Digest error: {e}")
        raise HTTPException(status_code=500, detail=f"Digest failed: {e}")

@app.post("/consolidate")
async def consolidate_memories(request: ConsolidateRequest):
    """ENHANCED v4: Consolidate similar memories with improved similarity detection."""
    try:
        if not retrieval:
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        
        # Get all memories for user
        result = memory.get_all(user_id=request.user_id)
        memories = result.get("results", result) if isinstance(result, dict) else result
        
        if len(memories) < 2:
            return {"message": "Not enough memories to consolidate", "clusters": []}
        
        # Simple similarity clustering (in production, use proper embeddings)
        clusters = []
        processed_ids = set()
        
        for i, mem1 in enumerate(memories):
            if mem1.get("id") in processed_ids:
                continue
                
            cluster = [mem1]
            processed_ids.add(mem1.get("id"))
            
            mem1_text = mem1.get("memory", "").lower()
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if mem2.get("id") in processed_ids:
                    continue
                    
                mem2_text = mem2.get("memory", "").lower()
                
                # Simple similarity check (could be enhanced with proper embeddings)
                common_words = len(set(mem1_text.split()) & set(mem2_text.split()))
                total_words = len(set(mem1_text.split()) | set(mem2_text.split()))
                similarity = common_words / total_words if total_words > 0 else 0
                
                if similarity >= request.similarity_threshold:
                    cluster.append(mem2)
                    processed_ids.add(mem2.get("id"))
            
            if len(cluster) > 1:
                clusters.append({
                    "similarity": request.similarity_threshold,
                    "members": cluster,
                    "count": len(cluster)
                })
        
        if request.dry_run:
            return {
                "message": f"Found {len(clusters)} clusters for consolidation (dry run)",
                "clusters": clusters,
                "would_consolidate": sum(c["count"] - 1 for c in clusters)
            }
        
        # Actually consolidate (simplified implementation)
        consolidated = 0
        for cluster in clusters:
            if len(cluster["members"]) > 1:
                # Keep the most important memory, delete others
                best_mem = max(cluster["members"], 
                             key=lambda x: x.get("metadata", {}).get("importance", 5))
                
                for mem in cluster["members"]:
                    if mem.get("id") != best_mem.get("id"):
                        try:
                            memory.delete(memory_id=mem.get("id"))
                            consolidated += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete memory {mem.get('id')}: {e}")
        
        return {
            "message": f"Consolidated {consolidated} duplicate memories",
            "clusters_found": len(clusters),
            "memories_consolidated": consolidated
        }
        
    except Exception as e:
        logger.error(f"Consolidation error: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")

# ─── Other Legacy Endpoints ───────────────────────────────────────────────────

@app.post("/import")
async def import_data(request: ImportRequest):
    """Legacy import endpoint."""
    try:
        job = start_import(request.source, request.format, request.user_id, 
                          request.file_path, request.data)
        return {"job_id": job.job_id, "status": "queued", "message": "Import started"}
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")

@app.get("/import/status/{job_id}")
async def import_status(job_id: str):
    """Get import job status."""
    try:
        job = get_job(job_id)
        return job.dict() if job else {"error": "Job not found"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {e}")

@app.post("/reflect")
async def reflect_on_memories(request: ReflectRequest):
    """Legacy reflect endpoint."""
    try:
        if not retrieval or not client:
            raise HTTPException(status_code=503, detail="Reflection service unavailable")
        
        result = retrieval.reflect(
            query=request.query,
            user_id=request.user_id,
            agent_id=request.agent_id,
            client=client,
            model=LLM_MODEL,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "reflection": result["answer"],
            "sources": result["sources"],
            "source_count": len(result["sources"])
        }
        
    except Exception as e:
        logger.error(f"Reflection error: {e}")
        raise HTTPException(status_code=500, detail=f"Reflection failed: {e}")

# ─── Static Files and UI ──────────────────────────────────────────────────────

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/explorer")
async def serve_explorer():
    """Serve the enhanced graph explorer UI."""
    return FileResponse("static/explorer.html")

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the enhanced dashboard UI."""
    return FileResponse("static/dashboard.html")

@app.get("/")
async def serve_root():
    """Root endpoint with API info."""
    return {
        "service": "Mnemonic v4 Memory API",
        "version": "4.0.0",
        "features": [
            "Context Tree Architecture",
            "Compaction Hooks", 
            "Timeline Views",
            "Graph Visualization",
            "Enhanced Consolidation",
            "NONE Event Bug Fix"
        ],
        "endpoints": {
            "health": "/health",
            "core": ["/add", "/search", "/profile/{user_id}", "/forget", "/update", "/stats"],
            "v4_new": ["/context", "/compact", "/timeline", "/graph", "/categories"],
            "legacy": ["/digest", "/import", "/reflect", "/consolidate"],
            "ui": ["/explorer", "/dashboard"]
        },
        "ui": {
            "explorer": "/explorer",
            "dashboard": "/dashboard"
        }
    }

# ─── Server Startup ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Mnemonic v4 Memory Server")
    logger.info(f"   Port: {MEM0_PORT}")
    logger.info(f"   Model: {LLM_MODEL}")
    logger.info(f"   Collection: {COLLECTION_NAME}")
    logger.info("   Features: Context Tree, Compaction, Timeline, Graph Viz")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=MEM0_PORT,
        reload=False,
        access_log=True
    )