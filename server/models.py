"""
Mnemonic v4 — Pydantic Models
All request/response models for the enhanced memory API with context tree architecture.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ─── Enums ────────────────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    personal = "personal"
    business = "business"
    technical = "technical"
    decision = "decision"
    relationship = "relationship"
    temporal = "temporal"
    uncategorized = "uncategorized"


# ─── Existing (backward-compatible) models ────────────────────────────────────

class AddRequest(BaseModel):
    messages: List[dict]
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    limit: int = 10


class ForgetRequest(BaseModel):
    memory_id: str


class UpdateRequest(BaseModel):
    memory_id: str
    data: str


# ─── New v2 models ───────────────────────────────────────────────────────────

class DigestRequest(BaseModel):
    source: str = Field(..., description="Source identifier, e.g. 'cron:morning-brief'")
    content: str = Field(..., description="Full text output from cron agent")
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"


class ImportRequest(BaseModel):
    source: str = Field(..., description="Source type: twitter, linkedin, text, json, csv")
    format: str = Field("auto", description="File format hint")
    user_id: str = "omar"
    file_path: Optional[str] = Field(None, description="Server-side file path to import")
    data: Optional[str] = Field(None, description="Inline data (for small payloads)")


class ImportStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    source: str
    total_items: int = 0
    processed_items: int = 0
    facts_extracted: int = 0
    errors: List[str] = []
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class AdvancedSearchRequest(BaseModel):
    query: str
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    limit: int = 10
    category: Optional[MemoryCategory] = None
    min_importance: Optional[int] = None
    max_importance: Optional[int] = None
    since: Optional[str] = Field(None, description="ISO datetime — only memories after this")
    until: Optional[str] = Field(None, description="ISO datetime — only memories before this")
    weighted: bool = Field(True, description="Apply importance/recency weighting to scores")


# ─── v3 extraction models ────────────────────────────────────────────────────

class ExtractedFact(BaseModel):
    text: str
    category: MemoryCategory = MemoryCategory.uncategorized
    importance: int = Field(5, ge=1, le=10)
    confidence: float = Field(0.95, ge=0.0, le=1.0)


class ExtractedEntity(BaseModel):
    name: str
    type: str = "person"  # person | company | project
    attributes: Dict[str, Any] = {}


class ExtractedRelationship(BaseModel):
    from_entity: str
    to_entity: str
    type: str = "related_to"
    context: str = ""


class ExtractionResult(BaseModel):
    facts: List[ExtractedFact] = []
    entities: List[ExtractedEntity] = []
    relationships: List[ExtractedRelationship] = []
    raw_facts: List[str] = []


# ─── v3 endpoint request models ──────────────────────────────────────────────

class ReflectRequest(BaseModel):
    query: str
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    limit: int = Field(20, ge=1, le=50)


class ConsolidateRequest(BaseModel):
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    similarity_threshold: float = Field(0.80, ge=0.5, le=0.99)
    dry_run: bool = Field(False, description="If true, return clusters without actually merging")


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Server-side path to the file to ingest")
    source: str = Field("file-ingest", description="Source label for stored memories")
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"


class MigrateRequest(BaseModel):
    user_id: str = Field("omar", description="User whose memories to migrate (or 'all')")
    source_collection: str = Field("openclaw_memories", description="Old collection name")
    dry_run: bool = Field(False, description="Preview migration without writing")


# ─── v4 new models — Context Tree Architecture ─────────────────────────────────

class ContextRequest(BaseModel):
    query: str
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    max_depth: int = Field(2, ge=0, le=3, description="Max tree depth to traverse (0=L0, 1=L1, 2=L2)")
    include_summaries: bool = Field(True, description="Include category summaries in response")


class CompactRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="Full conversation transcript")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    extract_decisions: bool = Field(True, description="Extract decision points")
    extract_facts: bool = Field(True, description="Extract factual information")
    extract_preferences: bool = Field(True, description="Extract user preferences")
    extract_actions: bool = Field(True, description="Extract action items")
    extract_temporal: bool = Field(True, description="Extract time-sensitive events")


class TimelineRequest(BaseModel):
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    category: Optional[MemoryCategory] = None
    min_importance: Optional[int] = Field(None, ge=1, le=10)
    from_date: Optional[str] = Field(None, description="ISO datetime string")
    to_date: Optional[str] = Field(None, description="ISO datetime string")
    limit: int = Field(50, ge=1, le=1000)


class CategorySummary(BaseModel):
    category: MemoryCategory
    l0_summary: str = Field(..., description="Brief category overview (~50 tokens)")
    l1_summary: Optional[str] = Field(None, description="Detailed summary (~200 tokens)")
    memory_count: int = 0
    last_updated: Optional[str] = None
    importance_avg: Optional[float] = None


class ContextTreeResponse(BaseModel):
    query: str
    relevant_categories: List[MemoryCategory]
    category_summaries: List[CategorySummary]
    context: str = Field(..., description="Assembled hierarchical context")
    memories_used: List[Dict[str, Any]] = []
    total_tokens: int = 0


class GraphRequest(BaseModel):
    user_id: str = "omar"
    agent_id: Optional[str] = "zeno"
    similarity_threshold: float = Field(0.7, ge=0.5, le=0.99)
    max_nodes: int = Field(100, ge=10, le=500)
    category: Optional[MemoryCategory] = None


class GraphNode(BaseModel):
    id: str
    memory: str
    category: str = "uncategorized"
    importance: int = 5
    created_at: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None


class GraphEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    similarity: float = Field(..., ge=0.0, le=1.0)
    weight: Optional[float] = Field(None, description="Visual weight for rendering")


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    categories: List[str]
    total_memories: int


class TimelineEntry(BaseModel):
    id: str
    memory: str
    category: str
    importance: int
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class TimelineResponse(BaseModel):
    entries: List[TimelineEntry]
    total_count: int
    filters_applied: Dict[str, Any]


class CompactResponse(BaseModel):
    facts_extracted: int
    decisions_extracted: int
    preferences_extracted: int
    actions_extracted: int
    temporal_events_extracted: int
    total_memories_created: int
    processing_time_ms: int


# ─── UI / graph models (legacy compatibility) ──────────────────────────────────

class MemoryNode(BaseModel):
    id: str
    memory: str
    category: str = "uncategorized"
    importance: int = 5
    created_at: Optional[str] = None
    neighbors: List[Dict[str, Any]] = []


class ClusterGroup(BaseModel):
    category: str
    count: int
    memories: List[Dict[str, Any]]