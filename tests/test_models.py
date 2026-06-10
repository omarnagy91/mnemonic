"""Unit tests for server/models.py request/response models."""

import pytest
from pydantic import ValidationError

from models import (
    CompactRequest,
    ContextRequest,
    ExtractedFact,
    GraphEdge,
    ImportStatus,
    MemoryCategory,
)


def test_memory_category_values():
    assert MemoryCategory("business") is MemoryCategory.business
    with pytest.raises(ValueError):
        MemoryCategory("unknown-category")


def test_graph_edge_accepts_from_to_aliases():
    edge = GraphEdge(**{"from": "a", "to": "b", "similarity": 0.91})
    assert edge.from_id == "a"
    assert edge.to_id == "b"


def test_graph_edge_accepts_field_names_too():
    edge = GraphEdge(from_id="a", to_id="b", similarity=0.5)
    assert edge.similarity == 0.5


def test_graph_edge_similarity_bounds():
    with pytest.raises(ValidationError):
        GraphEdge(from_id="a", to_id="b", similarity=1.5)


def test_extracted_fact_defaults_and_bounds():
    fact = ExtractedFact(text="User opened a second branch in Mansoura")
    assert fact.category is MemoryCategory.uncategorized
    assert 1 <= fact.importance <= 10
    with pytest.raises(ValidationError):
        ExtractedFact(text="x", importance=11)


def test_import_status_defaults():
    job = ImportStatus(job_id="abc12345", status="queued", source="text")
    assert job.processed_items == 0
    assert job.errors == []


def test_context_request_depth_bounds():
    assert ContextRequest(query="q", max_depth=0).max_depth == 0
    with pytest.raises(ValidationError):
        ContextRequest(query="q", max_depth=4)


def test_compact_request_defaults_all_extractors_on():
    req = CompactRequest(messages=[{"role": "user", "content": "hello"}])
    assert req.extract_facts and req.extract_decisions and req.extract_preferences
    assert req.extract_actions and req.extract_temporal
