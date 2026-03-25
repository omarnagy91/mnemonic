"""
Mnemonic v2 — Social Data Import Pipeline
Handles Twitter archive, LinkedIn CSV, plain text, generic JSON/CSV.
Processes in batches, runs async, reports progress.
"""

import os
import io
import csv
import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from models import ImportStatus, MemoryCategory
from extractor import SmartExtractor

logger = logging.getLogger("mnemonic.importer")

# In-memory job store (good enough for single-process server)
_jobs: Dict[str, ImportStatus] = {}


def get_job(job_id: str) -> Optional[ImportStatus]:
    return _jobs.get(job_id)


def list_jobs() -> List[ImportStatus]:
    return list(_jobs.values())


def _create_job(source: str) -> ImportStatus:
    job_id = str(uuid.uuid4())[:8]
    job = ImportStatus(
        job_id=job_id,
        status="queued",
        source=source,
        started_at=datetime.utcnow().isoformat(),
    )
    _jobs[job_id] = job
    return job


# ─── Parsers ──────────────────────────────────────────────────────────────────

def _parse_twitter_archive(data: str) -> List[str]:
    """
    Parse Twitter archive tweets.js (starts with 'window.YTD.tweet.part0 = [...]')
    Returns list of tweet text strings.
    """
    # Strip the JS variable assignment
    stripped = data.strip()
    if stripped.startswith("window."):
        eq_idx = stripped.index("=")
        stripped = stripped[eq_idx + 1:].strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]

    try:
        tweets_data = json.loads(stripped)
    except json.JSONDecodeError:
        # Try as plain JSON array
        tweets_data = json.loads(data)

    texts = []
    for item in tweets_data:
        tweet = item.get("tweet", item)
        text = tweet.get("full_text", tweet.get("text", ""))
        if text and not text.startswith("RT @"):
            texts.append(text)
    return texts


def _parse_linkedin_csv(data: str) -> List[str]:
    """Parse LinkedIn connections CSV export."""
    reader = csv.DictReader(io.StringIO(data))
    entries = []
    for row in reader:
        parts = []
        name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
        if name:
            parts.append(f"Connected with {name}")
        company = row.get("Company", "")
        position = row.get("Position", "")
        if company and position:
            parts.append(f"who works as {position} at {company}")
        elif company:
            parts.append(f"who works at {company}")
        connected_on = row.get("Connected On", "")
        if connected_on:
            parts.append(f"on {connected_on}")
        if parts:
            entries.append(" ".join(parts))
    return entries


def _parse_generic_json(data: str) -> List[str]:
    """Parse generic JSON — extract string values."""
    parsed = json.loads(data)
    texts = []

    def _walk(obj: Any):
        if isinstance(obj, str) and len(obj) > 10:
            texts.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)

    _walk(parsed)
    return texts


def _parse_generic_csv(data: str) -> List[str]:
    """Parse generic CSV — join each row into a sentence."""
    reader = csv.DictReader(io.StringIO(data))
    return [", ".join(f"{k}: {v}" for k, v in row.items() if v) for row in reader]


def _parse_plain_text(data: str) -> List[str]:
    """Split plain text into paragraphs or chunks."""
    paragraphs = [p.strip() for p in data.split("\n\n") if p.strip()]
    # If too few paragraphs, split by lines
    if len(paragraphs) <= 1:
        paragraphs = [l.strip() for l in data.split("\n") if l.strip() and len(l.strip()) > 20]
    return paragraphs


# ─── Batch processing ────────────────────────────────────────────────────────

BATCH_SIZE = 50  # items per extraction batch


def _chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


async def _process_import(
    job: ImportStatus,
    items: List[str],
    extractor: SmartExtractor,
    memory,  # mem0 Memory instance
    user_id: str,
    source_tag: str,
):
    """Process import items in batches, extract facts, store via mem0."""
    job.status = "processing"
    job.total_items = len(items)

    for batch in _chunk_list(items, BATCH_SIZE):
        batch_text = "\n\n".join(batch)

        # Limit batch text to avoid token overflow (~50k chars ≈ ~12k tokens)
        if len(batch_text) > 50000:
            batch_text = batch_text[:50000]

        try:
            result = extractor.extract_from_text(batch_text)
            for fact in result.facts:
                try:
                    memory.add(
                        messages=[{"role": "user", "content": fact.text}],
                        user_id=user_id,
                        metadata={
                            "source": source_tag,
                            "category": fact.category.value,
                            "importance": fact.importance,
                            "imported_at": datetime.utcnow().isoformat(),
                        },
                    )
                    job.facts_extracted += 1
                except Exception as e:
                    job.errors.append(f"Store error: {str(e)[:100]}")
        except Exception as e:
            job.errors.append(f"Extraction error: {str(e)[:200]}")

        job.processed_items += len(batch)
        # Yield control to event loop
        await asyncio.sleep(0)

    job.status = "completed"
    job.completed_at = datetime.utcnow().isoformat()
    logger.info(f"Import job {job.job_id} completed: {job.facts_extracted} facts from {job.total_items} items")


async def start_import(
    source: str,
    data: str,
    extractor: SmartExtractor,
    memory,
    user_id: str = "omar",
) -> ImportStatus:
    """
    Parse source data, create async import job.
    Returns ImportStatus immediately; processing runs in background.
    """
    job = _create_job(source)
    source_tag = f"import:{source}"

    # Parse based on source type
    try:
        if source == "twitter":
            items = _parse_twitter_archive(data)
        elif source == "linkedin":
            items = _parse_linkedin_csv(data)
        elif source == "json":
            items = _parse_generic_json(data)
        elif source == "csv":
            items = _parse_generic_csv(data)
        else:  # text or fallback
            items = _parse_plain_text(data)
    except Exception as e:
        job.status = "failed"
        job.errors.append(f"Parse error: {str(e)[:300]}")
        job.completed_at = datetime.utcnow().isoformat()
        return job

    if not items:
        job.status = "completed"
        job.total_items = 0
        job.completed_at = datetime.utcnow().isoformat()
        return job

    # Launch background task
    asyncio.create_task(_process_import(job, items, extractor, memory, user_id, source_tag))
    return job


def load_file_data(file_path: str, max_bytes: int = 200 * 1024 * 1024) -> str:
    """
    Stream-read a file up to max_bytes to avoid OOM on large archives.
    """
    size = os.path.getsize(file_path)
    read_size = min(size, max_bytes)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(read_size)
