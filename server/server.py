#!/usr/bin/env python3
"""
mem0 Memory API Server for OpenClaw
REST API for memory operations: add, search, profile, forget
Uses Qdrant (local Docker) for vector storage and OpenAI for LLM extraction
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from mem0 import Memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0-server")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.environ.get("MEM0_EMBEDDING_MODEL", "text-embedding-3-small")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": LLM_MODEL,
            "api_key": OPENAI_API_KEY,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBEDDING_MODEL,
            "api_key": OPENAI_API_KEY,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": "openclaw_memories",
        }
    },
    "version": "v1.1",
}

memory = Memory.from_config(config)
app = FastAPI(title="mem0 Memory Server", version="1.0.0")

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

@app.get("/health")
async def health():
    return {"status": "ok", "llm": LLM_MODEL, "embedding": EMBEDDING_MODEL}

@app.post("/add")
async def add_memory(req: AddRequest):
    try:
        result = memory.add(
            messages=req.messages,
            user_id=req.user_id,
            agent_id=req.agent_id,
            metadata=req.metadata,
        )
        logger.info(f"Added memory for user={req.user_id}: {result}")
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_memory(req: SearchRequest):
    try:
        results = memory.search(
            query=req.query,
            user_id=req.user_id,
            agent_id=req.agent_id,
            limit=req.limit,
        )
        return {"ok": True, "results": results}
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{user_id}")
async def get_profile(user_id: str, agent_id: Optional[str] = "zeno"):
    try:
        memories = memory.get_all(user_id=user_id, agent_id=agent_id)
        return {"ok": True, "memories": memories}
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/forget")
async def forget_memory(req: ForgetRequest):
    try:
        memory.delete(req.memory_id)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Failed to forget memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update")
async def update_memory(req: UpdateRequest):
    try:
        result = memory.update(req.memory_id, data=req.data)
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    try:
        all_memories = memory.get_all(user_id="omar")
        count = len(all_memories.get("results", []) if isinstance(all_memories, dict) else all_memories)
        return {"ok": True, "total_memories": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MEM0_PORT", "8765"))
    logger.info(f"Starting mem0 server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
