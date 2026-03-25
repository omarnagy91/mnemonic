<div align="center">

# 🧠 Mnemonic

**Self-hosted AI memory for OpenClaw agents.**

Give your AI persistent, intelligent memory — no cloud subscription required.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Architecture](#architecture) · [v4 Features](#v4-features)

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Plugin-purple.svg)](https://github.com/openclaw/openclaw)
[![mem0](https://img.shields.io/badge/Powered%20by-mem0-green.svg)](https://github.com/mem0ai/mem0)

</div>

## What is Mnemonic?

Your AI forgets everything between conversations. Mnemonic fixes that — **locally, for free.**

It's a self-hosted memory layer for [OpenClaw](https://github.com/openclaw/openclaw) agents that:

- 🌳 **Context Tree** — hierarchical memory organized by category with L0/L1/L2 tiered loading
- 🔄 **Auto-captures** facts, preferences, and decisions from every conversation
- 🔍 **Auto-recalls** relevant memories before each AI turn
- ⚡ **Contradiction resolution** — "moved to SF" supersedes "lives in NYC"
- 📦 **Compaction hook** — saves context before token limits hit
- 📊 **Visual explorer** — graph visualization, timeline, dashboard
- 🏠 **Runs 100% locally** — your data never leaves your server

Think of it as [Supermemory](https://supermemory.ai) or [mem0 Cloud](https://mem0.ai), but self-hosted and free.

## v4 Features

### Context Tree Architecture
Memories organized into categories (personal, business, technical, decisions, relationships, temporal) with hierarchical summaries:

- **L0**: Category summaries (~50 tokens each, always loaded)
- **L1**: Detailed summaries (~200 tokens, loaded when relevant)
- **L2**: Individual memories (loaded for specific queries)

```bash
# Get assembled context for a query
curl -X POST http://localhost:8765/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"what projects am I working on?","user_id":"default"}'
```

### Compaction Hook
Save valuable context before token limits hit:

```bash
curl -X POST http://localhost:8765/compact \
  -H 'Content-Type: application/json' \
  -d '{"messages":[...],"user_id":"default","session_id":"abc"}'
```

### Memory Graph Visualization
Interactive graph with real cosine similarity edges:

- **Explorer UI**: `http://localhost:8765/explorer` — vis-network graph + timeline
- **Dashboard**: `http://localhost:8765/dashboard` — Chart.js analytics
- **Graph API**: `GET /graph?user_id=default` — similarity-based edges computed via Qdrant

### Timeline & Categories
```bash
# Chronological timeline with filters
curl 'http://localhost:8765/timeline?user_id=default&category=business&min_importance=7'

# Category summaries
curl 'http://localhost:8765/categories?user_id=default'
```

## Quick Start

### Prerequisites

- Docker (for Qdrant vector database)
- Python 3.10+ (for mem0 API server)
- An OpenAI API key (for embeddings + fact extraction)

### 1. Start Qdrant

```bash
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/.data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

### 2. Install & Start the API Server

```bash
pip install mem0ai fastapi uvicorn openai qdrant-client

export OPENAI_API_KEY="sk-..."
cd server && python server.py
# → Running on http://127.0.0.1:8765
```

### 3. Install the OpenClaw Plugin

```bash
cp -r plugin/ ~/.openclaw/extensions/openclaw-mem0/
```

Add to `~/.openclaw/openclaw.json` — see [plugin configuration](#configuration).

## Architecture

```
OpenClaw Gateway
  └── openclaw-mem0 plugin (TypeScript)
        ├── before_agent_start → search mem0 → inject context
        ├── agent_end → extract last turn → feed to mem0
        └── tools: mem0_store, mem0_recall, mem0_forget, mem0_profile
              │
              ▼
        Mnemonic API Server (Python/FastAPI, localhost:8765)
              │
              ├── Context Tree (hierarchical category summaries)
              ├── LLM Extraction (OpenAI GPT — fact extraction + contradiction resolution)
              ├── Smart Categorizer (6 categories + importance scoring)
              └── Qdrant Vector DB (Docker, localhost:6333)
```

### Components

| Component | Port | RAM | Purpose |
|-----------|------|-----|---------|
| **Qdrant** | 6333 | ~30MB | Vector storage + similarity search |
| **Mnemonic API** | 8765 | ~100MB | REST API + context tree + LLM extraction |
| **Plugin** | — | ~0MB | OpenClaw integration (runs in gateway) |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with component status |
| `POST` | `/add` | Add conversation (mem0 extracts facts) |
| `POST` | `/search` | Semantic memory search |
| `POST` | `/context` | **v4** Context tree assembly |
| `POST` | `/compact` | **v4** Compaction hook |
| `GET` | `/profile/{user_id}` | All memories for user |
| `GET` | `/graph` | **v4** Similarity graph data |
| `GET` | `/timeline` | **v4** Chronological timeline |
| `GET` | `/categories` | **v4** Category summaries |
| `POST` | `/consolidate` | Merge duplicate memories |
| `GET` | `/stats` | Memory statistics |
| `GET` | `/explorer` | Graph visualization UI |
| `GET` | `/dashboard` | Analytics dashboard |

## vs. Alternatives

| Feature | Mnemonic v4 | ByteRover | Supermemory | mem0 Cloud |
|---------|-------------|-----------|-------------|------------|
| Self-hosted | ✅ | ❌ Cloud | ❌ Cloud | ❌ Cloud |
| Context Tree | ✅ | ✅ | ❌ | ❌ |
| Graph visualization | ✅ | ❌ | ❌ | ❌ |
| Compaction hook | ✅ | ✅ | ❌ | ❌ |
| Contradiction resolution | ✅ | ✅ | ✅ | ✅ |
| Cost | ~$2/mo | $20+/mo | $20+/mo | $20+/mo |
| Data privacy | ✅ Your server | ❌ Their cloud | ❌ Their cloud | ❌ Their cloud |

## License

MIT — use it however you want.

## Credits

- [mem0](https://github.com/mem0ai/mem0) — memory engine (Apache 2.0)
- [Qdrant](https://github.com/qdrant/qdrant) — vector database (Apache 2.0)
- [vis-network](https://github.com/visjs/vis-network) — graph visualization
- [Chart.js](https://www.chartjs.org/) — analytics charts
- [OpenClaw](https://github.com/openclaw/openclaw) — agent platform
- Built by [NeuraScale](https://neurascale.org)
