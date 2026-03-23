<div align="center">

# 🧠 Mnemonic

**Self-hosted AI memory for OpenClaw agents.**

Give your AI persistent, intelligent memory — no cloud subscription required.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Architecture](#architecture) · [Configuration](#configuration)

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Plugin-purple.svg)](https://github.com/openclaw/openclaw)
[![mem0](https://img.shields.io/badge/Powered%20by-mem0-green.svg)](https://github.com/mem0ai/mem0)

</div>

## What is Mnemonic?

Your AI forgets everything between conversations. Mnemonic fixes that — **locally, for free.**

It's a self-hosted memory layer for [OpenClaw](https://github.com/openclaw/openclaw) agents that:

- 🧠 **Auto-captures** facts, preferences, and decisions from every conversation
- 🔍 **Auto-recalls** relevant memories before each AI turn
- 🔄 **Resolves contradictions** — "moved to SF" automatically supersedes "lives in NYC"
- ⏰ **Handles temporal info** — "exam tomorrow" expires after the date
- 👤 **Builds user profiles** — persistent facts + recent context
- 🏠 **Runs 100% locally** — your data never leaves your server

Think of it as [Supermemory](https://supermemory.ai) or [mem0 Cloud](https://mem0.ai), but self-hosted and free.

## Quick Start

### Prerequisites

- [OpenClaw](https://github.com/openclaw/openclaw) installed and running
- Docker (for Qdrant vector database)
- Python 3.10+ (for mem0 API server)
- An OpenAI API key (for embeddings + fact extraction)

### 1. Start Qdrant (Vector Database)

```bash
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/.data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

### 2. Install & Start the mem0 API Server

```bash
# Install dependencies
pip install mem0ai fastapi uvicorn

# Copy the server
cp server/server.py /path/to/mem0-server/server.py

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Start the server
python server/server.py
# → Running on http://127.0.0.1:8765
```

For production, use the included systemd service:

```bash
cp services/mem0-server.service ~/.config/systemd/user/
# Edit the service file to set your OPENAI_API_KEY and WorkingDirectory
systemctl --user daemon-reload
systemctl --user enable --now mem0-server
```

### 3. Install the OpenClaw Plugin

```bash
# Copy plugin files
cp -r plugin/ ~/.openclaw/extensions/openclaw-mem0/
```

Add to your `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "openclaw-mem0"
    },
    "entries": {
      "openclaw-mem0": {
        "enabled": true,
        "config": {
          "apiUrl": "http://127.0.0.1:8765",
          "userId": "default",
          "agentId": "assistant",
          "autoRecall": true,
          "autoCapture": true,
          "maxRecallResults": 10,
          "debug": false
        }
      }
    },
    "installs": {
      "openclaw-mem0": {
        "source": "path",
        "spec": "openclaw-mem0",
        "installPath": "~/.openclaw/extensions/openclaw-mem0",
        "version": "1.0.0",
        "resolvedName": "openclaw-mem0",
        "resolvedVersion": "1.0.0",
        "resolvedSpec": "openclaw-mem0@1.0.0"
      }
    },
    "allow": ["openclaw-mem0"]
  }
}
```

Restart OpenClaw:

```bash
openclaw gateway restart
```

That's it. Memory is now active.

## How It Works

```
You: "What's the status of my project?"

1. 🔍 Auto-Recall: Plugin searches mem0 for "project status"
2. 📋 Context Injection: Relevant memories injected before AI responds
3. 🤖 AI Response: Agent responds with full context
4. 💾 Auto-Capture: mem0 extracts new facts from the conversation
5. 🧠 LLM Processing: GPT processes facts, resolves contradictions
```

### What Gets Captured

- Facts you share ("I work at Acme Corp")
- Preferences ("I prefer dark mode")
- Decisions ("We decided to use PostgreSQL")
- Project context ("The deadline is March 30th")

### What Gets Filtered

- Short messages and greetings
- System events and heartbeats
- Duplicate information
- Previously injected memory context

## Architecture

```
OpenClaw Gateway
  └── openclaw-mem0 plugin (TypeScript)
        ├── before_agent_start → search mem0 → inject context
        ├── agent_end → extract last turn → feed to mem0
        └── tools: mem0_store, mem0_recall, mem0_forget, mem0_profile
              │
              ▼
        mem0 API Server (Python/FastAPI, localhost:8765)
              │
              ├── LLM Extraction (OpenAI GPT — extracts facts, resolves contradictions)
              └── Qdrant Vector DB (Docker, localhost:6333)
```

### Components

| Component | Port | RAM | Purpose |
|-----------|------|-----|---------|
| **Qdrant** | 6333 | ~30MB | Vector storage for memory embeddings |
| **mem0 API** | 8765 | ~100MB | REST API + LLM-powered fact extraction |
| **Plugin** | — | ~0MB | OpenClaw integration (runs in gateway process) |

### Cost

- **Qdrant + mem0:** Free (self-hosted)
- **OpenAI Embeddings:** ~$0.02/million tokens
- **LLM Extraction:** ~$0.02-0.05/day with gpt-4.1-nano
- **Total:** ~$1-2/month for active use

## Configuration

### Plugin Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `apiUrl` | string | `http://127.0.0.1:8765` | mem0 API server URL |
| `userId` | string | `default` | User ID for memory scoping |
| `agentId` | string | `assistant` | Agent ID for memory scoping |
| `autoRecall` | boolean | `true` | Inject memories before each AI turn |
| `autoCapture` | boolean | `true` | Extract facts after each AI turn |
| `maxRecallResults` | number | `10` | Max memories injected per turn |
| `debug` | boolean | `false` | Verbose logging |

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `MEM0_PORT` | `8765` | API server port |
| `MEM0_LLM_MODEL` | `gpt-4.1-nano` | LLM for fact extraction |
| `MEM0_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/add` | Add conversation messages (mem0 extracts facts) |
| `POST` | `/search` | Semantic memory search |
| `GET` | `/profile/{user_id}` | Get all memories for a user |
| `DELETE` | `/forget` | Delete a memory by ID |
| `PUT` | `/update` | Update a memory |
| `GET` | `/stats` | Memory statistics |

## Tools Available to Agents

Once installed, your OpenClaw agent gets these tools:

| Tool | Description |
|------|-------------|
| `mem0_store` | Manually save information to memory |
| `mem0_recall` | Search memories by query |
| `mem0_forget` | Delete a specific memory |
| `mem0_profile` | View all stored memories |

## vs. Alternatives

| Feature | Mnemonic | Supermemory | mem0 Cloud | OpenClaw LanceDB |
|---------|----------|-------------|------------|------------------|
| Self-hosted | ✅ | ❌ ($paid) | ❌ ($paid) | ✅ |
| LLM extraction | ✅ | ✅ | ✅ | ❌ |
| Contradiction resolution | ✅ | ✅ | ✅ | ❌ |
| User profiles | ✅ | ✅ | ✅ | ❌ |
| Auto-recall | ✅ | ✅ | ✅ | ✅ |
| Auto-capture | ✅ | ✅ | ✅ | ✅ |
| Cost | ~$1/mo | $20+/mo | $20+/mo | ~$0 |
| Setup time | 10 min | 2 min | 2 min | 1 min |

## License

MIT — use it however you want.

## Credits

- [mem0](https://github.com/mem0ai/mem0) — the memory engine (Apache 2.0)
- [Qdrant](https://github.com/qdrant/qdrant) — vector database (Apache 2.0)
- [OpenClaw](https://github.com/openclaw/openclaw) — the agent platform
- Built by [NeuraScale](https://neurascale.org)
