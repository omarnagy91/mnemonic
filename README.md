<div align="center">

# Mnemonic

### Self-hosted, categorized memory for AI agents. Your conversation history stays on your box.

[![CI](https://github.com/omarnagy91/mnemonic/actions/workflows/ci.yml/badge.svg)](https://github.com/omarnagy91/mnemonic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-9D7BEA.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-1A1A1A.svg)
[![Built on mem0](https://img.shields.io/badge/built%20on-mem0%20%2B%20Qdrant-1A1A1A.svg)](https://github.com/mem0ai/mem0)

</div>

---

## What this is

A small FastAPI server that gives an AI agent long-term memory you host yourself. It wraps [mem0](https://github.com/mem0ai/mem0) (which does the vector storage and the add/update/delete reasoning over facts) and adds the parts a single library leaves to you: automatic categorization, a tiered "context tree" so you load a useful slice of memory instead of dumping everything into the prompt, a compaction hook for long sessions, and a visual explorer for seeing what the agent actually remembers.

It runs on your own machine against your own [Qdrant](https://github.com/qdrant/qdrant). The conversation history and the vectors never leave your box, and the only paid dependency is whatever LLM you point the categorizer and the synthesis step at.

### A note on where this came from

Mnemonic was built as the memory layer for an in-house multi-agent stack (it still ships an OpenClaw plugin under `plugin/`). It is being generalized into something any agent runtime can use. That history shows in a few places: single-user defaults, a couple of modules that are implemented but not yet wired into the main add path (see the [roadmap](#roadmap)). The core server, categorization, retrieval, and the explorer work today; the rough edges are named honestly below rather than hidden.

## What it does

- **Categorized storage.** Every memory is auto-sorted into one of seven categories (personal, business, technical, decision, relationship, temporal, uncategorized) with an importance score, so retrieval and summaries can reason about *kinds* of memory, not just a flat vector pile.
- **Context tree (L0/L1/L2).** Instead of returning raw hits, `/context` assembles a tiered view: L0 category summaries, then progressively more detail. You give the agent a compact map of what it knows and drill in only where the query needs it.
- **Vector search.** `/search` is mem0 + Qdrant similarity search with weighted scoring (recency and importance), plus a `/reflect` endpoint that synthesizes an answer across the retrieved memories.
- **Fact resolution.** mem0 decides add vs update vs delete when a new fact arrives, so "moved to a new city" updates the old fact rather than stacking a contradiction. An explicit `ContradictionDetector` (an LLM judges keep-old / keep-new / merge) ships in `contradiction.py` for stricter control; wiring it into the `/add` path is a tracked issue, not a done feature.
- **Compaction hook.** `/compact` saves a session's working context before it hits a token limit, so a long conversation does not lose its early turns.
- **Import pipeline.** `/import` ingests history from text, JSON, CSV, and exported social data into the memory store as an async job.
- **Visual explorer.** `/explorer` and `/dashboard` render the graph, the timeline, and per-category views in the browser, so you can debug what the agent remembers instead of guessing.

## Quickstart

You need Python 3.11+, Docker (for Qdrant), and an OpenAI API key for the categorizer and synthesis steps.

```bash
git clone https://github.com/omarnagy91/mnemonic && cd mnemonic

# 1. Qdrant (vector store) in Docker
docker run -d -p 6333:6333 -v "$(pwd)/.data/qdrant:/qdrant/storage" qdrant/qdrant

# 2. server deps + key
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# 3. run it
uvicorn server.server:app --host 0.0.0.0 --port 8080
```

Store and recall:

```bash
curl -X POST localhost:8080/add -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"I prefer Postgres over Mongo for new projects."}],"user_id":"me"}'

curl -X POST localhost:8080/search -H 'content-type: application/json' \
  -d '{"query":"what database do I like?","user_id":"me"}'
```

Then open `http://localhost:8080/explorer` to see the graph and timeline. The full route list is in [server/server.py](server/server.py) (`/add`, `/search`, `/context`, `/reflect`, `/compact`, `/timeline`, `/graph`, `/categories`, `/import`, plus health and stats).

## How it works

```
   conversation ──▶ /add ──▶ categorize + score ──▶ mem0 (add/update/delete) ──▶ Qdrant
                                                                                    │
   query ──▶ /context ──▶ retrieval (vector + weighted) ──▶ L0/L1/L2 context tree ◀─┘
                              │
                              └─▶ /reflect ──▶ synthesized answer across memories
```

mem0 owns the vector storage and the per-fact add/update/delete decision. Mnemonic owns the categorization, the importance weighting, the tiered context assembly, the compaction and import endpoints, and the explorer UI.

## Honest comparison

| Option | What it is | When to use it instead |
| --- | --- | --- |
| [mem0](https://github.com/mem0ai/mem0) (self-host) | The library Mnemonic is built on | You want just the memory primitive and will build your own categorization, context assembly, and UI. Mnemonic is those layers, pre-built. |
| [Mem0 Platform](https://mem0.ai) / [Supermemory](https://supermemory.ai) | Hosted memory APIs | You would rather pay a monthly fee than run Qdrant and a server, and you are comfortable with conversation history living in their cloud. |
| [Zep](https://github.com/getzep/zep) | Mature self-hosted memory server with its own store | You want a larger, more battle-tested project with a bigger community. Zep is further along; Mnemonic is smaller and easier to read end to end. |

Mnemonic's niche: small, self-hosted, category-aware, and readable in one sitting, with the conversation data staying on your machine.

## Roadmap

The honest near-term list. Help is welcome on any of these (see [issues](https://github.com/omarnagy91/mnemonic/issues)).

- Wire the explicit `ContradictionDetector` into the `/add` path behind a config flag, with tests.
- Finish wiring the import pipeline endpoints to the job runner and document the supported formats.
- Remove the single-user (`user_id="me"`) and in-house defaults so a fresh deploy is multi-user out of the box.
- Route-level tests with a mocked mem0 instance (today the pure-function modules are unit-tested, the routes are not).
- A `docker-compose.yml` that brings up Qdrant and the server together.
- Optional MCP server wrapper so MCP-speaking agents can use it without the HTTP client.

## Contributing

The pure-function modules (`categorizer`, `extractor`, `importer`, `models`) have unit tests that run offline in under a second:

```bash
pip install -r requirements.txt && pip install pytest
pytest -q
```

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the [good first issues](https://github.com/omarnagy91/mnemonic/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Security reports go to the address in [SECURITY.md](SECURITY.md).

## License and author

MIT, see [LICENSE](LICENSE).

Built by [Omar G. Nagy](https://omargnagy.com), AI Systems Engineer. I build memory and evaluation infrastructure for LLM and agent products. Case study: [omargnagy.com/work/mnemonic](https://omargnagy.com/work/mnemonic).
