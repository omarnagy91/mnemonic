# Security Policy

## Reporting a vulnerability

Email **omar@neurascale.org** with a description of the issue, steps to reproduce, and the impact you believe it has. Please do not open a public issue for security problems.

You will get an acknowledgment within 72 hours and a status update once the report has been assessed. Fixes for confirmed issues in the latest release are prioritized over feature work.

## Supported versions

| Version | Supported |
|---|---|
| latest release (v4.x) | yes |
| anything older | no |

## Deployment notes you should know

Mnemonic is designed to run on hardware you control, and the current release makes assumptions you should be aware of before exposing it to a network:

- **No authentication.** The API server binds to `0.0.0.0:8765` and every endpoint is unauthenticated. Run it on localhost, behind a firewall, or behind a reverse proxy that adds auth. Do not expose the port to the public internet.
- **Qdrant is also unauthenticated** in the quickstart configuration (port 6333).
- **Memory content transits OpenAI.** Fact extraction, summaries, and embeddings call the OpenAI API with your memory text. Storage is local; processing is not (a fully local pipeline is on the roadmap).
- `OPENAI_API_KEY` is read from the environment and never written to disk by the server. Keep it out of shell history and unit files that are world-readable.
