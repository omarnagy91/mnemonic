# Contributing to Mnemonic

Thanks for considering a contribution. This is a small, single-maintainer project; the bar for a useful PR is low and turnaround on review is usually within a few days.

## Where help is wanted

The [roadmap in the README](README.md#roadmap) is the source of truth. In short:

1. **Generalization**: removing the single-user defaults and the leftover conventions from the original in-house agent stack.
2. **Wiring fixes**: a few modules (contradiction detection, the import pipeline) are implemented but not correctly connected to routes.
3. **Tests**: the pure-function modules are unit-tested; route-level tests with a mocked mem0 instance do not exist yet.
4. **Docs**: corrections and clarifications are always accepted.

Issues labeled [`good first issue`](https://github.com/omarnagy91/mnemonic/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are scoped so you do not need to understand the whole codebase first.

## Development setup

```bash
git clone https://github.com/omarnagy91/mnemonic && cd mnemonic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest
```

Running the server locally additionally needs Docker (for Qdrant) and an `OPENAI_API_KEY`; see the [Quickstart](README.md#quickstart). The unit tests do not need either.

## Before you open a PR

```bash
ruff check server tests        # lint, must be clean
python -m compileall -q server # syntax across all modules
pytest -q                      # unit tests, must pass
```

CI runs exactly these on Python 3.11 and 3.12.

## Guidelines

- Keep PRs scoped to one change. Two small PRs beat one mixed one.
- New pure functions (parsers, filters, scoring) should come with a unit test in `tests/`.
- Behavior changes to routes should update the README's API section in the same PR.
- No new hard dependencies without discussion in an issue first. The current footprint (FastAPI, mem0, Qdrant client, OpenAI client) is deliberate.
- Match the existing code style; ruff with default rules is the arbiter.

## Reporting bugs

Use the bug report template. The most useful bug report includes: the request you sent (curl line or equivalent), the response or traceback, and your Python version. `GET /health` output helps when the problem looks environmental.

## Security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE).
