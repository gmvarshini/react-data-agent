# CLAUDE.md

Notes for working on this repo with Claude Code.

## Project

A data analysis agent for a small sales dataset, built with LangGraph and a local
Llama 3.2 model through Ollama. You ask a question in plain English and the agent
answers by calling tools that run pandas code on the data, so the numbers are
real and every step is printed as a trace.

There are two modes:

- `single`: one ReAct agent with all four tools (`src/agent.py`)
- `multi`: a supervisor that routes work to two specialist agents (`src/multi_agent.py`)

## Layout

- `src/config.py`: all settings, loaded from `.env` with pydantic-settings
- `src/tools.py`: the four tools and their Pydantic input schemas
- `src/agent.py`: single agent built with `create_react_agent`
- `src/multi_agent.py`: supervisor graph, segment analyst and statistics analyst
- `src/main.py`: command line entry point
- `tests/`: tool tests and supervisor graph tests
- `scripts/generate_data.py`: rebuilds `data/sales_data.csv` (seed 42)

## Commands

```bash
uv sync                                    # install
uv run pytest                              # tests
uv run ruff check                          # lint
uv run python -m src.main --question "Which product category has the highest total revenue?"
uv run python -m src.main --mode multi --question "..."
```

Ollama must be running with `llama3.2` pulled to run the agent. The tests do not
need Ollama.

## Rules

- Settings go in `src/config.py` and `.env.example`, never hard coded.
- Every tool needs a clear docstring and a Pydantic input schema. The model picks
  tools only from the name, docstring and schema, so wording matters.
- Tool logic lives in a plain `_function` that takes an optional DataFrame, so it
  can be tested without the agent.
- Tools validate their inputs and raise `ValueError` with a clear message.
- Keep `temperature=0` for the agent models.
- Tests must not call an LLM. Use the small fixture DataFrame for tools and fake
  routers and workers for the supervisor graph.
- Type hints and Google style docstrings on all functions.
- Plain English in docs and comments. No em dashes.

## Adding a tool

1. Write `_my_tool(..., frame=None)` in `src/tools.py` with its logic.
2. Add a Pydantic input schema and a `@tool` wrapper with a precise docstring.
3. Add it to `ALL_TOOLS` and to one specialist in `WORKER_TOOLS` in `src/multi_agent.py`.
4. Add tests with known expected values in `tests/test_tools.py`.
5. Mention it in the README tools table.

## CI/CD

`.github/workflows/tests.yml` runs ruff and pytest on every push and pull request.
After the tests pass on `main`, it builds the Docker image and publishes it to
`ghcr.io`. Keep `uv run ruff check` and `uv run pytest` green before pushing.
