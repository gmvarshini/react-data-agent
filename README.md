# LangGraph ReAct Multi-Agent System

A ReAct style agent for data analysis, built with LangGraph and a local Ollama
model. It runs in two modes: a single ReAct agent, or a multi-agent system
where a supervisor routes work to two specialist agents. You ask a plain English question about a sales dataset, and the agent
reasons step by step, calls purpose built tools to query the data, and returns
a precise, data backed answer along with the full trace of how it got there.

## What is the ReAct pattern?

ReAct stands for "Reason and Act". It is a way of structuring a language model
so that it interleaves two activities in a loop:

1. **Reason.** The model thinks about the question and decides what it needs to
   find out next.
2. **Act.** Instead of guessing, the model calls a tool (a normal function)
   that fetches or computes real information.

The tool returns a result, the model reads it, reasons again, and either calls
another tool or produces a final answer. This loop continues until the model
has enough information to respond.

The value of this pattern is grounding. A language model on its own will
happily invent numbers. By forcing it to act through tools that run real code
against real data, the answers become verifiable. The trace also makes the
agent transparent, because you can see exactly which tools were called and what
they returned.

In this project the loop is implemented as a small LangGraph state machine with
two nodes:

- An `agent` node that calls the model. The model either answers or requests a
  tool call.
- A `tools` node that runs the requested tool and feeds the result back.

Control flows START to `agent`. If the model asks for a tool, the graph moves
to `tools` and back to `agent`. When the model answers with no tool call, the
graph moves to END.

## Tools available to the agent

All tools operate on a sales dataset with one row per sale and the columns
`date`, `product_category`, `region`, `units_sold` and `revenue`.

| Tool | Purpose |
| --- | --- |
| `filter_by_region` | Filter the data to one region (North, South, East, West) and return its row count, total units sold and total revenue. |
| `aggregate_by_category` | Aggregate a metric (`revenue` or `units_sold`) per product category using sum, mean, min, max or count. Returns categories ranked from highest to lowest. |
| `compute_summary_statistics` | Return count, sum, mean, median, min, max and standard deviation for a numeric column across the whole dataset. |
| `find_top_performers` | Return the individual sales records with the highest values for a chosen metric, ranked in descending order. |

Each tool declares a Pydantic input schema and a descriptive docstring. The
model selects a tool based only on these descriptions and schemas, so the
wording is part of the design (see the note on prompt and schema design below).

## Multi-agent mode: a supervisor and two specialists

The single agent sees all four tools. The multi-agent mode
(`src/multi_agent.py`) splits the work into a small team instead:

| Agent | Role | Tools |
| --- | --- | --- |
| `supervisor` | Reads the question, decides who acts next, and writes the final answer. Never analyses data itself. | none |
| `segment_analyst` | Questions about one region or comparing product categories. | `filter_by_region`, `aggregate_by_category` |
| `statistics_analyst` | Questions about the overall distribution or the best individual sales. | `compute_summary_statistics`, `find_top_performers` |

Each specialist is its own small ReAct agent. The graph is:

```
START -> supervisor -> segment_analyst    -> supervisor
                    -> statistics_analyst -> supervisor
                    -> END (when the supervisor chooses FINISH)
```

The supervisor uses structured output (a Pydantic `Route` model) so its
decision is always one of `segment_analyst`, `statistics_analyst` or `FINISH`.

**Why split the work?** Fewer tools per agent means shorter prompts and fewer
wrong tool choices. A question with two parts, such as "How much revenue came
from the North region, and what is the median revenue?", is handled by two
specialists, one after the other.

**Context engineering.** A specialist adds only its short final report to the
shared state, not every internal tool call. This keeps the supervisor's context
small and focused.

**Tradeoffs.** More agents means more model calls, so answers are slower, and
there are more places where something can go wrong. For simple questions the
single agent is often enough.

Run it with:

```bash
uv run python -m src.main --mode multi --question "How much revenue came from the North region, and what is the median revenue?"
```

The graph logic is tested without a language model: `tests/test_multi_agent.py`
replaces the supervisor's decisions and the specialists with simple fakes and
checks routing, finishing, the step limit, and that every tool belongs to
exactly one specialist.

## Example questions

- Which product category has the highest total revenue?
- What is the average number of units sold per category?
- How much revenue came from the North region?
- What are the top five sales by revenue?
- What is the median and standard deviation of revenue across all sales?

## Requirements

- [UV](https://docs.astral.sh/uv/) for environment and dependency management.
- [Ollama](https://ollama.com/) running locally with the configured model
  pulled. The default is `llama3.2`, which you can pull with:

  ```bash
  ollama pull llama3.2
  ```

## Setup

```bash
# Install dependencies into a managed virtual environment
uv sync

# Copy the example environment file and adjust if needed
cp .env.example .env

# (Optional) regenerate the sample dataset
uv run python scripts/generate_data.py
```

Configuration is read from `.env` (or the environment) into a typed settings
object:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2` | Local Ollama model that drives the agent. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Address of the Ollama server. |
| `MAX_AGENT_STEPS` | `6` | Maximum reasoning and tool steps before the agent stops. |
| `DATA_PATH` | `./data/sales_data.csv` | Path to the sales dataset. |

## Running the agent

```bash
uv run python -m src.main --question "Which product category has the highest total revenue?"
```

The command prints the full reasoning trace followed by the final answer.

## Running the tests

```bash
uv run pytest
```

The tests exercise each tool against a small fixture dataset with known
expected values, so they run quickly and do not require Ollama.

## CI/CD with GitHub Actions

The workflow in `.github/workflows/tests.yml` has two jobs:

1. **CI (`test`).** On every push and pull request to `main`: install the
   dependencies with uv, lint with ruff, and run pytest. A failing check marks
   the commit red.
2. **CD (`publish`).** Only after the tests pass on a push to `main`: build the
   Docker image and publish it to the GitHub Container Registry as
   `ghcr.io/<owner>/react-data-agent`, tagged `latest` and with the commit id.

Run the published image (Ollama must be running on your machine):

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  ghcr.io/<owner>/react-data-agent --question "Which product category has the highest total revenue?"
```

## Worked example

Question:

```
Which product category has the highest total revenue?
```

A typical run produces a trace like the following. The exact wording of the
model's reasoning will vary, but the structure and the numbers are stable
because the tools run deterministic code against the dataset.

```
======================================================================
QUESTION: Which product category has the highest total revenue?
MODEL: llama3.2   MAX STEPS: 6
======================================================================

REASONING TRACE
----------------------------------------------------------------------
[1] USER QUESTION
    Which product category has the highest total revenue?

[2] AGENT REASONING -> tool call
    thought: I need total revenue per category, so I will aggregate revenue.
    calls: aggregate_by_category({'metric': 'revenue', 'operation': 'sum'})

[3] TOOL RESULT (aggregate_by_category)
    {"metric": "revenue", "operation": "sum", "by_category":
     {"Electronics": 406870.86, "Furniture": 332305.76, "Toys": 75566.06,
      "Clothing": 64499.54, "Groceries": 20549.73}}

[4] AGENT ANSWER
    Electronics has the highest total revenue at 406,870.86, ahead of
    Furniture at 332,305.76. The remaining categories trail well behind.

======================================================================
FINAL ANSWER
----------------------------------------------------------------------
Electronics has the highest total revenue at 406,870.86, ahead of
Furniture at 332,305.76. The remaining categories trail well behind.
======================================================================
```

The agent reasoned that it needed revenue grouped by category, called the
single tool that provides exactly that, read the ranked result, and answered
using the concrete figures rather than an estimate.

## Why the maximum step limit matters

A ReAct agent runs in a loop, and loops can fail to terminate. A model might
keep calling tools without ever deciding it has enough information, or bounce
between two tools indefinitely. Every one of those steps sends another request
to the model, which costs time and, on hosted models, money.

The `MAX_AGENT_STEPS` setting caps how many steps the agent may take. It is
translated into a LangGraph recursion limit and enforced by the graph at run
time. If the limit is reached, the run stops cleanly and reports that it did
not finish, rather than spinning forever. This single number is the primary
safety and cost control for the whole system, which is why it lives in the
configuration and not buried in code.

## Why tool descriptions matter

The model never sees the code inside a tool. It sees only the tool's name, its
docstring, and its input schema. Those three things are the entire basis on
which the model decides which tool to call and what arguments to pass.

This makes prompt and schema design a core engineering skill in an agent, not
an afterthought. A vague description such as "does category stuff" leaves the
model guessing, while a precise one such as "aggregate a metric per product
category, ranked highest to lowest" tells it exactly when the tool applies. The
same is true of the field descriptions in each Pydantic schema, which guide the
model toward valid arguments. In this project the descriptions are written to
be specific about purpose and inputs, because that is what steers the agent
toward the right tool for each question.

## Project structure

```
react-data-agent/
├── data/
│   └── sales_data.csv        # Sample sales dataset (180 rows)
├── scripts/
│   └── generate_data.py      # Deterministic dataset generator
├── src/
│   ├── config.py             # Typed application configuration
│   ├── tools.py              # The four analysis tools
│   ├── agent.py              # LangGraph ReAct agent construction
│   ├── multi_agent.py        # Supervisor + two specialist agents
│   └── main.py               # Command line interface
├── tests/
│   ├── test_tools.py         # Tool unit tests against a known fixture
│   └── test_multi_agent.py   # Supervisor graph tests with fake agents
├── .github/workflows/
│   └── tests.yml             # CI (ruff + pytest) and CD (Docker image)
├── Dockerfile                # Container image for the agent CLI
├── .env.example
├── pyproject.toml
└── README.md
```
