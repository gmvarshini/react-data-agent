Instructions for Claude Code
You are building a ReAct style agent for data analysis. Follow these steps in order.
Step 1: Initialize the project
Create the directory structure above, initialize Git, and create a `.gitignore` excluding `.env`, `__pycache__`, and `.venv`.
Step 2: Set up UV
Run `uv init` and `uv venv`. Add dependencies with `uv add langgraph langchain langchain-community pandas pydantic pydantic-settings python-dotenv` and dev dependencies `uv add --dev pytest ruff`.
Step 3: Prepare a sample dataset
Create or source `data/sales_data.csv` with at least 100 rows covering columns such as date, product category, region, units sold, and revenue, suitable for the agent to query and analyze.
Step 4: Build the configuration module
In `src/config.py`, define a Pydantic Settings class `AppConfig` loading `OLLAMA_MODEL` (default "llama3.2"), `MAX_AGENT_STEPS` (default 6), and `DATA_PATH` (default "./data/sales_data.csv"). Include full docstrings and type hints, plus `.env.example` and `.env`.
Step 5: Build the tools module
In `src/tools.py`, define at least four tools using Pydantic models for input schemas and clear docstrings: `filter_by_region`, `aggregate_by_category`, `compute_summary_statistics`, and `find_top_performers`. Each tool should operate on the sales dataframe and return structured results. Wrap each as a LangGraph compatible tool with a clear description so the agent can select appropriately.
Step 6: Build the agent module
In `src/agent.py`, construct a LangGraph agent using the ReAct pattern that has access to the tools defined above, includes a system prompt explaining its role as a data analysis assistant, and enforces a maximum step count from config to prevent infinite loops. Include a docstring explaining the graph structure, nodes, and edges.
Step 7: Build the main entry point
In `src/main.py`, write a command line interface accepting a `--question` argument, runs the agent on that question, and prints both the final answer and the full trace of reasoning steps and tool calls taken to reach it, formatted clearly for readability.
Step 8: Write tests
In `tests/test_tools.py`, write pytest tests for each tool verifying correct filtering, aggregation, and statistics computation against known expected values from a small fixture dataset.
Step 9: Write the README
Explain the ReAct pattern conceptually, the tools available to the agent, example questions the agent can answer, how to run it with UV, and a worked example showing the full reasoning trace for one question. Do not use em dashes. Keep the tone professional and clear for someone unfamiliar with agentic AI.
Step 10: Finalize and publish
Run `ruff check`, fix issues, commit with a descriptive message, and push to a new GitHub repository named `react-data-agent`. Report the repository URL.
---
Key Concepts to Highlight for Learning
Include a short section explaining why the maximum step limit matters for agent safety and cost control, and how tool descriptions directly influence which tool the agent selects, since prompt and schema design is itself a core skill being demonstrated here.