FROM python:3.12-slim

# Install UV, the fast dependency manager, from its official static binary.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first so this layer is cached when only code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the application code and the sample dataset.
COPY src ./src
COPY data ./data

# The agent needs a running Ollama server. Inside a container, "localhost"
# is the container itself, so by default we point at the Docker host.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Pass the question when running, for example:
#   docker run --rm ghcr.io/<user>/react-data-agent --question "..."
ENTRYPOINT ["uv", "run", "--no-dev", "python", "-m", "src.main"]
