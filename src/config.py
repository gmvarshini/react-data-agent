"""Application configuration for the ReAct data analysis agent.

This module centralises all runtime configuration in a single typed object.
Values are read from environment variables (optionally provided through a
``.env`` file) and fall back to sensible defaults when not set. Keeping
configuration in one place makes the agent easy to reconfigure without
touching application code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Typed application settings loaded from the environment or a ``.env`` file.

    Attributes:
        ollama_model: Name of the Ollama chat model the agent should use. The
            model must already be pulled locally (for example via
            ``ollama pull llama3.2``). Read from the ``OLLAMA_MODEL``
            environment variable.
        max_agent_steps: Maximum number of reasoning/tool steps the agent may
            take before it is forced to stop. This is a safety limit that
            guards against infinite reasoning loops and runaway token cost.
            Read from the ``MAX_AGENT_STEPS`` environment variable.
        ollama_base_url: Address of the Ollama server. The default works when
            Ollama runs on the same machine. Inside Docker, point it at the
            host, for example ``http://host.docker.internal:11434``. Read from
            the ``OLLAMA_BASE_URL`` environment variable.
        data_path: Filesystem path to the sales dataset (a CSV file) that the
            tools load and query. Read from the ``DATA_PATH`` environment
            variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = Field(
        default="llama3.2",
        validation_alias="OLLAMA_MODEL",
        description="Name of the local Ollama model to drive the agent.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Address of the Ollama server.",
    )
    max_agent_steps: int = Field(
        default=6,
        validation_alias="MAX_AGENT_STEPS",
        ge=1,
        description="Maximum reasoning/tool steps before the agent stops.",
    )
    data_path: str = Field(
        default="./data/sales_data.csv",
        validation_alias="DATA_PATH",
        description="Path to the sales dataset CSV file.",
    )


def get_config() -> AppConfig:
    """Build and return an :class:`AppConfig` instance.

    Returns:
        A fully populated :class:`AppConfig` object with values resolved from
        the environment, a ``.env`` file, or the declared defaults.
    """

    return AppConfig()
