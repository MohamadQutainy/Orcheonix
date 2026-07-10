"""
Orcheonix - Centralized Configuration

All settings (model name, API keys, thresholds) live here.
Agents and orchestrator import from this module only.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM (OpenAI API)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-nano")

# External API keys
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Runtime thresholds
MIN_RESULT_LEN = int(os.getenv("MIN_RESULT_LEN", "200"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

# Serper API
SERPER_URL = "https://google.serper.dev/search"

# Fast-agent behavior. This replaces the old local-model /no_think control token.
NO_THINK_PREFIX = os.getenv(
    "FAST_AGENT_INSTRUCTION",
    "Be concise, factual, and tool-grounded. Do not reveal private chain-of-thought.",
)


def missing_required_settings(require_web_tools: bool = True) -> list[str]:
    """Return missing environment variables needed for a full production run."""
    required = ["OPENAI_API_KEY"]
    if require_web_tools:
        required.extend(["FIRECRAWL_API_KEY", "SERPER_API_KEY"])
    return [name for name in required if not os.getenv(name)]
