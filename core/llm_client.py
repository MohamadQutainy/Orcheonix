"""
Orcheonix - Shared OpenAI Client

Single OpenAI API client shared by all agents.
No agent should initialize its own raw OpenAI client.
"""

from openai import OpenAI, AsyncOpenAI
from core.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

_client_kwargs = {"api_key": OPENAI_API_KEY or "missing-openai-api-key"}
if OPENAI_BASE_URL:
    _client_kwargs["base_url"] = OPENAI_BASE_URL

# Synchronous client - used by most agents
client = OpenAI(**_client_kwargs)

# Async client - used by research_agent (3-step async pipeline)
async_client = AsyncOpenAI(**_client_kwargs)
