"""
Orcheonix - Agent Unit Tests

Tests that agents can be imported and expose expected interfaces.
Does NOT require live OpenAI or web API calls.
"""

import sys
import os

# Add orcheonix root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_competitor_agent_importable():
    from orcheonix_agents.competitor_agent import run_competitor_analysis
    assert callable(run_competitor_analysis)


def test_finance_agent_importable():
    from orcheonix_agents.finance_agent import run_market_analysis
    assert callable(run_market_analysis)


def test_research_agent_importable():
    from orcheonix_agents.research_agent import run_research_process
    assert callable(run_research_process)


def test_strategy_agent_importable():
    from orcheonix_agents.strategy_agent import run_strategy_consultation
    assert callable(run_strategy_consultation)


def test_competitor_agent_has_schema():
    from orcheonix_agents.competitor_agent import CompetitorDataSchema
    assert hasattr(CompetitorDataSchema, "model_json_schema")


def test_llm_client_importable():
    from core.llm_client import client, MODEL_NAME
    assert client is not None
    assert MODEL_NAME == "gpt-5.4-nano" or isinstance(MODEL_NAME, str)


def test_config_importable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from core.config import MODEL_NAME, missing_required_settings
    assert isinstance(MODEL_NAME, str)
    assert "OPENAI_API_KEY" in missing_required_settings(require_web_tools=False)
