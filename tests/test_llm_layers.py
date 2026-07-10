"""Mock tests for OpenAI-backed LLM calls and parallel orchestration."""

import asyncio
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _chat_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_competitor_llm_complete_uses_openai_model(monkeypatch):
    from core.config import MODEL_NAME
    from orcheonix_agents import competitor_agent

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _chat_response("mocked strategic analysis")

    monkeypatch.setattr(competitor_agent.client.chat.completions, "create", fake_create)

    result = competitor_agent._llm_complete("Analyze this market")

    assert result == "mocked strategic analysis"
    assert captured["model"] == MODEL_NAME
    assert captured["messages"][0]["role"] == "system"
    assert "Analyze this market" in captured["messages"][1]["content"]


def test_planner_think_parses_openai_response(monkeypatch):
    from orchestrator.react_planner import ReActPlanner, PlannerState, client

    def fake_create(**kwargs):
        return _chat_response("A1_Competitor, A3_Research")

    monkeypatch.setattr(client.chat.completions, "create", fake_create)

    state = PlannerState(query="Analyze AI legal tech")
    planner = ReActPlanner(state)
    planner.think()

    assert state.agents_to_run == ["A1_Competitor", "A2_Finance", "A3_Research"]


def test_observe_runs_upstream_agents_in_parallel(monkeypatch):
    from orchestrator.react_planner import ReActPlanner, PlannerState

    async def fake_upstream(query):
        await asyncio.sleep(0.05)
        return "market competitor research data revenue compliance strategy " * 20

    async def fake_strategy(query, context):
        return "strategy risk compliance roadmap budget market analysis " * 20

    state = PlannerState(
        query="Analyze AI legal tech",
        agents_to_run=["A1_Competitor", "A2_Finance", "A3_Research"],
    )
    planner = ReActPlanner(state)
    planner.agent_funcs = {
        "A1_Competitor": ("Competitor", fake_upstream, False, "competitor_agent"),
        "A2_Finance": ("Finance", fake_upstream, False, "finance_agent"),
        "A3_Research": ("Research", fake_upstream, False, "research_agent"),
        "A4_Strategy": ("Strategy", fake_strategy, True, "strategy_agent"),
    }

    start = time.perf_counter()
    asyncio.run(planner.observe())
    elapsed = time.perf_counter() - start

    assert {"A1_Competitor", "A2_Finance", "A3_Research", "A4_Strategy"} <= set(state.results)
    assert elapsed < 0.14
