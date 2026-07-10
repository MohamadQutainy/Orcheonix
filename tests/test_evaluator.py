"""
Orcheonix — Evaluator Unit Tests

Tests all 5 evaluation metrics with known inputs.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.evaluator import AgentEvaluator


@pytest.fixture
def evaluator():
    return AgentEvaluator()

def test_confidence_empty(evaluator):
    assert evaluator.score_confidence("") == 0.0

def test_confidence_short(evaluator):
    assert evaluator.score_confidence("short text") == 0.0

def test_confidence_long_with_keywords(evaluator):
    text = "market analysis risk strategy data revenue compliance competitor research " * 50
    score = evaluator.score_confidence(text)
    assert 0.5 < score <= 1.0

def test_relevance_exact_match(evaluator):
    query = "Tesla stock price analysis"
    output = "The Tesla stock price analysis shows strong growth in the market."
    score = evaluator.score_relevance(output, query)
    assert score > 0.5

def test_relevance_no_match(evaluator):
    query = "Tesla stock price analysis"
    output = "The weather in London is rainy today with clouds and fog."
    score = evaluator.score_relevance(output, query)
    assert score < 0.5

def test_relevance_empty(evaluator):
    assert evaluator.score_relevance("", "query") == 0.0

def test_completeness_rich_output(evaluator):
    text = """# Analysis Report

## Market Overview
- Revenue: $1.5B
- Growth: 23%

| Metric | Value |
|--------|-------|
| Price  | $150  |

Competitor analysis shows strong market positioning.
"""
    score = evaluator.score_completeness(text, "competitor_agent")
    assert score > 0.5

def test_completeness_empty(evaluator):
    assert evaluator.score_completeness("", "competitor_agent") == 0.0

def test_hallucination_low_risk(evaluator):
    text = "According to [1] https://example.com, the market grew by 15% in 2025."
    score = evaluator.score_hallucination_risk(text)
    assert score < 0.3

def test_hallucination_high_risk(evaluator):
    text = "This will always work and is guaranteed to succeed 100% of the time without a doubt."
    score = evaluator.score_hallucination_risk(text)
    assert score > 0.3

def test_hallucination_empty(evaluator):
    assert evaluator.score_hallucination_risk("") == 1.0

def test_evaluate_agent_output(evaluator):
    result = evaluator.evaluate_agent_output(
        agent_name="competitor_agent",
        output="market analysis competitor data " * 100,
        query="Analyze competitors for Tesla",
        latency_seconds=4.5,
    )
    assert "scores" in result
    assert "overall" in result
    assert "flags" in result
    assert result["latency_seconds"] == 4.5
    assert 0.0 <= result["overall"] <= 1.0
