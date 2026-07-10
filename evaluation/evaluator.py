"""
Orcheonix — Agent Output Evaluator

Measures the quality of multi-agent output without requiring ground truth labels.
All evaluation results are logged to logs/eval_log.jsonl.
"""

import os
import json
import re
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
EVAL_LOG_FILE = os.path.join(LOG_DIR, "eval_log.jsonl")


class AgentEvaluator:
    """
    Evaluates agent outputs using heuristic-based metrics.
    No LLM calls required — pure deterministic scoring.
    """

    EXPECTED_MARKERS = {
        "competitor_agent": ["competitor", "pricing", "features", "analysis", "market"],
        "finance_agent": ["price", "revenue", "stock", "market", "financial"],
        "research_agent": ["research", "findings", "sources", "analysis", "data"],
        "strategy_agent": ["strategy", "risk", "compliance", "roadmap", "budget"],
    }

    HALLUCINATION_TRIGGERS = [
        "100%", "always", "never", "guaranteed", "absolutely",
        "without a doubt", "certainly", "undoubtedly", "impossible",
        "every single", "zero chance", "no way",
    ]

    def score_confidence(self, text: str) -> float:
        if not text or len(text) < 200:
            return 0.0
        score = min(len(text) / 3000, 1.0) * 0.6
        keywords = [
            "analysis", "market", "risk", "strategy", "data",
            "revenue", "compliance", "competitor", "research",
        ]
        hits = sum(1 for k in keywords if k.lower() in text.lower())
        score += (hits / len(keywords)) * 0.4
        return round(min(score, 1.0), 2)

    def score_relevance(self, text: str, query: str) -> float:
        if not text or not query:
            return 0.0
        query_words = [
            w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 3
        ]
        if not query_words:
            return 0.5
        text_lower = text.lower()
        hits = sum(1 for w in query_words if w in text_lower)
        return round(min(hits / len(query_words), 1.0), 2)

    def score_completeness(self, text: str, agent_name: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        structure_score = 0.0
        has_headings = bool(re.search(r'^#{1,3}\s', text, re.MULTILINE))
        has_bullets = bool(re.search(r'^[\-\*]\s', text, re.MULTILINE))
        has_tables = "|" in text and "---" in text
        has_numbers = bool(re.search(r'\$[\d,.]+|\d+%', text))

        structure_score += 0.15 if has_headings else 0.0
        structure_score += 0.10 if has_bullets else 0.0
        structure_score += 0.15 if has_tables else 0.0
        structure_score += 0.10 if has_numbers else 0.0

        markers = self.EXPECTED_MARKERS.get(agent_name, [])
        if markers:
            marker_hits = sum(1 for m in markers if m in text_lower)
            marker_score = marker_hits / len(markers)
        else:
            marker_score = 0.5

        return round(min(structure_score + marker_score * 0.5, 1.0), 2)

    def score_hallucination_risk(self, text: str) -> float:
        if not text:
            return 1.0
        text_lower = text.lower()
        trigger_count = sum(
            1 for t in self.HALLUCINATION_TRIGGERS if t.lower() in text_lower
        )
        trigger_risk = min(trigger_count * 0.15, 0.6)
        has_citations = bool(re.search(r'\[\d+\]|https?://', text))
        source_risk = 0.0 if has_citations else 0.2
        length_risk = 0.2 if len(text) < 500 else 0.0

        return round(min(trigger_risk + source_risk + length_risk, 1.0), 2)

    def evaluate_agent_output(
        self,
        agent_name: str,
        output: str,
        query: str,
        latency_seconds: float = 0.0,
    ) -> dict:
        confidence = self.score_confidence(output)
        relevance = self.score_relevance(output, query)
        completeness = self.score_completeness(output, agent_name)
        hallucination_risk = self.score_hallucination_risk(output)

        overall = round(
            confidence * 0.25
            + relevance * 0.25
            + completeness * 0.25
            + (1.0 - hallucination_risk) * 0.25,
            2,
        )

        flags = []
        if confidence < 0.3:
            flags.append("low confidence")
        if relevance < 0.3:
            flags.append("low relevance")
        if completeness < 0.3:
            flags.append("incomplete output")
        if hallucination_risk > 0.5:
            flags.append("high hallucination risk")

        result = {
            "agent": agent_name,
            "query": query[:200],
            "scores": {
                "confidence": confidence,
                "relevance": relevance,
                "completeness": completeness,
                "hallucination_risk": hallucination_risk,
            },
            "latency_seconds": round(latency_seconds, 2),
            "overall": overall,
            "flags": flags,
        }
        self._log_evaluation(result)
        return result

    def evaluate_full_report(
        self, report: str, query: str, agent_outputs: dict
    ) -> dict:
        agent_evals = {}
        for agent_name, output_data in agent_outputs.items():
            output_text = output_data.get("content", "") if isinstance(output_data, dict) else str(output_data)
            latency = output_data.get("latency", 0.0) if isinstance(output_data, dict) else 0.0
            agent_evals[agent_name] = self.evaluate_agent_output(
                agent_name, output_text, query, latency
            )
        report_eval = self.evaluate_agent_output(
            "final_report", report, query
        )
        return {
            "agent_evaluations": agent_evals,
            "report_evaluation": report_eval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _log_evaluation(self, result: dict):
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result,
            }
            with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
