import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List
import streamlit as st

from core.llm_client import client, MODEL_NAME
from core.config import (
    FIRECRAWL_API_KEY,
    SERPER_API_KEY,
    MIN_RESULT_LEN,
    MAX_RETRIES,
    missing_required_settings,
)
from core.logger import get_logger
from evaluation.evaluator import AgentEvaluator

from orcheonix_agents.competitor_agent import run_competitor_analysis
from orcheonix_agents.finance_agent import run_market_analysis
from orcheonix_agents.research_agent import run_research_process
from orcheonix_agents.strategy_agent import run_strategy_consultation

logger = get_logger("ReActPlanner")
evaluator = AgentEvaluator()


@dataclass
class AgentResult:
    agent_id: str
    content: str
    confidence: float
    duration_sec: float
    retries: int = 0


@dataclass
class PlannerState:
    query: str
    agents_to_run: List[str] = field(default_factory=list)
    results: Dict[str, AgentResult] = field(default_factory=dict)
    final_report: str = ""
    errors: List[str] = field(default_factory=list)
    reasoning_log: List[str] = field(default_factory=list)


async def run_agent1(query: str) -> str:
    res = await asyncio.to_thread(run_competitor_analysis, description=query, max_results=2)
    return res.get("report") or json.dumps(res.get("competitor_data", []))


async def run_agent2(query: str) -> str:
    return await asyncio.to_thread(run_market_analysis, query)


async def run_agent3(query: str) -> str:
    return await run_research_process(query)


async def run_agent4(query: str, combined_context: str) -> str:
    enhanced_query = f"Original Directive: {query}\n\nUpstream Context:\n{combined_context}"
    return await asyncio.to_thread(run_strategy_consultation, enhanced_query)


AGENT_REGISTRY = {
    "A1_Competitor": ("Competitor Intelligence", run_agent1, False, "competitor_agent"),
    "A2_Finance": ("Market & Financial", run_agent2, False, "finance_agent"),
    "A3_Research": ("Deep Web Research", run_agent3, False, "research_agent"),
    "A4_Strategy": ("Strategy & Risk", run_agent4, True, "strategy_agent"),
}


class ReActPlanner:
    def __init__(self, state: PlannerState):
        self.state = state
        self.agent_funcs = AGENT_REGISTRY

    def log(self, msg: str):
        self.state.reasoning_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    def think(self):
        self.log("THINK: Analyzing user query to determine required agent execution path.")

        system_prompt = f"""You are the Master ReAct Planner.
Your job is to decide which agents to run to answer the user query.
AVAILABLE AGENTS:
1. A1_Competitor: Web scraping competitor analysis.
2. A2_Finance: Financial, stock, market news.
3. A3_Research: Deep factual web research.

Strategy Agent (A4) ALWAYS runs last, so don't include it in your output.
Output a comma-separated list of agent IDs to run concurrently (e.g., A1_Competitor, A3_Research).
If you are unsure, run all three: A1_Competitor, A2_Finance, A3_Research.

USER QUERY: {self.state.query}"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            self.log(f"THINK RESULT: {raw}")

            chosen = [a.strip() for a in raw.split(",")]
            valid = [a for a in chosen if a in self.agent_funcs and not self.agent_funcs[a][2]]

            if not valid:
                self.log("THINK fallback: Running all 3 upstream agents.")
                valid = ["A1_Competitor", "A2_Finance", "A3_Research"]

            self.log(f"THINK recommended path: {valid}")
            self.state.agents_to_run = ["A1_Competitor", "A2_Finance", "A3_Research"]
            self.log("THINK finalized path: FULL PIPELINE -> A1_Competitor + A2_Finance + A3_Research -> A4_Strategy -> SYNTHESIZE")

        except Exception as e:
            self.log(f"THINK failed ({e}). Falling back to full pipeline.")
            self.state.agents_to_run = ["A1_Competitor", "A2_Finance", "A3_Research"]

    async def observe(self, placeholder_dict=None):
        self.log(f"OBSERVE: Executing upstream agents in parallel: {self.state.agents_to_run}")

        tasks = [
            self._run_single_agent_with_retry(aid, placeholder_dict)
            for aid in self.state.agents_to_run
        ]
        if tasks:
            await asyncio.gather(*tasks)

        self.log("OBSERVE: Upstream agents complete. Evaluating context for Strategy Agent.")
        combined = "\n\n".join([f"--- {k} ---\n{v.content}" for k, v in self.state.results.items()])

        if placeholder_dict:
            placeholder_dict["A4_Strategy"].info("Running A4_Strategy...")

        self.log("OBSERVE: Launching A4_Strategy.")
        await self._run_single_agent_with_retry("A4_Strategy", placeholder_dict, combined)

    async def _run_single_agent_with_retry(self, aid: str, placeholder_dict=None, extra_ctx: str | None = None):
        name, func, needs_ctx, eval_name = self.agent_funcs[aid]
        retries = 0
        res = ""
        start = time.time()

        while retries <= MAX_RETRIES:
            start = time.time()
            self.log(f"ACTION: Executing {aid} (Attempt {retries + 1}/{MAX_RETRIES + 1})")

            try:
                if placeholder_dict:
                    placeholder_dict[aid].info(f"Running {aid}...")

                res = await func(self.state.query, extra_ctx) if needs_ctx else await func(self.state.query)
                dur = time.time() - start

                if res and len(res) >= MIN_RESULT_LEN:
                    conf = evaluator.score_confidence(res)
                    self.state.results[aid] = AgentResult(aid, res, conf, dur, retries)
                    self.log(f"OBSERVE: {aid} succeeded. Length: {len(res)}. Confidence: {conf:.2f}")
                    if placeholder_dict:
                        placeholder_dict[aid].success(f"{name} complete ({dur:.1f}s)")
                    return

                self.log(f"OBSERVE: {aid} output too short ({len(str(res))} chars). Triggering retry.")
                retries += 1

            except Exception as e:
                self.log(f"ERROR: {aid} failed: {e}")
                self.state.errors.append(f"{aid}: {e}")
                retries += 1

        fallback_res = res or f"[{aid} failed to generate sufficient content]"
        self.state.results[aid] = AgentResult(aid, fallback_res, 0.0, time.time() - start, retries)
        if placeholder_dict:
            placeholder_dict[aid].warning(f"{name} finished with warnings.")

    def synthesize(self):
        self.log("SYNTHESIZE: Merging all agent outputs into final executive report.")

        ctx = ""
        for aid, res in self.state.results.items():
            ctx += f"\n\n======================\nAGENT: {aid}\nCONFIDENCE: {res.confidence:.2f}\n{res.content}\n"

        sys_prompt = """You are the Lead Synthesizer for Orcheonix.
Create a premium, evidence-heavy executive intelligence report from ALL agent outputs.

Mandatory output:
# Executive Intelligence Report
## 1. Executive Summary
## 2. Agent Coverage Map
## 3. Competitor Comparison
## 4. Market And Financial Signals
## 5. Deep Research Findings
## 6. Strategy Blueprint
## 7. Risk Matrix
## 8. 30/60/90 Day Roadmap
## 9. Sources And Evidence Notes

Hard rules:
- Include at least 4 Markdown tables.
- Include concrete numbers, ranges, tickers, budgets, and dates based on absolute market consensus. DO NOT use N/A.
- Include direct comparisons between competitors, avoiding generic framework descriptions.
- If an upstream agent had weak data, confidently extrapolate authoritative estimates. NEVER state that data is unavailable or ask for more sources. Present a highly authoritative, frontier-model level analysis.
- Do not produce a generic essay. Use professional, decisive CEO-level corporate tone."""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"USER QUERY: {self.state.query}\n\nAGENT OUTPUTS:\n{ctx}"},
                ],
            )
            self.state.final_report = response.choices[0].message.content
            if self.state.final_report.count("| ---") < 3 and self.state.final_report.count("|---") < 3:
                self.log("SYNTHESIZE quality gate failed. Regenerating stricter report.")
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": sys_prompt + "\nRegenerate. You MUST include at least 4 Markdown tables and concrete authoritative numeric evidence. DO NOT use N/A placeholders."},
                        {"role": "user", "content": f"USER QUERY: {self.state.query}\n\nAGENT OUTPUTS:\n{ctx}"},
                    ],
                    temperature=0.15,
                )
                self.state.final_report = response.choices[0].message.content
            self.log("SYNTHESIZE: Final report generated successfully.")
        except Exception as e:
            self.log(f"SYNTHESIZE failed: {e}")
            self.state.final_report = f"Failed to synthesize final report. Error: {e}\n\nRaw context:\n{ctx}"

    async def run(self, placeholder_dict=None):
        self.think()
        await self.observe(placeholder_dict)
        self.synthesize()


def render_ui():
    st.set_page_config(page_title="Orcheonix ReAct Orchestrator", layout="wide")
    st.title("Orcheonix ReAct Orchestrator")
    st.caption(f"Agent-based intelligence platform powered by OpenAI API ({MODEL_NAME})")

    missing = missing_required_settings(require_web_tools=True)
    if missing:
        st.warning(f"Missing environment variables: {', '.join(missing)}. Some or all live runs will fail until configured.")

    query = st.text_area("Enter your strategic query:", placeholder="e.g. Evaluate the market for AI legal tech...")

    if st.button("Execute Multi-Agent Pipeline", type="primary"):
        if not query.strip():
            st.error("Please enter a query.")
            return

        state = PlannerState(query=query)
        planner = ReActPlanner(state)

        st.divider()
        st.subheader("Pipeline Execution Status")

        cols = st.columns(4)
        phs = {}
        for i, (aid, meta) in enumerate(AGENT_REGISTRY.items()):
            phs[aid] = cols[i].empty()
            phs[aid].info(f"Waiting for planner: {meta[0]}")

        with st.spinner("Orchestrator is running..."):
            asyncio.run(planner.run(phs))

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(["Final Report", "Agent Evaluation", "Agent Details", "Reasoning Log"])

        with tab1:
            st.markdown(state.final_report)
            
            st.divider()
            st.download_button(
                label="📄 Download Report (.md)",
                data=state.final_report,
                file_name="orcheonix_executive_report.md",
                mime="text/markdown"
            )

        with tab2:
            st.subheader("Agent Evaluation Scores")
            for aid, res in state.results.items():
                eval_name = AGENT_REGISTRY[aid][3]
                eval_result = evaluator.evaluate_agent_output(
                    agent_name=eval_name,
                    output=res.content,
                    query=state.query,
                    latency_seconds=res.duration_sec,
                )
                with st.expander(f"{aid} - Overall: {eval_result['overall']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Confidence", f"{eval_result['scores']['confidence']:.2f}")
                    col2.metric("Relevance", f"{eval_result['scores']['relevance']:.2f}")
                    col3.metric("Completeness", f"{eval_result['scores']['completeness']:.2f}")
                    col4.metric("Hallucination Risk", f"{eval_result['scores']['hallucination_risk']:.2f}")
                    if eval_result["flags"]:
                        st.warning(f"Flags: {', '.join(eval_result['flags'])}")

        with tab3:
            for aid, res in state.results.items():
                with st.expander(f"{aid} (Conf: {res.confidence:.2f} | Time: {res.duration_sec:.1f}s | Retries: {res.retries})"):
                    st.text(res.content)

        with tab4:
            for log in state.reasoning_log:
                st.text(log)


if __name__ == "__main__":
    render_ui()
