"""
Deep Research Engine
----------------------
Core logic of the Deep Research Engine — NO Streamlit UI calls at module level.
Safe to import from react_planner.py without side effects.
"""

from typing import Dict, Any
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from firecrawl import FirecrawlApp
from agents.tool import function_tool

from core.llm_client import async_client, MODEL_NAME
from core.config import FIRECRAWL_API_KEY, NO_THINK_PREFIX
from core.logger import get_logger, log_agent_run
import time

set_tracing_disabled(True)
logger = get_logger("ResearchAgent")

def get_model() -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=async_client,
    )

@function_tool
async def deep_research(
    query: str,
    max_depth: int = 3,
    time_limit: int = 180,
    max_urls: int = 10,
) -> Dict[str, Any]:
    """Perform comprehensive web research using Firecrawl's agent endpoint."""
    try:
        firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        research_schema = {
            "type": "object",
            "properties": {
                "final_analysis": {
                    "type": "string",
                    "description": "Comprehensive synthesis of key insights and conclusions on the topic"
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"}
                        }
                    },
                    "description": "List of source URLs used to build the analysis"
                }
            },
            "required": ["final_analysis", "sources"]
        }

        response = firecrawl_app.agent(
            prompt=(
                f"Research the following topic thoroughly by searching and reading "
                f"multiple web sources, then synthesize the findings: {query}"
            ),
            schema=research_schema,
            max_credits=max_urls,
            timeout=time_limit,
        )

        if not response.success or not response.data:
            return {"error": response.error or "No data returned", "success": False}

        data = response.data
        sources = data.get("sources", [])

        return {
            "success": True,
            "final_analysis": data.get("final_analysis", ""),
            "sources_count": len(sources),
            "sources": sources,
        }
    except Exception as e:
        logger.error(f"Deep research tool error: {e}")
        return {"error": str(e), "success": False}


def build_research_agent() -> Agent:
    return Agent(
        name="research_agent",
        instructions=f"{NO_THINK_PREFIX}\nYou are a precise research assistant. Your sole job is to gather raw, factual data.\n"
        "1. Use the deep_research tool to collect raw telemetry and actual verified data on the topic.\n"
        "2. Never extrapolate, hallucinate, or predict; just structure what you discover on the web.\n"
        "3. Provide clear citations for every source returned by the tool.",
        tools=[deep_research],
        model=get_model(),
    )


def build_elaboration_agent() -> Agent:
    return Agent(
        name="elaboration_agent",
        instructions=f"{NO_THINK_PREFIX}\nYou are a premium corporate technical analyst. Expand the raw research report:\n"
        "1. Use markdown tables for performance and architectural comparisons.\n"
        "2. Provide Mermaid.js diagrams where relevant (```mermaid ... ```).\n"
        "3. Focus on architectural clarity and technical realism.",
        model=get_model(),
    )


def build_critic_agent() -> Agent:
    return Agent(
        name="critic_agent",
        instructions=f"{NO_THINK_PREFIX}\nYou are a strict Chief Technology Editor. Sanitize and verify the report:\n"
        "1. Remove placeholder URLs (example.com, arxiv.org/abs/xxxx, etc.).\n"
        "2. Ensure all diagrams use valid Mermaid.js syntax.\n"
        "3. Keep tone academic, cold, and professional. Remove buzzwords.\n"
        "4. Output the final cleanest markdown report.",
        model=get_model(),
    )


async def run_research_process(topic: str) -> str:
    start_time = time.time()
    try:
        research_agent    = build_research_agent()
        elaboration_agent = build_elaboration_agent()
        critic_agent      = build_critic_agent()

        research_result = await Runner.run(research_agent, topic)
        initial_report  = research_result.final_output

        elaboration_input  = f"TOPIC: {topic}\n\nRAW FACTUAL DATA:\n{initial_report}"
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report    = elaboration_result.final_output

        critic_input  = f"Review, clean up, and sanitize this report:\n\n{enhanced_report}"
        critic_result = await Runner.run(critic_agent, critic_input)

        log_agent_run(logger, "research_agent", topic, time.time() - start_time)
        return critic_result.final_output
    except Exception as e:
        log_agent_run(logger, "research_agent", topic, time.time() - start_time, error=str(e))
        raise
