"""
Competitor Intelligence Agent
------------------------------
Part of the Orcheonix multi-agent system (Agent 1 of 4).

INPUT  : Company URL or short description
OUTPUT : Structured competitor comparison table + strategic analysis report

Pipeline:
  1. OpenAI       -> identifies likely direct competitors
  2. Serper       -> resolves official domains and gathers search snippets
  3. Firecrawl    -> extracts structured data when credits are available
  4. OpenAI       -> produces a table-heavy strategic competitor report

If Firecrawl is unavailable or out of credits, the agent falls back to Serper
snippets so the pipeline still returns useful competitor intelligence.
"""

import json
import time
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field

from core.config import FIRECRAWL_API_KEY, NO_THINK_PREFIX, SERPER_API_KEY, SERPER_URL
from core.llm_client import MODEL_NAME, client
from core.logger import get_logger, log_agent_run

logger = get_logger("CompetitorAgent")


class CompetitorDataSchema(BaseModel):
    company_name: str = Field(description="Name of the company")
    pricing: str = Field(description="Pricing details, tiers, and plans")
    key_features: List[str] = Field(description="Main features and capabilities")
    tech_stack: List[str] = Field(description="Technologies and tools used")
    marketing_focus: str = Field(description="Main marketing angles and target audience")
    customer_feedback: str = Field(description="Customer testimonials and feedback")


def _llm_complete(prompt: str, temperature: float = 0.2) -> str:
    """Direct LLM API call using the shared OpenAI client."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": NO_THINK_PREFIX},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


NON_COMPETITOR_DOMAINS = {
    "wikipedia", "semrush", "similarweb", "owler", "crunchbase",
    "linkedin", "facebook", "twitter", "x", "instagram", "youtube",
    "pinterest", "reddit", "quora", "medium", "tiktok", "glassdoor",
    "indeed", "g2", "capterra", "trustpilot", "pitchbook", "zoominfo",
    "builtwith", "alexa", "globaldata", "investopedia", "businessinsider",
    "marketwatch", "yahoo", "google", "fool", "nasdaq", "yelp",
}


def _root_domain(netloc: str) -> str:
    netloc = netloc.replace("www.", "")
    parts = netloc.split(".")
    return parts[-2] if len(parts) >= 2 else netloc


def _serper_search(query: str, num: int = 5) -> list[dict]:
    if not SERPER_API_KEY:
        return []
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    resp = requests.post(SERPER_URL, headers=headers, json={"q": query, "num": num}, timeout=20)
    resp.raise_for_status()
    return resp.json().get("organic", [])


def find_competitor_urls(url: str = "", description: str = "", max_results: int = 3) -> List[str]:
    if not url and not description:
        raise ValueError("Provide either a URL or a description.")

    excluded_domain = _root_domain(urlparse(url).netloc) if url else None
    subject = url if url else description

    naming_prompt = f"""List exactly {max_results} real, well-known direct competitor companies of: {subject}

Rules:
- Return ONLY company names, one per line
- No URLs, no explanations, no numbering, no extra text
- Must be real companies that actually compete in the same market"""

    raw = _llm_complete(naming_prompt)
    company_names = [line.strip("- *.").strip() for line in raw.strip().split("\n") if line.strip()]
    company_names = company_names[:max_results]

    urls: List[str] = []
    seen_domains = set()
    if excluded_domain:
        seen_domains.add(excluded_domain)

    for name in company_names:
        try:
            for r in _serper_search(f"{name} official website", num=5):
                link = r.get("link", "")
                if not link.startswith("http"):
                    continue
                parsed = urlparse(link)
                root_domain = _root_domain(parsed.netloc)

                if root_domain in seen_domains or root_domain in NON_COMPETITOR_DOMAINS:
                    continue

                seen_domains.add(root_domain)
                urls.append(f"{parsed.scheme}://{parsed.netloc.replace('www.', '')}")
                break
        except Exception as e:
            logger.warning(f"Error resolving competitor URL for {name}: {e}")
            continue

        if len(urls) >= max_results:
            break

    return urls


def _fallback_competitor_info(competitor_url: str) -> Optional[dict]:
    """Use Serper snippets + OpenAI when Firecrawl cannot extract a site."""
    try:
        domain = urlparse(competitor_url).netloc.replace("www.", "")
        results = _serper_search(
            f"{domain} pricing features customers reviews product technology stack",
            num=8,
        )
        snippets = [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in results
            if r.get("snippet") or r.get("title")
        ]
        if not snippets:
            return None

        prompt = f"""Build competitor intelligence from these search snippets.
Return ONLY valid JSON with keys:
company_name, pricing, key_features, tech_stack, marketing_focus, customer_feedback.
Use 'Not publicly listed' when pricing is unavailable. Include only facts supported by snippets.

Competitor URL: {competitor_url}
Snippets:
{json.dumps(snippets, indent=2)}"""
        text = _llm_complete(prompt)
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        parsed = json.loads(text)
        return {
            "competitor_url": competitor_url,
            "company_name": parsed.get("company_name", domain),
            "pricing": parsed.get("pricing", "Not publicly listed"),
            "key_features": list(parsed.get("key_features", []))[:6],
            "tech_stack": list(parsed.get("tech_stack", []))[:6],
            "marketing_focus": parsed.get("marketing_focus", "N/A"),
            "customer_feedback": parsed.get("customer_feedback", "N/A"),
            "source_mode": "serper_fallback",
            "sources": [s["link"] for s in snippets if s.get("link")][:5],
        }
    except Exception as e:
        logger.warning(f"Fallback competitor extraction failed for {competitor_url}: {e}")
        return None


def extract_competitor_info(competitor_url: str) -> Optional[dict]:
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        extraction_prompt = (
            "Extract: company name, pricing details and plans, key features, "
            "technology stack, marketing focus and target audience, "
            "customer feedback and testimonials."
        )
        response = app.extract(
            [competitor_url],
            prompt=extraction_prompt,
            schema=CompetitorDataSchema.model_json_schema(),
        )

        if not (hasattr(response, "success") and response.success):
            logger.warning(f"Firecrawl extraction failed for {competitor_url}; using Serper fallback.")
            return _fallback_competitor_info(competitor_url)
        if not (hasattr(response, "data") and response.data):
            logger.warning(f"Firecrawl returned no data for {competitor_url}; using Serper fallback.")
            return _fallback_competitor_info(competitor_url)

        d = response.data
        get = (lambda k, default: d.get(k, default)) if isinstance(d, dict) else (lambda k, default: getattr(d, k, default))

        return {
            "competitor_url": competitor_url,
            "company_name": get("company_name", "N/A"),
            "pricing": get("pricing", "N/A"),
            "key_features": get("key_features", [])[:6],
            "tech_stack": get("tech_stack", [])[:6],
            "marketing_focus": get("marketing_focus", "N/A"),
            "customer_feedback": get("customer_feedback", "N/A"),
            "source_mode": "firecrawl",
        }
    except Exception as e:
        logger.warning(f"Firecrawl failed for {competitor_url}: {e}. Using Serper fallback.")
        return _fallback_competitor_info(competitor_url)


def generate_analysis_report(competitor_data: list, subject: str = "") -> str:
    formatted_data = json.dumps(competitor_data, indent=2)

    return _llm_complete(
        f"""You are a senior competitive intelligence analyst.
Analyze the competitor data for this user objective: {subject}

Competitor data:
{formatted_data}

Return a rich Markdown report with these exact sections:
## Competitor Snapshot Table
A markdown table with columns: Company, URL, Pricing, Key Features, Positioning, Evidence Quality.

## Feature And Positioning Comparison
A markdown table comparing at least 5 features/capabilities across competitors.

## Market Gaps And Opportunities
At least 5 bullets with concrete opportunity angles.

## Pricing And Packaging Implications
Use available pricing evidence. If pricing is not public, state that and infer packaging options carefully.

## Differentiation Strategy
Specific product, GTM, trust/compliance, and AI engineering recommendations.

## Sources
List source URLs when available.

Rules:
- Include numbers, counts, table rows, and comparison language.
- Do not return a generic paragraph.
- Do not invent exact prices if they are not in the data; say Not publicly listed.
""",
        temperature=0.25,
    )


def run_competitor_analysis(url: str = "", description: str = "", max_results: int = 3) -> dict:
    start_time = time.time()
    try:
        urls = find_competitor_urls(url=url, description=description, max_results=max_results)
        if not urls:
            report = generate_analysis_report([], description or url)
            return {"competitor_urls": [], "competitor_data": [], "report": report}

        competitor_data = []
        for comp_url in urls:
            info = extract_competitor_info(comp_url)
            if info:
                competitor_data.append(info)

        report = generate_analysis_report(competitor_data, description or url)

        log_agent_run(logger, "competitor_agent", description or url, time.time() - start_time)
        return {
            "competitor_urls": urls,
            "competitor_data": competitor_data,
            "report": report,
        }
    except Exception as e:
        log_agent_run(logger, "competitor_agent", description or url, time.time() - start_time, error=str(e))
        raise


def render_ui():
    st.set_page_config(page_title="Competitor Intelligence Agent", layout="wide")

    if not FIRECRAWL_API_KEY or not SERPER_API_KEY:
        st.error("Missing API keys. Check .env file.")
        st.stop()

    st.title("Competitor Intelligence Agent")
    st.caption("Agent 1 of 4 - Orcheonix multi-agent system")
    st.info("Provide a company URL or a short description to find and analyze competitors.")

    url = st.text_input("Company URL (optional):")
    description = st.text_area("Company description (used if no URL):")

    if st.button("Analyze Competitors", type="primary"):
        if not url and not description:
            st.warning("Please provide a URL or a description.")
            return

        with st.spinner("Finding and analyzing competitors..."):
            try:
                result = run_competitor_analysis(url=url, description=description)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        if result["competitor_data"]:
            table = [{
                "Company": c["company_name"],
                "URL": c["competitor_url"],
                "Pricing": c["pricing"][:120],
                "Key Features": ", ".join(c["key_features"][:3]),
                "Source Mode": c.get("source_mode", "unknown"),
            } for c in result["competitor_data"]]
            st.dataframe(pd.DataFrame(table), use_container_width=True)

        st.markdown(result["report"])


if __name__ == "__main__":
    render_ui()