# fed_agent/agent.py

from typing import Any, Dict
import json
import re

from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import AIMessage

from prompts import (
    memory_summary_prompt,
    memory_facts_prompt,
    routing_prompt,
    ml_forecast_explain_prompt,
)
from fomc_tools import analyze_fomc_statement
from sp500_tools import analyze_sp500, ml_forecast_sp500
from state import SUMMARY_KEY, FACTS_KEY


def create_fed_agent(llm) -> Any:
    graph = StateGraph(MessagesState)

    # ---------------------------------------------------------------------
    # MEMORY NODE
    # ---------------------------------------------------------------------
    def memory_node(state: Dict) -> Dict:
        """
        Maintain a running summary + facts for the conversation.
        """
        messages = state.get("messages", [])
        summary = state.get(SUMMARY_KEY, "")
        facts = state.get(FACTS_KEY, {})

        # Get latest user message content (if any)
        last_msg_text = ""
        for msg in reversed(messages):
            # LangChain messages usually have .content
            content = getattr(msg, "content", None)
            if content:
                last_msg_text = content
                break

        if not last_msg_text:
            # Nothing new to summarize
            return {
                "messages": messages,
                SUMMARY_KEY: summary,
                FACTS_KEY: facts,
            }

        # 1) Update summary
        summary_prompt = memory_summary_prompt(summary, last_msg_text)
        summary_resp = llm.invoke(summary_prompt)
        new_summary = getattr(summary_resp, "content", summary_resp)

        # 2) Update facts from summary
        facts_prompt = memory_facts_prompt(new_summary)
        facts_resp = llm.invoke(facts_prompt)
        raw_facts = getattr(facts_resp, "content", facts_resp)

        try:
            parsed_facts = json.loads(raw_facts)
        except Exception:
            parsed_facts = raw_facts  # keep as string if not valid JSON

        return {
            "messages": messages,
            SUMMARY_KEY: new_summary,
            FACTS_KEY: parsed_facts,
        }

    # ---------------------------------------------------------------------
    # REASONING / ROUTING NODE
    # ---------------------------------------------------------------------
    def reasoning_node(state: Dict) -> Dict:
        messages = state.get("messages", [])
        summary = state.get(SUMMARY_KEY, "")
        facts = state.get(FACTS_KEY, {})

        if not messages:
            return {"messages": [AIMessage(content="Ask me about FOMC meetings or the S&P 500.")]}

        latest_msg = messages[-1]
        latest_text = getattr(latest_msg, "content", "").strip()

        # -------------------------------
        # 1) Use router prompt
        # -------------------------------
        route_prompt = routing_prompt(summary, facts, latest_text)
        router_resp = llm.invoke(route_prompt)
        router_text = getattr(router_resp, "content", router_resp)

        decision = "FOMC_ANALYZE"  # default
        reason = ""

        # Parse Decision: ... and Reason: ...
        for line in router_text.splitlines():
            line = line.strip()
            if line.lower().startswith("decision:"):
                decision = line.split(":", 1)[1].strip().upper()
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        # Hard override: if user explicitly mentions ML, force ML_FORECAST
        text_lower = latest_text.lower()
        if "machine learning" in text_lower or re.search(r"\bml\b", text_lower):
            decision = "ML_FORECAST"
        elif decision == "ML_FORECAST" and not (
            "machine learning" in text_lower or re.search(r"\bml\b", text_lower)
        ):
            # If router guessed ML_FORECAST without explicit ML mention, demote to SP500_ANALYZE
            decision = "SP500_ANALYZE"

        # -------------------------------
        # 2) Execute decision
        # -------------------------------
        if decision == "GREETING":
            # Lightweight greeting behavior
            short_greetings = {"hi", "hello", "hey", "hi!", "hello!", "hey!"}

            if text_lower.strip() in short_greetings:
                # Simple one-line greeting for plain "hi"/"hello"/"hey"
                output = "Hi! 👋 I'm FedSentiment."
            else:
                # Slightly richer response for small talk like "how are you?"
                output = (
                    "Hey! 😊 I'm FedSentiment. I'm doing well, thanks for asking.\n\n"
                    "I can help you explore Federal Reserve (FOMC) statements, interest "
                    "rates, and S&P 500 reactions — what would you like to look at?"
                )

        elif decision == "FOMC_ANALYZE":
            output = analyze_fomc_statement(llm, latest_text)

        elif decision == "SP500_ANALYZE":
            output = analyze_sp500(llm, latest_text)

        elif decision == "ML_FORECAST":
            # 1. Get numeric ML forecast
            ml_text = ml_forecast_sp500()

            # 2. Ask LLM to interpret it (but stay grounded in ml_text)
            explain_prompt = ml_forecast_explain_prompt(ml_text, latest_text)
            explain_resp = llm.invoke(explain_prompt)
            explain_text = getattr(explain_resp, "content", explain_resp)

            output = f"{ml_text}\n\nLLM Interpretation:\n{explain_text}"

        else:
            # Out-of-scope or unknown decision
            output = (
                "I'm FedSentiment, focused on Federal Reserve (FOMC) communications, "
                "interest rates, and S&P 500 behavior.\n\n"
                "Please ask about FOMC meetings, policy tone, rate direction, or the stock market "
                "(including ML-based S&P 500 forecasts)."
            )

        print(f"DEBUG: Agent decided on {decision} because {reason}")

        return {"messages": [AIMessage(content=output)]}


    # ---------------------------------------------------------------------
    # GRAPH WIRING
    # ---------------------------------------------------------------------
    graph.add_node("memory", memory_node)
    graph.add_node("reasoning", reasoning_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "reasoning")
    graph.set_finish_point("reasoning")

    return graph.compile()
