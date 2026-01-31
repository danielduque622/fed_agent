# fed_agent/prompts.py

def get_role_prompt():
    return (
        "You are FedSentiment, an AI assistant analyzing Federal Reserve (FOMC) "
        "statements, interest rate policy, and S&P 500 outlooks based only on the provided data."
    )


def fomc_prompt(statement_text: str, user_query: str) -> str:
    """
    Prompt used for single-meeting FOMC analysis.

    The model is explicitly told:
    - Only use information from the provided statement/context.
    - If the user asks for information not supported by the context,
      it should say so instead of inventing details.
    """
    return (
        "System: You are an expert Federal Reserve (FOMC) analyst.\n"
        "You MUST base your answer ONLY on the FOMC statement text in the context.\n"
        "If the user asks for information that is not supported by the context, "
        "you must say you cannot tell from this statement alone.\n\n"
        "Task: Summarize the meeting and infer the likely next rate action (CUT / HOLD / HIKE)\n"
        "based on this FOMC statement.\n\n"
        f"Context (FOMC statement, including its date):\n{statement_text}\n\n"
        f"User Question: {user_query}\n\n"
        "Output format (strict):\n"
        "Summary: <brief, 2–4 sentences>\n"
        "Tone: Hawkish | Neutral | Dovish\n"
        "Likely Rate Direction: CUT | HOLD | HIKE | Unknown\n"
        "Confidence: Low | Medium | High\n"
    )


def sentiment_prompt(statement_excerpt: str) -> str:
    return (
        "System: Monetary policy sentiment classifier.\n"
        "Task: Determine the stance of this text towards tighter vs. looser policy.\n\n"
        f"Text: {statement_excerpt}\n\n"
        "Output format:\n"
        "Tone: Hawkish | Neutral | Dovish\n"
        "Reasoning: <one sentence>\n"
    )


def sp500_prompt(
    fomc_summary: str,
    fed_rate: float,
    sp_trend: str,
    user_question: str,
) -> str:
    return (
        "System: Market strategist.\n"
        "Task: Provide a qualitative S&P 500 outlook specifically in response to the user's question.\n\n"
        "Context:\n"
        f"- Latest FOMC communication (summary): {fomc_summary}\n"
        f"- Current federal funds rate: {fed_rate}\n"
        f"- Recent S&P 500 behaviour: {sp_trend}\n\n"
        f"User Question: {user_question}\n\n"
        "Interpretation rules:\n"
        "- 'Bullish' = S&P 500 is more likely to RISE in the short term.\n"
        "- 'Bearish' = S&P 500 is more likely to FALL in the short term.\n"
        "- 'Neutral' = S&P 500 is more likely to be roughly FLAT / mixed.\n"
        "- If the user asks about what happens AFTER a rate CUT/HOLD/HIKE or AFTER the next meeting,\n"
        "  you must base your outlook on how such a policy stance typically affects equities, given the\n"
        "  provided FOMC context and recent S&P 500 trend.\n\n"
        "You MUST follow this exact output format (three lines only, in this order):\n"
        "Outlook: <Bullish | Bearish | Neutral>\n"
        "Rationale: <brief explanation linking Fed policy, the user's question, and S&P 500 behaviour>\n"
        "Confidence: <Low | Medium | High>\n"
    )


def ml_forecast_explain_prompt(ml_summary: str, user_question: str) -> str:
    """
    Prompt to explain the ML S&P 500 forecast for the user.

    This matches agent.py, where we:
      1) Call ml_forecast_sp500() to get a human-readable text summary (ml_summary).
      2) Pass that summary + the original user question into this prompt.

    The LLM is forced to stay grounded in the ML summary string and not invent
    new numbers or model details.
    """
    return (
        "System: You are a financial data analyst explaining an ML-based S&P 500 forecast.\n"
        "You are given the model's own textual output and the user's question.\n"
        "You MUST:\n"
        "- Stay grounded in the model output text provided.\n"
        "- Do NOT invent specific numeric values that are not already in the model output.\n"
        "- Treat this as an educational explanation, not trading advice.\n\n"
        f"Model output:\n{ml_summary}\n\n"
        f"User question: {user_question}\n\n"
        "Output format:\n"
        "Predicted Level: <copy the predicted S&P 500 level from the model output, or 'not specified'>\n"
        "Interpretation: <plain-language explanation of what this forecast means in context>\n"
        "Risk Commentary: <brief note on uncertainty and limitations of this simple ML model>\n"
    )


def memory_summary_prompt(existing_summary: str, last_user_message: str) -> str:
    return (
        "System: Conversation summarizer.\n"
        "Task: Update the running summary (1–3 sentences).\n\n"
        f"Existing Summary: {existing_summary}\n\n"
        f"New Message: {last_user_message}\n\n"
        "Output: <Updated summary>"
    )


def memory_facts_prompt(running_summary: str) -> str:
    return (
        "System: Fact extractor.\n"
        "Task: Extract stable facts as a JSON object.\n\n"
        f"Running Summary: {running_summary}\n\n"
        "Output: JSON object\n"
    )


def routing_prompt(running_summary: str, facts: dict, latest_user_message: str) -> str:
    return (
        "System: Agent router.\n"
        "Task: Choose which tool to call.\n\n"
        f"Summary: {running_summary}\n"
        f"Facts: {facts}\n"
        f"User Message: {latest_user_message}\n\n"
        "You MUST follow these rules strictly:\n"
        "- If the user explicitly asks for an ML-based or machine learning forecast "
        "  (mentions 'ML', 'ml', 'machine learning', 'regression', 'model', or "
        "  'linear regression'), then choose ML_FORECAST.\n"
        "- If the user asks about the S&P 500 outlook, direction, or future levels "
        "  WITHOUT explicitly mentioning ML or machine learning, choose SP500_ANALYZE "
        "  so the LLM gives a qualitative outlook.\n"
        "- If the user is asking directly about FOMC statements or interest rate "
        "  decisions, choose FOMC_ANALYZE.\n"
        "- Simple greetings or light small talk like 'hi', 'hello', 'hey', "
        "  'how are you', 'how’s it going', or similar -> choose GREETING.\n"
        "- For anything outside Fed/FOMC/rates/S&P 500, choose OOS.\n\n"
        "Choose: FOMC_ANALYZE, SP500_ANALYZE, ML_FORECAST, GREETING, OOS.\n\n"
        "Output format:\n"
        "Decision: <one>\n"
        "Reason: <short>\n"
    )
