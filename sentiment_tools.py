# fed_agent/tools/sentiment_tools.py
from prompts import sentiment_prompt

def analyze_sentiment(llm, statement_excerpt: str) -> str:
    prompt = sentiment_prompt(statement_excerpt)
    return llm.invoke(prompt)
