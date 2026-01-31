# fed_agent/state.py

SUMMARY_KEY = "summary"
FACTS_KEY = "facts"

def init_memory_state():
    """
    Initialize memory state structure for LangGraph agent.
    """
    return {
        SUMMARY_KEY: "",
        FACTS_KEY: {}
    }
