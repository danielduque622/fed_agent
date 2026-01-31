# fed_agent/app.py
import os
import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from agent import create_fed_agent
from langchain_google_genai import GoogleGenerativeAI

st.set_page_config(page_title="FedSentiment", page_icon="💬", layout="wide")

st.title("💬 FedSentiment")
st.markdown(
    """
    Your AI assistant for exploring Federal Reserve (FOMC) communications,
    interest rate decisions, and S&P 500 outlooks.
    """
)

# --- Sidebar configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"], index=0)
    gemini_api_key = st.text_input("🔑 Gemini API Key", type="password")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    show_steps = st.checkbox("Show Agent Steps", value=False)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask me anything about the Fed or markets..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        llm = GoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        agent_graph = create_fed_agent(llm)

        # Convert chat history
        langchain_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            else:
                langchain_messages.append(AIMessage(content=msg["content"]))

        state = {"messages": langchain_messages}
        with st.spinner("Thinking..."):
            final_state = agent_graph.invoke(state)
            response = final_state["messages"][-1].content

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
