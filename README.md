# FedSentiment: AI-Powered FOMC Analysis & Market Forecasting
FedSentiment is an AI-driven application designed to decode complex Federal Reserve FOMC communications. It translates technical monetary policy statements into clear natural language insights, classifies policy sentiment, and forecasts potential market outcomes for the S&P 500
## Project Overview
The Federal Reserve’s FOMC statements significantly impact interest rates, employment, and inflation. However, these documents are often too technical for the general public or small businesses to interpret effectively.
### FedSentiment solves this by:
- Retrieving accurate FOMC meeting data based on natural language queries (e.g., "latest meeting" or "first meeting in 2017").
- Summarizing content without hallucinations to provide consistent, easy-to-understand insights.
- Classifying policy tone into three categories: Hawkish, Neutral, or Dovish.
- Forecasting S&P 500 price movements using both qualitative LLM reasoning and quantitative Machine Learning models.
## Technical Architecture
The project evolved from a basic Transformer-based analyzer into a sophisticated agentic workflow using LangChain and LangGraph.
### Core Components
- LLM Engine: Powered by Gemini (via LangChain) to handle semantic understanding and conversational reasoning.
- Agentic Logic: Uses a LangGraph ReAct Agent to implement a reasoning + acting structure.
- Tooling System:
  - FOMC Analyzer: Loads and summarizes specific meeting statements.
  - Sentiment Analyzer: Classifies policy stance using structured prompting.
  - S&P 500 Predictor: Combines qualitative outlooks with a Linear Regression model using lag features (up to 5 days).
- Interface: A streamlined Streamlit dashboard for user interaction.
## Performance Metrics
### LLM Performance 
The model was evaluated using a dataset of 10 meetings (20 total rows of statements and minutes).
- Date-Resolution Accuracy: 100% 
- Numerical Reasoning Accuracy: 90% 
- Tone Classification Consistency: 80% 
- Tone Classification Accuracy: 60% 
- Hallucination Rate: 0% 
### ML Performance (S&P 500 Forecasting) 
The Linear Regression model utilizes $t-5$ lag features to ensure computational efficiency.
- MAPE: 0.65% 
- RMSE: 0.97% 
- Directional Accuracy: 51.5% 
## Lessons Learned
- Ground Truth is Key: LLMs do not inherently know the "truth"; they require a well-structured dataset to reference through specific tools and routing logic.
- Tool-Based Reliability: Building dedicated tools for information retrieval is significantly more reliable than relying on a standalone LLM for summaries.
- Numerical Limitations: Even advanced models can struggle with basic math in financial contexts, requiring explicit logic instructions via frameworks like LangChain.
