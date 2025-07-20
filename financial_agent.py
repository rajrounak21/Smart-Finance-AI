# streamlit_app.py

import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from key import OPENAI_API_KEY
import os

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGo()],
    description="An AI web searcher that uses DuckDuckGo to fetch up-to-date information from the web. Useful for recent news and public web data.",
    instructions=["Always include sources"],
    show_tool_calls=False,  # Hide tool logs
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    description="An AI financial analyst that uses Yahoo Finance tools to fetch stock prices, company info, and analyst recommendations.",
    instructions=["Use tables to display data"],
    show_tool_calls=False,  # Hide tool logs
    markdown=True,
)

# Combine into a team
agent_team = Agent(
    team=[web_agent, finance_agent],
    role="An orchestrator that assigns tasks to the appropriate agents for finance and web-related questions.",
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=False,  # Hide tool logs
    markdown=True,
)

# --- Streamlit UI ---

st.set_page_config(page_title="AI Financial Insights", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Financial Insights")

user_query = st.text_input("Enter your query (e.g., TSLA stock analyst recommendations and news):", "")

if st.button("Run Query") and user_query:
    with st.spinner("Running agentic team..."):
        response = agent_team.run(user_query)
        st.markdown(response.content, unsafe_allow_html=True)
