import os

import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.google import Gemini
from key import GOOGLE_API_KEY
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Define the web agent for real-time search and financial data
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Gemini(id="gemini-2.5-pro-exp-03-25"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Define the finance agent for financial data
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Gemini(id="gemini-2.5-pro-exp-03-25"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Combine both agents into a team
agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit app setup
st.set_page_config(page_title="AI Financial Insights", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Financial Insights")

# Text input for stock symbol
stock_symbol = st.text_input("📌 Enter Stock Symbol (e.g., NVDA, AAPL, TSLA)")

# Button to trigger the analysis
if st.button("🔍 Analyze Stock", key="stock_analysis_button"):
    if stock_symbol:
        with st.spinner("⏳ Fetching financial insights..."):
            try:
                # Step 1: Send task to finance agent
                st.info("📤 Sending task to Finance Agent...")
                finance_response = finance_agent.run(
                    f"Get the latest analyst recommendations for {stock_symbol}. "
                    f"Focus on the latest recommendations available. "
                    f"Output a summary including ratings and price targets."
                )

                st.success("✅ Received analyst recommendations.")

                # Step 2: Send task to web agent
                st.info("🌐 Sending task to Web Agent for recent news...")
                web_response = web_agent.run(
                    f"Find recent news about {stock_symbol} that could affect its stock price. "
                    f"Focus on the last 7 days only. Include sources."
                )

                st.success("✅ Received recent news insights.")

                # Step 3: Display results
                st.subheader(f"📊 Financial Insights for {stock_symbol}")
                st.markdown("### 🧾 Analyst Recommendations")
                st.markdown(finance_response.content)

                st.markdown("### 📰 Latest News")
                st.markdown(web_response.content)

            except Exception as error:
                st.error(f"❌ Error during analysis: {error}")
    else:
        st.warning("⚠️ Please enter a stock symbol to analyze.")
