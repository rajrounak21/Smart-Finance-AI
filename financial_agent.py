import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
import os
from key import GOOGLE_API_KEY

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Multimodal Financial AI", page_icon="📊", layout="wide")
st.title("📊 Smart Financial AI: Stock & Market Insights")
st.header("📈 Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_financial_agent():
    return Agent(
        name="Multimodal Finance AI",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                company_news=True,
                stock_fundamentals=True
            ),
            DuckDuckGo()  # Web search for real-time financial news
        ],
        markdown=True,
    )

multimodal_finance_agent = initialize_financial_agent()

# **Text input is now on top**
stock_symbol = st.text_input("📌 Enter Stock Symbol (e.g., NVDA, AAPL, TSLA)")

# **Button is below for better UI flow**
if st.button("🔍 Analyze Stock", key="stock_analysis_button"):
    if stock_symbol:
        with st.spinner("⏳ Fetching financial insights and web research..."):
            try:
                analysis_prompt = f"""
                Conduct a financial analysis of {stock_symbol}.
                Include:
                - Stock price
                - Analyst recommendations
                - Company news
                - Key stock fundamentals 
                - Recent web search results for the latest developments
                - Use tables for displaying data
                Format the response in a structured, easy-to-read format.
                Only show relevant insights, not everything.
                """
                response = multimodal_finance_agent.run(analysis_prompt)
                st.subheader("📊 Financial Insights")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"❌ An error occurred during analysis: {error}")
    else:
        st.warning("⚠️ Please enter a stock symbol to analyze.")
