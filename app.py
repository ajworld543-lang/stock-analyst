"""
Web page for the Stock Analyst agent (built with Streamlit).
Needs stock_agent.py in the same folder.
"""
import os
import streamlit as st
import stock_agent as sa

st.set_page_config(page_title="Stock Analyst AI", page_icon="📈", layout="wide")

# ---- read secrets (API key + optional password) set in Streamlit Cloud ----
password = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    password = st.secrets.get("APP_PASSWORD")
except Exception:
    pass  # no secrets file (e.g. running locally with an environment variable)

st.title("📈 Stock Analyst AI")
st.caption("Type a company name and get a detailed fundamental analysis. Educational use only, not investment advice.")

# ---- optional password so strangers can't use up your free quota ----
if password:
    entered = st.text_input("Password", type="password")
    if entered != password:
        st.info("Enter the password to continue.")
        st.stop()

if "GEMINI_API_KEY" not in os.environ:
    st.error("GEMINI_API_KEY is missing. Add it in the app's Secrets settings.")
    st.stop()


# ---- cached helpers (same stock within 6 hours = instant, saves your free quota) ----
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_all(name: str):
    ticker = sa.resolve_ticker(name)
    if not ticker:
        return None, None
    data = "\n\n".join([
        sa.get_yfinance_data(ticker),
        sa.get_screener_data(ticker),
        sa.get_news(name),
    ])
    return ticker, data


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def make_report(name: str, ticker: str, data: str):
    return sa.analyse(name, ticker, data)


# ---- the search box ----
with st.form("search"):
    name = st.text_input("Stock name", placeholder="e.g. Tata Motors, Reliance, HDFC Bank")
    go = st.form_submit_button("Analyse")

if go and name.strip():
    name = name.strip()
    try:
        with st.spinner("Finding the stock and collecting data..."):
            ticker, data = fetch_all(name)
        if not ticker:
            st.error("Couldn't find that stock. Try the exact ticker, e.g. RELIANCE.NS")
            st.stop()

        with st.spinner(f"Analysing {ticker} (this takes about a minute)..."):
            report = make_report(name, ticker, data)

        st.subheader(f"{name}  ({ticker})")
        st.markdown(report)
        st.download_button("Download report", report, file_name=f"{ticker}_analysis.md")

        with st.expander("See the raw data the AI used"):
            st.text(data)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
