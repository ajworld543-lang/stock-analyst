"""
Free Stock Fundamental Analysis Agent
-------------------------------------
Type a company name -> it gathers data from yfinance, Screener.in and Google News,
then asks a free Gemini model to write a detailed fundamental report.

Setup:
    pip install yfinance requests beautifulsoup4 feedparser google-genai
    # get a free key at https://aistudio.google.com/apikey
    export GEMINI_API_KEY="your_key"        (Windows: set GEMINI_API_KEY=your_key)

Run:
    python stock_agent.py "Tata Motors"
    python stock_agent.py            (it will ask you for the name)
"""
import os
import sys
import time
import datetime
from urllib.parse import quote

import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
from google import genai

# Pick any Flash / Flash-Lite model that is marked "free" on Google's pricing page.
GEMINI_MODEL = "gemini-2.5-flash"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal research script)"}


# ---------------------------------------------------------------- 1. Resolve name -> ticker
def resolve_ticker(name: str):
    """Turn 'Tata Motors' into 'TATAMOTORS.NS' (prefers NSE, then BSE)."""
    if "." in name and " " not in name:          # user already typed a ticker
        return name.upper()
    try:
        quotes = yf.Search(name, max_results=10).quotes
    except Exception:
        quotes = []
    for suffix in (".NS", ".BO"):
        for q in quotes:
            if q.get("symbol", "").endswith(suffix):
                return q["symbol"]
    return quotes[0]["symbol"] if quotes else None


# ---------------------------------------------------------------- 2. Data sources
def get_yfinance_data(ticker: str) -> str:
    t = yf.Ticker(ticker)
    info = t.info or {}
    currency = info.get("financialCurrency", "INR")
    scale, unit = (1e7, "Rs Cr") if currency == "INR" else (1e6, f"{currency} Mn")

    keys = [
        "longName", "sector", "industry", "marketCap", "currentPrice",
        "trailingPE", "forwardPE", "priceToBook", "pegRatio", "enterpriseToEbitda",
        "returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins",
        "debtToEquity", "currentRatio", "dividendYield", "payoutRatio",
        "heldPercentInsiders", "heldPercentInstitutions", "beta",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "longBusinessSummary",
    ]
    out = ["## yfinance: key stats"]
    out += [f"{k}: {info.get(k)}" for k in keys if info.get(k) is not None]

    def pick(df, rows, title):
        try:
            rows = [r for r in rows if r in df.index]
            if not rows:
                return f"## {title}: not available"
            sub = (df.loc[rows] / scale).round(1)
            return f"## yfinance: {title} ({unit})\n{sub.to_string()}"
        except Exception:
            return f"## {title}: not available"

    out.append(pick(t.financials, ["Total Revenue", "Operating Income", "EBITDA", "Net Income", "Diluted EPS"], "Income statement (annual)"))
    out.append(pick(t.quarterly_financials, ["Total Revenue", "Operating Income", "Net Income"], "Income statement (quarterly)"))
    out.append(pick(t.balance_sheet, ["Total Assets", "Total Debt", "Stockholders Equity", "Cash And Cash Equivalents"], "Balance sheet"))
    out.append(pick(t.cashflow, ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"], "Cash flow"))
    return "\n".join(out)


def get_screener_data(ticker: str) -> str:
    """Scrapes the public Screener.in company page (personal use, be polite)."""
    symbol = ticker.split(".")[0]
    html = None
    for suffix in ("consolidated/", ""):
        try:
            r = requests.get(f"https://www.screener.in/company/{quote(symbol)}/{suffix}", headers=HEADERS, timeout=20)
            if r.status_code == 200:
                html = r.text
                break
        except requests.RequestException:
            pass
    if not html:
        return "## Screener.in: data not available for this ticker"

    soup = BeautifulSoup(html, "html.parser")
    out = ["## Screener.in: top ratios"]
    for li in soup.select("#top-ratios li"):
        n, v = li.select_one(".name"), li.select_one(".value")
        if n and v:
            out.append(f"{n.get_text(strip=True)}: {v.get_text(' ', strip=True)}")

    out.append("## Screener.in: pros")
    out += ["- " + li.get_text(strip=True) for li in soup.select(".pros li")]
    out.append("## Screener.in: cons")
    out += ["- " + li.get_text(strip=True) for li in soup.select(".cons li")]

    sections = {
        "profit-loss": "Profit & Loss (10 yrs, incl. growth tables)",
        "balance-sheet": "Balance sheet",
        "cash-flow": "Cash flows",
        "ratios": "Ratios (ROCE, debtor days, etc.)",
        "shareholding": "Shareholding pattern",
        "peers": "Peer comparison",
    }
    for sec_id, title in sections.items():
        sec = soup.select_one(f"#{sec_id}")
        if not sec:
            continue
        out.append(f"## Screener.in: {title}")
        for tr in sec.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.select("th,td")]
            if any(cells):
                out.append(" | ".join(cells))
        # compound growth boxes (sales/profit CAGR, ROE) live in small tables
        for tbl in sec.select("table.ranges-table"):
            out.append(" ; ".join(x.get_text(" ", strip=True) for x in tbl.select("tr")))
    return "\n".join(out)[:60000]


def get_news(name: str) -> str:
    url = f"https://news.google.com/rss/search?q={quote(name + ' stock')}+when:30d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = [f"- {e.title} ({e.get('published', '')[:16]})" for e in feed.entries[:12]]
    return "## Recent news (last 30 days)\n" + ("\n".join(items) if items else "none found")


# ---------------------------------------------------------------- 3. LLM analysis
PROMPT = """You are a careful, skeptical equity research analyst.
Using ONLY the data below, write a detailed fundamental analysis of {name} ({ticker}).
If a data point is missing, say so rather than guessing. Quote actual numbers.

Structure the report in Markdown:
1. Business overview (what it does, how it makes money, moat)
2. Growth (revenue/profit trends, 3/5/10-yr CAGR where available)
3. Profitability (margins, ROE, ROCE and their trend)
4. Balance sheet & leverage (debt/equity, interest coverage, liquidity)
5. Cash flow quality (CFO vs net profit, free cash flow, capex)
6. Valuation (PE, PB, EV/EBITDA vs own history and vs peers)
7. Ownership (promoter holding & pledging, FII/DII trend)
8. Recent news & catalysts
9. Key risks / red flags
10. Bull case vs bear case
11. Scorecard (1-10 for Growth, Profitability, Balance sheet, Cash flow, Valuation) and a final summary

End with: "Educational analysis only, not investment advice."

=== DATA ===
{data}
"""


def analyse(name: str, ticker: str, data: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = PROMPT.format(name=name, ticker=ticker, data=data)
    for attempt in range(4):                     # simple retry for free-tier rate limits
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
        except Exception as e:
            if attempt == 3:
                raise
            wait = 20 * (attempt + 1)
            print(f"  LLM busy/rate-limited ({e.__class__.__name__}); retrying in {wait}s...")
            time.sleep(wait)


# ---------------------------------------------------------------- 4. Run
def main():
    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("Set GEMINI_API_KEY first (free key: https://aistudio.google.com/apikey)")

    name = " ".join(sys.argv[1:]) or input("Stock name: ").strip()
    print(f"Resolving '{name}'...")
    ticker = resolve_ticker(name)
    if not ticker:
        sys.exit("Could not find that stock. Try the exact ticker, e.g. RELIANCE.NS")
    print(f"Ticker: {ticker}\nFetching data...")

    data = "\n\n".join([get_yfinance_data(ticker), get_screener_data(ticker), get_news(name)])
    print("Analysing with Gemini...\n")
    report = analyse(name, ticker, data)

    print(report)
    os.makedirs("reports", exist_ok=True)
    path = f"reports/{ticker}_{datetime.date.today()}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
