
import datetime as dt
import io

import pandas as pd
import streamlit as st
import yfinance as yf
import requests


# ------------- CONFIG & HELPERS ------------------------------------------------

st.set_page_config(page_title="NASDAQ 3‑Month Gainers Screener", layout="wide")

@st.cache_data(show_spinner=True)
def load_nyse_tickers() -> pd.DataFrame:
    import pandas as pd
    import requests
    from io import StringIO

    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), sep="|")

    # Remove footer
    df = df[df["ACT Symbol"] != "File Creation Time"]

    # NYSE only
    df = df[df["Exchange"] == "N"]

    # Clean
    df["ACT Symbol"] = df["ACT Symbol"].astype(str).str.strip()
    df["Security Name"] = df["Security Name"].astype(str).str.strip()

    # Filter common stock only
    exclude_terms = [
        "Warrant", "Unit", "Note", "Preferred", "Right",
        "Bond", "Debenture", "Trust", "Fund", "ETF"
    ]

    pattern = "|".join(exclude_terms)
    mask = ~df["Security Name"].str.contains(pattern, case=False, na=False)
    df = df[mask]

    df = df[df["Test Issue"] == "N"]

    df = df[["ACT Symbol", "Security Name"]].copy()
    df.columns = ["ticker", "name"]

    return df.reset_index(drop=True)

@st.cache_data(show_spinner=True)
def load_nasdaq_tickers() -> pd.DataFrame:
    import pandas as pd
    import requests
    from io import StringIO

    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), sep="|")

    # Remove footer row
    df = df[df["Symbol"] != "File Creation Time"]

    # Drop missing tickers
    df = df.dropna(subset=["Symbol"])

    # Clean fields
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Security Name"] = df["Security Name"].astype(str).str.strip()

    # Filter: common stock only
    exclude_terms = [
        "Warrant", "Unit", "Note", "Preferred", "Right",
        "Bond", "Debenture", "Trust", "Fund", "ETF"
    ]

    pattern = "|".join(exclude_terms)
    mask = ~df["Security Name"].str.contains(pattern, case=False, na=False)

    df = df[mask]

    # Keep only ticker + name
    df = df[df["Test Issue"] == "N"]
    df = df[["Symbol", "Security Name"]].copy()
    df.columns = ["ticker", "name"]

    return df.reset_index(drop=True)

@st.cache_data(show_spinner=True)
def load_sp500_tickers() -> pd.DataFrame:
    import pandas as pd
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    df = tables[0]

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)           # BRK.B → BRK-B
        .str.replace(r"[^A-Za-z0-9\-]", "", regex=True)  # remove † footnotes
        .str.strip()
    )

    df = df[["Symbol", "Security"]].copy()
    df.columns = ["ticker", "name"]

    return df.reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_prices(tickers, start, end):
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=False,
    )


def fetch_3m_returns(tickers: list[str]) -> pd.DataFrame:
    import math, datetime as dt, pandas as pd, yfinance as yf
    import streamlit as st

    rows = []
    end = dt.date.today()
    start = end - dt.timedelta(days=90)
    batch_size = 150
    batches = math.ceil(len(tickers) / batch_size)

    # Streamlit progress
    progress = st.progress(0)
    status = st.empty()

    for i in range(batches):
        batch = tickers[i*batch_size:(i+1)*batch_size]
        status.text(f"Processing batch {i+1}/{batches} ({len(batch)} tickers)...")

        try:
            data = yf.download(
                tickers=batch,
                start=start,
                end=end + dt.timedelta(days=1),
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"Skipped batch {i+1}: {e}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                try:
                    closes = data[t]["Close"].dropna()
                except Exception:
                    continue
                if len(closes) < 40:
                    continue
                first = closes.iloc[:5].mean()
                last = closes.iloc[-5:].mean()
                rows.append({"ticker": t, "3m_return_pct": (last / first - 1) * 100})
        else:
            closes = data["Close"].dropna()
            if len(closes) >= 40:
                first = closes.iloc[:5].mean()
                last = closes.iloc[-5:].mean()
                rows.append({"ticker": batch[0], "3m_return_pct": (last / first - 1) * 100})

        progress.progress((i + 1) / batches)

    progress.empty()
    status.empty()

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=True)
def fetch_metadata(tickers: list[str]) -> pd.DataFrame:
    rows = []

    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            continue

        price = info.get("currentPrice")
        market_cap = info.get("marketCap")
        avg_volume = info.get("averageVolume")
        beta = info.get("beta")
        sector = info.get("sector")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")

        rows.append({
            "ticker": t,
            "price": price,
            "market_cap": market_cap,
            "avg_volume": avg_volume,
            "beta": beta,
            "sector": sector,
            "52w_high": fifty_two_week_high,
        })

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=True)
def fetch_additional_returns(tickers: list[str]) -> pd.DataFrame:
    import datetime as dt
    import pandas as pd

    end = dt.date.today()
    start_6m = end - dt.timedelta(days=180)
    start_1m = end - dt.timedelta(days=30)

    rows = []

    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=start_6m, end=end)
        except Exception:
            continue

        if hist.empty:
            continue

        closes = hist["Close"].dropna()

        # FIX: normalize timezone
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        if len(closes) < 20:
            continue

        # 6‑month return
        first_6m = closes.iloc[:5].mean()
        last = closes.iloc[-5:].mean()
        ret_6m = (last / first_6m - 1) * 100

        # 1‑month return
        last_1m = closes[closes.index >= pd.Timestamp(start_1m)]
        if len(last_1m) >= 5:
            first_1m = last_1m.iloc[:5].mean()
            last_1m_val = last_1m.iloc[-5:].mean()
            ret_1m = (last_1m_val / first_1m - 1) * 100
        else:
            ret_1m = None

        rows.append({
            "ticker": t,
            "1m_return_pct": ret_1m,
            "6m_return_pct": ret_6m,
        })

    return pd.DataFrame(rows)

# ------------- SIDEBAR UI ------------------------------------------------------
universe_choice = st.sidebar.selectbox(
    "Select Exchange",
    ["NYSE","NASDAQ", "S&P500"],
    index=0
)

st.sidebar.title("Stock Screener")
st.sidebar.markdown("**Filter for top 3‑month gainers on NASDAQ.**")

if universe_choice == "NASDAQ":
    universe_raw = load_nasdaq_tickers()

elif universe_choice == "NYSE":
    universe_raw = load_nyse_tickers()

elif universe_choice == "S&P500":
    universe_raw = load_sp500_tickers()


# Universe size for slider 
universe_size = len(universe_raw) 

st.sidebar.markdown(f"Loaded **{universe_size}** {universe_choice} tickers.") 

# Slider 
min_slider = 1
max_universe = st.sidebar.slider(
    "Universe size (top N tickers alphabetically)",
    min_value=min_slider,
    max_value=universe_size, #max_slider,
    value=universe_size, #min(max_slider, 300),
    step=1,
    key="universe_slider_v2"
)

# Apply selection 
universe_df = (universe_raw.sort_values("ticker").head(max_universe).reset_index(drop=True))


top_n = st.sidebar.slider(
    "Show top N gainers",
    min_value=10,
    max_value=200,
    value=25,
    step=5,
)


run_button = st.sidebar.button("Run Screener")

if st.sidebar.button("🔄 Refresh price data"):
    st.cache_data.clear()
    st.rerun()


# ------------- MAIN LAYOUT WITH TABS -----------------------------------------------------

st.title(f"📈 {universe_choice} 3-Month Top Gainers Screener")
st.caption(
    "Uses free Yahoo Finance data via `yfinance`. Returns are approximate and for informational purposes only."
)
st.markdown(
    f"**Current universe:** {len(universe_df)} {universe_choice} stocks "
    f"(subset of full list for performance)."
)

with st.expander("View current universe tickers"):
    st.dataframe(universe_df, use_container_width=True, hide_index=True)

if not run_button:
    st.info("Set your options in the sidebar and click **Run Screener**.")
    st.stop()

with st.spinner("Fetching price history and computing 3‑month returns..."):
    tickers = universe_df["ticker"].tolist()
    returns_df = fetch_3m_returns(tickers)
    returns_df["3m_return_pct"] = pd.to_numeric(returns_df["3m_return_pct"], errors="coerce")
    returns_df = returns_df.dropna(subset=["3m_return_pct"])

if returns_df.empty:
    st.error("No return data could be computed. Try a smaller universe or different tickers.")
    st.stop()



# Merge with names
results = (
    returns_df.merge(universe_df, on="ticker", how="left")
    .sort_values("3m_return_pct", ascending=False)
    .reset_index(drop=True)
)

top_results = results.head(top_n)

# Fetch metadata and additional returns
meta_df = fetch_metadata(top_results["ticker"].tolist())
extra_returns_df = fetch_additional_returns(top_results["ticker"].tolist())


# ---------------- FIXED MERGE ----------------

# Ensure all 'ticker' columns are clean and string type
for df in [top_results, meta_df, extra_returns_df]:
    df.columns = df.columns.str.strip()  # remove whitespace from column names
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()  # clean ticker values

# Merge safely
enhanced = (
    top_results
    .merge(meta_df, on="ticker", how="left")
    .merge(extra_returns_df, on="ticker", how="left")
)

# Compute distance from 52-week high
enhanced["dist_from_52w_high_pct"] = (enhanced["price"] / enhanced["52w_high"] - 1) * 100

display_df = enhanced.rename(columns={
    "ticker": "Ticker",
    "name": "Name",
    "price": "Price",
    "3m_return_pct": "3M Return (%)",
    "1m_return_pct": "1M Return (%)",
    "6m_return_pct": "6M Return (%)",
    "avg_volume": "Avg Volume",
    "market_cap": "Market Cap",
    "sector": "Sector",
    "beta": "Beta",
    "dist_from_52w_high_pct": "Dist from 52W High (%)",
})

# ---------- TABS ----------
tab_gainers, tab_enhanced, tab_chart = st.tabs(["Top Gainers", "Enhanced Metrics", "Performance Chart"])

# --- Tab 1: Top Gainers ---
with tab_gainers:
    st.subheader(f"Top {len(top_results)} {universe_choice} gainers over last ~3 months")
    st.dataframe(
        top_results.assign(**{"3m_return_pct": top_results["3m_return_pct"].round(2)})
        .rename(columns={"ticker": "Ticker", "name": "Name", "3m_return_pct": "3M Return (%)"}),
        use_container_width=True,
        hide_index=True,
    )

# --- Tab 2: Enhanced Metrics with expanders per ticker ---
import altair as alt

# --- Tab 2: Enhanced Metrics with expanders per ticker ---
with tab_enhanced:
    st.subheader("Enhanced Metrics with Per-Ticker Details")

    for i, row in enhanced.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        with st.expander(f"{ticker} - {name}"):
            # 1️⃣ Prepare metrics table
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Price", "1M Return (%)", "3M Return (%)", "6M Return (%)",
                    "Avg Volume", "Market Cap", "Sector", "Beta", "Distance from 52W High (%)"
                ],
                "Value": [
                    row.get("price"),
                    round(row.get("1m_return_pct", 0), 2),
                    round(row.get("3m_return_pct", 0), 2),
                    round(row.get("6m_return_pct", 0), 2),
                    row.get("avg_volume"),
                    row.get("market_cap"),
                    row.get("sector"),
                    row.get("beta"),
                    round(row.get("dist_from_52w_high_pct", 0), 2)
                ]
            })

            # 2️⃣ Fetch key financial highlights from yfinance info
            try:
                info = yf.Ticker(ticker).info

                financial_highlights = {
                    "P/E Ratio": info.get("trailingPE", "N/A"),
                    "Forward P/E": info.get("forwardPE", "N/A"),
                    "Dividend Yield (%)": round(info.get("dividendYield", 0)*100, 2) if info.get("dividendYield") else "N/A",
                    "EPS (TTM)": info.get("earningsPerShare", "N/A"),
                    "Revenue (TTM)": f"${info.get('totalRevenue'):,}" if info.get('totalRevenue') else "N/A",
                    "Profit Margin (%)": round(info.get("profitMargins", 0)*100, 2) if info.get("profitMargins") else "N/A",
                }

                # 3️⃣ Convert to DataFrame matching the metrics_df format
                financial_df = pd.DataFrame({
                    "Metric": list(financial_highlights.keys()),
                    "Value": list(financial_highlights.values())
                })

                # 4️⃣ Concatenate financial highlights to metrics_df
                metrics_df = pd.concat([metrics_df, financial_df], ignore_index=True)

            except Exception as e:
                st.warning(f"Could not fetch financial highlights: {e}")

            # 2️⃣ Color-code numeric returns
            def color_return(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        color = "green"
                    elif val < 0:
                        color = "red"
                    else:
                        color = "black"
                    return f"color: {color}"
                return None

            metrics_styled = metrics_df.style.applymap(color_return, subset=["Value"])

            # 3️⃣ Horizontal layout: metrics on left, chart on right
            col1, col2 = st.columns([1, 2])
            with col1:
                st.table(metrics_styled)
            with col2:
                # 4️⃣ Fetch price history and show interactive chart
                try:
                    hist = yf.Ticker(ticker).history(period="3mo")["Close"]
                    hist_df = hist.reset_index()
                    hist_df.columns = ["Date", "Close"]

                    chart = (
                        alt.Chart(hist_df)
                        .mark_line(color="#1f77b4")
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Close:Q", title="Adjusted Close Price"),
                            tooltip=["Date:T", "Close:Q"]
                        )
                        .properties(
                            title=f"{ticker} - {name} | Last 3 Months",
                            width=700,
                            height=370  
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not load chart for {ticker}: {e}")

                # --- News Section ---
                st.markdown("**Recent News:**")
                try:
                    news_items = yf.Ticker(ticker).news
                    if news_items:
                        for n in news_items[:5]:  # top 5 latest
                            content = n.get("content", {})
                            title = content.get("title", "No title")
                            url = content.get("canonicalUrl", {}).get("url", "#")
                            publisher = content.get("provider", {}).get("displayName", "")
                            st.markdown(f"- [{title}]({url}) — {publisher}")
                    else:
                        st.info("No recent news found.")
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")



# --- Tab 3: Performance Chart ---
with tab_chart:
    st.subheader("Performance Bar Chart")
    chart_df = top_results[["ticker", "3m_return_pct"]].sort_values("3m_return_pct", ascending=False)
    import altair as alt
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("3m_return_pct:Q", title="3-Month Return (%)"),
            y=alt.Y("ticker:N", sort=None, title="Ticker"),
            tooltip=["ticker", "3m_return_pct"]
        )
        .properties(height=max(500, 20 * len(chart_df)))
    )
    st.altair_chart(chart, use_container_width=True)

st.caption(
    "Note: Returns are based on first vs. last available adjusted close in the last ~90 days."
)

st.caption("Developed by Win Ltd | Data via Yahoo Finance")


#add saved watch list
#add daily email / alerts
#brand it logo + tagline
#show it to 5 traders
#would you pay for this

# MVP - Stream lit hosted web append
#Browser
#   ↓
# Streamlit Frontend
#   ↓
# Python Business Logic
#   ↓
# yfinance + cached data
