# ===============================
# app.py — ₹20–₹30 Stock Scout (India)
# 100% FREE • No API Keys • Yahoo Finance Only
# Streamlit single-file app (no loose ends)
# ===============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="₹20–₹30 Stock Scout — India", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("₹20–₹30 Stock Scout — India")
st.caption("Completely Free • No API Keys • Yahoo Finance • Scouting Tool (Not Investment Advice)")

# -------------------------------
# NSE Stock Universe (Liquid, Large + Mid Caps)
# Expandable — safe for Yahoo
# -------------------------------
NSE_STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
    "AXISBANK.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "ITC.NS","HINDUNILVR.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","COALINDIA.NS",
    "ADANIENT.NS","ADANIPORTS.NS","TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS",
    "HINDALCO.NS","ULTRACEMCO.NS","GRASIM.NS","TECHM.NS","WIPRO.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
    "ASIANPAINT.NS","TITAN.NS","MARUTI.NS","M&M.NS","HEROMOTOCO.NS",
    "EICHERMOT.NS","BPCL.NS","IOC.NS","HCLTECH.NS","INDUSINDBK.NS",
    "PNB.NS","BANKBARODA.NS","CANBK.NS","IDFCFIRSTB.NS","FEDERALBNK.NS",
]

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Filters")
MIN_SCORE = st.sidebar.slider("Minimum Score", 50, 90, 65, 5)
PRICE_MIN, PRICE_MAX = st.sidebar.slider("Price Range (₹)", 50, 1500, (100, 1200), 50)

# -------------------------------
# Cached Data Fetch
# -------------------------------
@st.cache_data(ttl=900)
def fetch_data(ticker):
    return yf.download(ticker, period="6mo", interval="1d", progress=False)

# -------------------------------
# Indicator Computation
# -------------------------------
def compute_indicators(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['Close'], 14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    df['VOL_AVG_20'] = df['Volume'].rolling(20).mean()
    df['HIGH_20'] = df['High'].rolling(20).max()
    df['HIGH_50'] = df['High'].rolling(50).max()
    df['ATR_AVG_20'] = df['ATR'].rolling(20).mean()
    return df

# -------------------------------
# Scoring Engine
# -------------------------------
def score_stock(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    # Hard Filters
    if latest['Close'] < PRICE_MIN or latest['Close'] > PRICE_MAX:
        return None
    if latest['Volume'] < 2 * latest['VOL_AVG_20']:
        return None
    if latest['ATR'] < 5:
        return None

    # Scoring Logic
    if latest['Volume'] >= 3 * latest['VOL_AVG_20']:
        score += 20
    if latest['Close'] > latest['HIGH_20']:
        score += 15
    if latest['Close'] > latest['HIGH_50']:
        score += 10
    if latest['RSI'] > 60:
        score += 10
    if latest['RSI'] > prev['RSI']:
        score += 10
    if latest['ATR'] > latest['ATR_AVG_20']:
        score += 10

    expected_move = max(1.5 * latest['ATR'], 20)

    return {
        "Symbol": "",
        "LTP": round(latest['Close'], 2),
        "Score": score,
        "Volume Multiplier": round(latest['Volume'] / latest['VOL_AVG_20'], 2),
        "ATR": round(latest['ATR'], 2),
        "RSI": round(latest['RSI'], 2),
        "Expected ₹ Move": round(expected_move, 2)
    }

# -------------------------------
# Run Scan
# -------------------------------
if st.button("🔄 Refresh Scan"):
    progress = st.progress(0)
    results = []

    for i, ticker in enumerate(NSE_STOCKS):
        try:
            df = fetch_data(ticker)
            if df.empty or len(df) < 60:
                continue

            df = compute_indicators(df)
            data = score_stock(df)

            if data and data['Score'] >= MIN_SCORE:
                data['Symbol'] = ticker.replace('.NS', '')
                results.append(data)

        except Exception:
            pass

        progress.progress((i + 1) / len(NSE_STOCKS))

    if results:
        output = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.subheader(f"Qualified Stocks ({len(output)})")
        st.dataframe(output, use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            output.to_csv(index=False),
            "₹20_scout_results.csv",
            "text/csv"
        )
    else:
        st.warning("No stocks met the criteria in this scan.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(f"Last App Load: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
st.caption("This tool identifies probability-based momentum candidates, not guaranteed outcomes.")
