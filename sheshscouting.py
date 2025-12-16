# ===============================
# app.py — ₹20–₹30 Stock Scout (India)
# STREAMLIT CLOUD SAFE VERSION
# NO ta LIBRARY • NO API KEYS • YAHOO FINANCE ONLY
# ===============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="₹20–₹30 Stock Scout — India", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("₹20–₹30 Stock Scout — India")
st.caption("Free • No API Keys • Yahoo Finance • Cloud-safe build")

# -------------------------------
# NSE Stock Universe (safe size)
# -------------------------------
NSE_STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
    "AXISBANK.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "ITC.NS","HINDUNILVR.NS","ONGC.NS","NTPC.NS","POWERGRID.NS",
    "ADANIENT.NS","ADANIPORTS.NS","TATAMOTORS.NS","TATASTEEL.NS"
]

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Scanner Settings")
MODE = st.sidebar.radio("Mode", ["Intraday", "Swing (Daily)"])
MIN_SCORE = st.sidebar.slider("Minimum Score", 50, 90, 65, 5)
PRICE_MIN, PRICE_MAX = st.sidebar.slider("Price Range (₹)", 50, 1500, (100, 1200), 50)

# -------------------------------
# Cached Fetchers
# -------------------------------
@st.cache_data(ttl=600)
def fetch_daily(ticker):
    return yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)

@st.cache_data(ttl=300)
def fetch_intraday(ticker):
    return yf.download(ticker, period="5d", interval="5m", progress=False, threads=False)

# -------------------------------
# Indicator Functions (PURE PANDAS)
# -------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# -------------------------------
# Indicator Engine
# -------------------------------
def compute_indicators(df):
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'])
    df['ATR'] = compute_atr(df)
    df['VOL_AVG_20'] = df['Volume'].rolling(20).mean()
    df['HIGH_20'] = df['High'].rolling(20).max()
    df['HIGH_50'] = df['High'].rolling(50).max()
    df['ATR_AVG_20'] = df['ATR'].rolling(20).mean()
    return df

# -------------------------------
# Scoring Engine
# -------------------------------
def score_stock(df, intraday=False):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    if pd.isna(latest['ATR']) or pd.isna(latest['RSI']):
        return None

    if latest['Close'] < PRICE_MIN or latest['Close'] > PRICE_MAX:
        return None

    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]

    if intraday:
        if latest['Volume'] < 1.5 * vol_avg:
            return None
    else:
        if latest['Volume'] < 2 * vol_avg:
            return None

    if latest['ATR'] < 5:
        return None

    if latest['Volume'] >= 3 * vol_avg:
        score += 20

    if not intraday and latest['Close'] > latest['HIGH_20']:
        score += 15

    if not intraday and latest['Close'] > latest['HIGH_50']:
        score += 10

    if latest['RSI'] > 60:
        score += 10

    if latest['RSI'] > prev['RSI']:
        score += 10

    if latest['ATR'] > latest['ATR_AVG_20']:
        score += 10

    expected_move = max(1.5 * latest['ATR'], latest['ATR'] if intraday else 20)

    return {
        "Symbol": "",
        "LTP": round(latest['Close'], 2),
        "Score": score,
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
            df = fetch_intraday(ticker) if MODE == "Intraday" else fetch_daily(ticker)
            if df is None or df.empty or len(df) < 50:
                continue

            df = compute_indicators(df)
            data = score_stock(df, intraday=(MODE == "Intraday"))

            if data and data['Score'] >= MIN_SCORE:
                data['Symbol'] = ticker.replace('.NS', '')
                results.append(data)

        except Exception:
            pass

        progress.progress((i + 1) / len(NSE_STOCKS))

    if results:
        out = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.subheader(f"{MODE} Candidates ({len(out)})")
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇️ Download CSV", out.to_csv(index=False), "scout_results.csv")
    else:
        st.warning("No qualifying stocks found in this scan.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
st.caption("Scouting tool only. Not investment advice.")
