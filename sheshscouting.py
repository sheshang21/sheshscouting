"""
₹20-₹30 Stock Scout — India
Production-Ready Streamlit App for Indian Equity Market Scanning
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="₹20-₹30 Stock Scout — India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NSE Top 200+ Stocks (Production List)
NSE_STOCKS = [
    # NIFTY 50
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
    'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
    'TITAN.NS', 'AXISBANK.NS', 'ULTRACEMCO.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS',
    'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'JSWSTEEL.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'INDUSINDBK.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
    'APOLLOHOSP.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'TATACONSUM.NS',
    'BAJAJ-AUTO.NS', 'SHRIRAMFIN.NS', 'BPCL.NS', 'LTIM.NS', 'SBILIFE.NS', 'TRENT.NS', 'HDFCLIFE.NS',
    
    # NIFTY NEXT 50 + Midcaps
    'VEDL.NS', 'ADANIGREEN.NS', 'GODREJCP.NS', 'SIEMENS.NS', 'DLF.NS', 'PIDILITIND.NS', 'ICICIGI.NS',
    'DABUR.NS', 'GAIL.NS', 'AMBUJACEM.NS', 'BANKBARODA.NS', 'HAVELLS.NS', 'DMART.NS', 'INDIGO.NS',
    'ADANIPOWER.NS', 'CHOLAFIN.NS', 'JINDALSTEL.NS', 'MOTHERSON.NS', 'PNB.NS', 'TORNTPHARM.NS',
    'BOSCHLTD.NS', 'SRF.NS', 'MCDOWELL-N.NS', 'BEL.NS', 'BERGEPAINT.NS', 'COLPAL.NS', 'ZOMATO.NS',
    'TATAPOWER.NS', 'CANBK.NS', 'UNIONBANK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS',
    'LUPIN.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'ALKEM.NS', 'MARICO.NS', 'SAIL.NS', 'NMDC.NS',
    'POLYCAB.NS', 'CROMPTON.NS', 'VOLTAS.NS', 'DIXON.NS', 'CUMMINSIND.NS', 'ABB.NS', 'THERMAX.NS',
    'LTTS.NS', 'TATAELXSI.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'MPHASIS.NS', 'OFSS.NS',
    'LALPATHLAB.NS', 'METROPOLIS.NS', 'JUBLFOOD.NS', 'BALKRISIND.NS', 'MRF.NS', 'APOLLOTYRE.NS',
    'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'IGL.NS', 'MGL.NS', 'PETRONET.NS',
    'INDHOTEL.NS', 'IRCTC.NS', 'CONCOR.NS', 'HAL.NS', 'BHEL.NS', 'BDL.NS',
    'RECLTD.NS', 'PFC.NS', 'LICHSGFIN.NS', 'AARTI.NS', 'DEEPAKNTR.NS', 'BALRAMCHIN.NS',
    'NATIONALUM.NS', 'HINDZINC.NS', 'RATNAMANI.NS', 'APL.NS', 'ESCORTS.NS', 'ASHOKLEY.NS',
]

# Sector Mapping
SECTOR_MAP = {
    'RELIANCE.NS': 'Oil & Gas', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Banking', 'INFY.NS': 'IT',
    'ICICIBANK.NS': 'Banking', 'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG', 'SBIN.NS': 'PSU Bank',
    'BHARTIARTL.NS': 'Telecom', 'KOTAKBANK.NS': 'Banking', 'LT.NS': 'Infrastructure', 'BAJFINANCE.NS': 'NBFC',
    'ASIANPAINT.NS': 'Paints', 'MARUTI.NS': 'Auto', 'HCLTECH.NS': 'IT', 'TITAN.NS': 'Consumer Durables',
    'AXISBANK.NS': 'Banking', 'ULTRACEMCO.NS': 'Cement', 'SUNPHARMA.NS': 'Pharma', 'WIPRO.NS': 'IT',
    'TATAMOTORS.NS': 'Auto', 'TATASTEEL.NS': 'Metals', 'ADANIENT.NS': 'Conglomerate', 'COALINDIA.NS': 'Mining',
}

# ================== HELPER FUNCTIONS ==================

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0
    except:
        return 0

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    try:
        if len(df) == 0:
            return 0
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).sum() / df['Volume'].sum()
        
        return vwap if not pd.isna(vwap) else 0
    except:
        return 0

def fetch_stock_data(symbol):
    """Fetch and process stock data for a single symbol"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')
        
        if hist.empty or len(hist) < 50:
            return None
        
        # Latest data
        latest = hist.iloc[-1]
        ltp = latest['Close']
        volume = latest['Volume']
        
        if ltp <= 0 or volume <= 0:
            return None
        
        # Calculate indicators
        atr = calculate_atr(hist)
        rsi = calculate_rsi(hist)
        
        # Volume analysis
        avg_volume_20 = hist['Volume'].tail(20).mean()
        volume_multiplier = volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # Price levels
        prev_high = hist['High'].iloc[-2] if len(hist) > 1 else ltp
        day_20_high = hist['High'].tail(20).max()
        
        # VWAP (today's data)
        today_data = hist.tail(1)
        vwap = calculate_vwap(today_data)
        
        # Traded value
        traded_value = (ltp * volume) / 1e7  # in crores
        
        # Calculate score
        score = 0
        
        # Volume criteria
        if volume_multiplier >= 3:
            score += 20
        elif volume_multiplier >= 2:
            score += 10
        
        # Breakout criteria
        if ltp > day_20_high * 0.99:  # Within 1% of 20-day high
            score += 15
        if ltp > prev_high:
            score += 10
        
        # RSI criteria
        if rsi > 60:
            score += 10
        
        # VWAP criteria
        if ltp > vwap * 0.99:  # Within 1% above VWAP
            score += 10
        
        # ATR expansion
        if atr > ltp * 0.025:
            score += 10
        
        # Momentum check
        if len(hist) >= 5:
            recent_close = hist['Close'].tail(5)
            if recent_close.is_monotonic_increasing:
                score += 15
        
        # Expected move calculation
        expected_move = max(atr * 1.5, ltp * 0.03)
        expected_move_pct = (expected_move / ltp) * 100
        
        # Breakout type
        breakout_types = []
        if ltp > vwap:
            breakout_types.append('VWAP')
        if ltp > prev_high:
            breakout_types.append('DayHigh')
        if ltp > day_20_high * 0.99:
            breakout_types.append('20DH')
        
        breakout_type = ', '.join(breakout_types) if breakout_types else 'Range'
        
        # Sector
        sector = SECTOR_MAP.get(symbol, 'Others')
        
        return {
            'Symbol': symbol.replace('.NS', ''),
            'LTP': round(ltp, 2),
            'Score': int(score),
            'Volume Multiplier': round(volume_multiplier, 2),
            'ATR': round(atr, 2),
            'Expected Move (₹)': round(expected_move, 2),
            'Expected Move (%)': round(expected_move_pct, 2),
            'RSI': round(rsi, 1),
            'Sector': sector,
            'Breakout Type': breakout_type,
            'Traded Value (Cr)': round(traded_value, 2),
            'Volume': int(volume),
            'Avg Volume 20D': int(avg_volume_20)
        }
    
    except Exception as e:
        return None

def scan_market(stocks_list, progress_bar, status_text):
    """Scan entire market with parallel processing"""
    results = []
    total = len(stocks_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in stocks_list}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f'📊 Scanned {completed}/{total} stocks... Found {len(results)} opportunities')
            
    return pd.DataFrame(results)

def apply_filters(df, min_price, max_price, min_score, min_atr, min_traded_value, min_volume_mult, min_expected_move):
    """Apply hard filters to stock data"""
    if df.empty:
        return df
    
    filtered = df[
        (df['LTP'] >= min_price) &
        (df['LTP'] <= max_price) &
        (df['Score'] >= min_score) &
        (df['ATR'] >= min_atr) &
        (df['Traded Value (Cr)'] >= min_traded_value) &
        (df['Volume Multiplier'] >= min_volume_mult) &
        (df['Expected Move (₹)'] >= min_expected_move)
    ]
    
    return filtered

# ================== STREAMLIT UI ==================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main {background-color: #0e1117;}
        .stButton>button {
            background: linear-gradient(90deg, #00d4aa 0%, #00a8e8 100%);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #00a8e8 0%, #00d4aa 100%);
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            color: #00d4aa;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 📈 ₹20–₹30 Stock Scout — India")
    st.markdown("### Live Market Scanner • NSE Equity • Production Ready")
    
    # Disclaimer
    st.warning("⚠️ **Disclaimer:** This is a scouting tool for educational purposes only. Not investment advice. "
               "Past patterns do not guarantee future results. Always do your own research and consult a financial advisor.")
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters & Settings")
    
    min_score = st.sidebar.slider("Minimum Score", 50, 100, 65, 5)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_price = st.number_input("Min Price (₹)", value=100, step=50)
    with col2:
        max_price = st.number_input("Max Price (₹)", value=1200, step=100)
    
    min_atr = st.sidebar.number_input("Min ATR (₹)", value=5.0, step=1.0)
    min_traded_value = st.sidebar.number_input("Min Traded Value (Cr)", value=20.0, step=5.0)
    min_volume_mult = st.sidebar.slider("Min Volume Multiplier", 1.0, 5.0, 2.0, 0.5)
    min_expected_move = st.sidebar.number_input("Min Expected Move (₹)", value=20.0, step=5.0)
    
    selected_sectors = st.sidebar.multiselect(
        "Filter by Sector",
        options=sorted(set(SECTOR_MAP.values())),
        default=[]
    )
    
    # Main controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🔄 Refresh Scan", use_container_width=True):
            st.session_state['scan_trigger'] = True
    
    with col2:
        st.metric("Total Stocks", len(NSE_STOCKS))
    
    with col3:
        if 'results_df' in st.session_state and st.session_state.get('results_df') is not None:
            st.metric("Opportunities", len(st.session_state['results_df']))
        else:
            st.metric("Opportunities", 0)
    
    with col4:
        if 'last_update' in st.session_state:
            st.caption(f"Updated: {st.session_state['last_update']}")
    
    # Scan logic
    if st.session_state.get('scan_trigger', False):
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text('🚀 Initializing market scan...')
        
        # Scan market
        raw_df = scan_market(NSE_STOCKS, progress_bar, status_text)
        
        if not raw_df.empty:
            # Apply filters
            filtered_df = apply_filters(
                raw_df,
                min_price,
                max_price,
                min_score,
                min_atr,
                min_traded_value,
                min_volume_mult,
                min_expected_move
            )
            
            if selected_sectors:
                filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
            
            # Sort by score
            filtered_df = filtered_df.sort_values('Score', ascending=False).reset_index(drop=True)
            
            # Store in session
            st.session_state['results_df'] = filtered_df
            st.session_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            st.session_state['results_df'] = pd.DataFrame()
        
        st.session_state['scan_trigger'] = False
        
        status_text.text(f'✅ Scan complete! Found {len(st.session_state.get("results_df", []))} opportunities')
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        st.rerun()
    
    # Display results
    if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
        df = st.session_state['results_df']
        
        if len(df) > 0:
            # Key metrics
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Avg Score", f"{df['Score'].mean():.1f}")
            with col2:
                st.metric("Avg Move (₹)", f"₹{df['Expected Move (₹)'].mean():.2f}")
            with col3:
                st.metric("Avg RSI", f"{df['RSI'].mean():.1f}")
            with col4:
                st.metric("Avg Vol Mult", f"{df['Volume Multiplier'].mean():.2f}x")
            with col5:
                top_sector = df['Sector'].mode()[0] if len(df) > 0 else "N/A"
                st.metric("Top Sector", top_sector)
            
            # Results table
            st.markdown("### 🎯 Top Opportunities")
            
            # Display columns
            display_df = df[[
                'Symbol', 'LTP', 'Score', 'Volume Multiplier', 'ATR',
                'Expected Move (₹)', 'Expected Move (%)', 'RSI', 'Sector', 'Breakout Type'
            ]]
            
            # Color coding
            def highlight_score(row):
                score = row['Score']
                if score >= 80:
                    return ['background-color: #10b981; color: white; font-weight: bold'] * len(row)
                elif score >= 70:
                    return ['background-color: #3b82f6; color: white; font-weight: bold'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=csv,
                file_name=f"stock_scout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Sector heatmap
            st.markdown("### 📊 Sector Distribution")
            sector_counts = df['Sector'].value_counts().head(10)
            st.bar_chart(sector_counts)
            
        else:
            st.info("ℹ️ No stocks match the current filter criteria. Try adjusting the filters or run a new scan.")
    else:
        st.info("👆 Click 'Refresh Scan' to start scanning the market!")
        st.session_state['scan_trigger'] = True
        st.rerun()

if __name__ == "__main__":
    main()
Free Indian stock scanner for ₹20–₹30 moves - Claude
