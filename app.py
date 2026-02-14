import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logic
import time

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="ProTrader AI", page_icon="⚡")

# --- SESSION STATE ---
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'app_password' not in st.session_state: st.session_state['app_password'] = "Arabella@30"

# Load secrets if available, else empty
if 'openai_key' not in st.session_state: 
    st.session_state['openai_key'] = st.secrets.get("api", {}).get("openai_key", "")
if 'tele_token' not in st.session_state: 
    st.session_state['tele_token'] = st.secrets.get("api", {}).get("telegram_token", "")
if 'tele_chat' not in st.session_state: 
    st.session_state['tele_chat'] = st.secrets.get("api", {}).get("telegram_chat_id", "")

# --- LOGIN ---
def login_screen():
    st.markdown("<h1 style='text-align: center;'>⚡ ProTrader AI Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("Login", use_container_width=True):
            if pwd == st.session_state['app_password']:
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("❌ Access Denied")

# --- MAIN APP ---
def main_app():
    # SIDEBAR
    st.sidebar.header("🔍 Market & Mode")
    
    # Search
    query = st.sidebar.text_input("Search Ticker (e.g., Gold, Apple)")
    ticker = "AAPL" # Default
    if query:
        res = logic.search_ticker(query)
        if res:
            sel = st.sidebar.selectbox("Select Result", list(res.keys()))
            ticker = res[sel]
    else:
        # Fallback manual entry
        manual = st.sidebar.text_input("Or Enter Symbol", "").upper()
        if manual: ticker = manual

    # Mode
    mode = st.sidebar.radio("Mode", ["Swing (Daily)", "Day Trading"])
    interval = "1d"
    if "Day" in mode:
        interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h"])
    
    # Risk
    st.sidebar.markdown("---")
    capital = st.sidebar.number_input("Capital ($)", 10000)
    risk = st.sidebar.slider("Risk (%)", 0.5, 3.0, 1.0)

    # TABS
    tabs = st.tabs(["📈 Dashboard", "🐶 Watchdog", "⚙️ Settings"])

    # --- TAB 1: DASHBOARD ---
    with tabs[0]:
        if ticker:
            st.title(f"{ticker} Analysis ({interval})")
            
            with st.spinner("Analyzing data..."):
                data = logic.get_data(ticker, interval=interval)
            
            # CHECK 1: Did we get data?
            if data is not None and not data.empty:
                processed, features = logic.train_model(data)
                
                # CHECK 2: Is the processed data valid? (Fix for SGLN error)
                if processed is None or processed.empty:
                    st.error(f"⚠️ **Insufficient Data for {ticker}**")
                    st.warning("This happens with ETFs/Commodities (like SGLN) that have low volume or gaps. Try a larger timeframe (Daily) or a different stock.")
                else:
                    # RENDER DASHBOARD
                    last_row = processed.iloc[-1]
                    conf = last_row['Confidence']
                    signal = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "WAIT ⚪"
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"${last_row['Close']:.2f}")
                    m2.metric("Signal", signal)
                    m3.metric("AI Confidence", f"{conf*100:.0f}%")
                    m4.metric("Market Trend", logic.get_market_status())

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=processed.index,
                        open=processed['Open'], high=processed['High'],
                        low=processed['Low'], close=processed['Close'], name='Price'))
                    
                    # Buy Markers
                    buys = processed[processed['Confidence'] > 0.6]
                    if not buys.empty:
                        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', 
                            marker=dict(color='#00FF00', size=8, symbol='triangle-up'), name='Buy Signal'))
                    
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Actions
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🛡️ Risk Calc")
                        shares, stop_dist = logic.calculate_position_size(capital, risk, last_row['Close'], last_row['ATR'])
                        st.info(f"**Buy:** {shares} Shares | **Stop Loss:** ${last_row['Close']-stop_dist:.2f}")
                    
                    with c2:
                        st.subheader("🤖 AI Tools")
                        if st.button("Analyze News"):
                            report, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                            st.success(report)
                        if st.button("Send Alert"):
                            res = logic.send_telegram_alert(st.session_state['tele_token'], st.session_state['tele_chat'], ticker, signal, last_row['Close'], interval)
                            st.success(res)
            else:
                st.error("❌ Data not found. Check the ticker or your internet.")

    # --- TAB 2: WATCHDOG ---
    with tabs[1]:
        st.header("🐶 24/7 Watchlist")
        st.caption("Stocks saved here are monitored by the Background Bot.")
        
        # Add
        c1, c2 = st.columns([3,1])
        new_t = c1.text_input("Add Stock").upper()
        if c2.button("Add"):
            if new_t: 
                logic.add_to_watchlist(new_t)
                st.rerun()
        
        # List
        watchlist = logic.get_watchlist()
        if watchlist:
            cols = st.columns(4)
            for i, stock in enumerate(watchlist):
                col = cols[i%4]
                col.info(stock)
                if col.button(f"Del {stock}", key=stock):
                    logic.remove_from_watchlist(stock)
                    st.rerun()
        else:
            st.warning("Watchlist empty.")

    # --- TAB 3: SETTINGS ---
    with tabs[2]:
        st.header("⚙️ Settings")
        st.session_state['openai_key'] = st.text_input("OpenAI Key", value=st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Telegram Token", value=st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Telegram Chat ID", value=st.session_state['tele_chat'], type="password")
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.rerun()

# Run
if not st.session_state['authenticated']:
    login_screen()
else:
    main_app()
