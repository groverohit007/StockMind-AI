import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logic
import time

st.set_page_config(layout="wide", page_title="ProTrader AI", page_icon="⚡")

# --- AUTH & SESSION ---
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'app_password' not in st.session_state: st.session_state['app_password'] = "Arabella@30"
if 'openai_key' not in st.session_state: st.session_state['openai_key'] = ""
if 'tele_token' not in st.session_state: st.session_state['tele_token'] = ""
if 'tele_chat' not in st.session_state: st.session_state['tele_chat'] = ""

def login_screen():
    st.markdown("<h1 style='text-align: center;'>⚡ ProTrader AI Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("Login", use_container_width=True) or st.text_input("Password", type="password") == st.session_state['app_password']:
            st.session_state['authenticated'] = True
            st.rerun()

def main_app():
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🔍 Market & Timeframe")
    
    # 1. Ticker Search
    query = st.sidebar.text_input("Search (e.g., Tesla, EURUSD)")
    ticker = "AAPL" # Default
    if query:
        res = logic.search_ticker(query)
        if res:
            sel = st.sidebar.selectbox("Select", list(res.keys()))
            ticker = res[sel]
    
    # 2. TRADING MODE (New Feature)
    mode = st.sidebar.radio("Trading Mode", ["Swing (Daily)", "Day Trading (Intraday)"])
    
    if "Day Trading" in mode:
        interval = st.sidebar.selectbox("Interval", ["15m", "30m", "60m", "1h"])
        st.sidebar.info(f"⚡ fast-paced mode: {interval} candles")
    else:
        interval = "1d"
        st.sidebar.info("🐢 Slow-paced mode: Daily candles")

    # 3. Risk Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ Risk Management")
    capital = st.sidebar.number_input("Account Balance ($)", 10000)
    risk = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)

    # --- MAIN CONTENT ---
    tabs = st.tabs(["📈 Chart & Signals", "⚙️ Settings"])

    with tabs[0]:
        if ticker:
            # Fetch Data based on Interval
            with st.spinner(f"Fetching {interval} data for {ticker}..."):
                data = logic.get_data(ticker, interval=interval)
            
            if data is not None:
                market_stat = logic.get_market_status()
                processed, features = logic.train_model(data)
                last_row = processed.iloc[-1]
                
                # Signal Logic
                conf = last_row['Confidence']
                signal = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "WAIT ⚪"
                
                # Header Metrics
                st.title(f"{ticker} Analysis ({interval})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"${last_row['Close']:.2f}")
                m2.metric("Signal", signal)
                m3.metric("AI Confidence", f"{conf*100:.0f}%")
                m4.metric("Market Trend", market_stat)

                # --- CHART SECTION ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=processed.index,
                    open=processed['Open'], high=processed['High'],
                    low=processed['Low'], close=processed['Close'], name='Price'))
                
                # Add Moving Averages
                fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
                fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_50'], line=dict(color='blue', width=1), name='SMA 50'))

                # Buy Markers
                buys = processed[processed['Confidence'] > 0.6]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', 
                    marker=dict(color='#00FF00', size=8, symbol='triangle-up'), name='Buy Signal'))

                fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- RISK & AI SECTION ---
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🛡️ Risk Calculator")
                    shares, sl_dist = logic.calculate_position_size(capital, risk, last_row['Close'], last_row['ATR'])
                    stop_price = last_row['Close'] - sl_dist
                    
                    st.info(f"""
                    **Risking {risk}% (${capital * (risk/100):.0f})**
                    
                    👉 **Buy:** {shares} Shares
                    🛑 **Stop Loss:** ${stop_price:.2f}
                    """)
                
                with c2:
                    st.subheader("🤖 AI Analyst")
                    if st.button("Analyze News"):
                        report, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                        st.success(report)
                    
                    if st.button("Send Alert to Telegram"):
                        res = logic.send_telegram_alert(st.session_state['tele_token'], 
                                                        st.session_state['tele_chat'], 
                                                        ticker, signal, last_row['Close'], interval)
                        st.info(res)
            else:
                st.error("Data not available for this timeframe. Try a larger stock or '1h' interval.")

    with tabs[1]:
        st.header("🔑 API Keys")
        st.session_state['openai_key'] = st.text_input("OpenAI Key", value=st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Telegram Token", value=st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Telegram Chat ID", value=st.session_state['tele_chat'], type="password")

if not st.session_state['authenticated']:
    login_screen()
else:
    main_app()
