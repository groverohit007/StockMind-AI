import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logic
import time

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="ProTrader AI", page_icon="⚡")

# --- SESSION STATE INITIALIZATION ---
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'app_password' not in st.session_state: st.session_state['app_password'] = "Arabella@30"
# Load keys from secrets.toml if available, else default to empty
if 'openai_key' not in st.session_state: 
    st.session_state['openai_key'] = st.secrets.get("api", {}).get("openai_key", "")
if 'tele_token' not in st.session_state: 
    st.session_state['tele_token'] = st.secrets.get("api", {}).get("telegram_token", "")
if 'tele_chat' not in st.session_state: 
    st.session_state['tele_chat'] = st.secrets.get("api", {}).get("telegram_chat_id", "")

# --- 1. LOGIN SCREEN ---
def login_screen():
    st.markdown("<h1 style='text-align: center;'>⚡ ProTrader AI Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        password_input = st.text_input("Enter Password", type="password")
        if st.button("Login", use_container_width=True):
            if password_input == st.session_state['app_password']:
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("❌ Access Denied")

# --- 2. MAIN APPLICATION ---
def main_app():
    if data is not None:
                market_stat = logic.get_market_status()
                
                # RUN MODEL WITH SAFETY CHECK
                processed, features = logic.train_model(data)
                
                # NEW: Check if processed data is valid
                if processed is None or processed.empty:
                    st.error(f"⚠️ Not enough data to analyze {ticker}. This often happens with ETFs (like SGLN), new listings, or if the timeframe is too short.")
                else:
                    # ... The rest of your code runs ONLY if data is valid ...
                    last_row = processed.iloc[-1]
                    conf = last_row['Confidence']
    
    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("🔍 Market Search")
    
    # Ticker Search
    query = st.sidebar.text_input("Search (e.g., Tesla, EURUSD)")
    ticker = "AAPL" # Default fallback
    if query:
        res = logic.search_ticker(query)
        if res:
            sel = st.sidebar.selectbox("Select Asset", list(res.keys()))
            ticker = res[sel]
        else:
            st.sidebar.warning("No results found.")
            ticker = query.upper() # Use raw input if search fails

    # Trading Mode
    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ Timeframe Strategy")
    mode = st.sidebar.radio("Mode", ["Swing (Daily)", "Day Trading (Intraday)"])
    
    if "Day Trading" in mode:
        interval = st.sidebar.selectbox("Candle Size", ["15m", "30m", "60m", "1h"])
        st.sidebar.caption("⚡ Fast-paced analysis")
    else:
        interval = "1d"
        st.sidebar.caption("🐢 Trend analysis")

    # Risk Management
    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ Risk Settings")
    capital = st.sidebar.number_input("Account Balance ($)", value=10000)
    risk = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0)

    # --- MAIN TABS (Defined OUTSIDE of 'if ticker' to prevent errors) ---
    tabs = st.tabs(["📈 Chart & Signals", "🐶 Watchdog", "⚙️ Settings"])

    # --- TAB 1: CHART & ANALYSIS ---
    with tabs[0]:
        if ticker:
            st.title(f"{ticker} Analysis ({interval})")
            
            # Fetch Data
            with st.spinner(f"Fetching {interval} data..."):
                data = logic.get_data(ticker, interval=interval)
            
            if data is not None:
                market_stat = logic.get_market_status()
                processed, features = logic.train_model(data)
                last_row = processed.iloc[-1]
                
                # Signal Logic
                conf = last_row['Confidence']
                signal = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "WAIT ⚪"
                
                # Top Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"${last_row['Close']:.2f}")
                m2.metric("Signal", signal)
                m3.metric("AI Confidence", f"{conf*100:.0f}%")
                m4.metric("Market Trend", market_stat)

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=processed.index,
                    open=processed['Open'], high=processed['High'],
                    low=processed['Low'], close=processed['Close'], name='Price'))
                
                # Moving Averages
                fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_20'], 
                                         line=dict(color='orange', width=1), name='SMA 20'))
                fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_50'], 
                                         line=dict(color='blue', width=1), name='SMA 50'))

                # Buy Markers
                buys = processed[processed['Confidence'] > 0.6]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', 
                    marker=dict(color='#00FF00', size=8, symbol='triangle-up'), name='Buy Signal'))

                fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # Action Area
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🛡️ Risk Calculator")
                    shares, sl_dist = logic.calculate_position_size(capital, risk, last_row['Close'], last_row['ATR'])
                    stop_price = last_row['Close'] - sl_dist
                    st.info(f"""
                    **Risking {risk}% (${capital * (risk/100):.0f})**
                    
                    👉 **Position Size:** {shares} Shares
                    🛑 **Stop Loss:** ${stop_price:.2f}
                    """)
                
                with c2:
                    st.subheader("🤖 AI Analyst")
                    if st.button("Analyze News"):
                        report, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                        st.success(report)
                    
                    if st.button("Send Telegram Alert"):
                        res = logic.send_telegram_alert(st.session_state['tele_token'], 
                                                        st.session_state['tele_chat'], 
                                                        ticker, signal, last_row['Close'], interval)
                        st.info(res)
            else:
                st.error(f"Data unavailable for {ticker}. Try a different interval or stock.")

    # --- TAB 2: WATCHDOG ---
    with tabs[1]:
        st.header("🐶 Watchdog Manager")
        st.caption("The 24/7 Bot will scan these stocks.")
        
        # Display Watchlist
        watchlist = logic.get_watchlist()
        if watchlist:
            st.write("### Currently Watching:")
            cols = st.columns(4)
            for i, stock in enumerate(watchlist):
                col = cols[i % 4]
                col.info(f"**{stock}**")
                if col.button(f"Remove {stock}", key=f"rem_{stock}"):
                    logic.remove_from_watchlist(stock)
                    st.rerun()
        else:
            st.warning("Watchlist is empty.")

        st.markdown("---")
        # Add New Stock
        c1, c2 = st.columns([3, 1])
        new_ticker = c1.text_input("Add Ticker", placeholder="NVDA").upper()
        if c2.button("Add to Watchdog"):
            if new_ticker:
                logic.add_to_watchlist(new_ticker)
                st.success(f"Added {new_ticker}")
                time.sleep(1)
                st.rerun()

    # --- TAB 3: SETTINGS ---
    with tabs[2]:
        st.header("⚙️ Settings")
        st.info("Enter keys here if not using secrets.toml")
        
        st.session_state['openai_key'] = st.text_input("OpenAI API Key", 
                                                       value=st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Telegram Bot Token", 
                                                       value=st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Telegram Chat ID", 
                                                      value=st.session_state['tele_chat'], type="password")
        
        st.markdown("---")
        new_pass = st.text_input("Change App Password", type="password")
        if st.button("Update Password"):
            if new_pass:
                st.session_state['app_password'] = new_pass
                st.success("Password Updated!")

# --- RUN APP ---
if not st.session_state['authenticated']:
    login_screen()
else:
    main_app()

