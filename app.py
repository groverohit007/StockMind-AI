import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logic
import time

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="ProTrader AI", page_icon="🔐")

# --- SESSION STATE INITIALIZATION ---
# This keeps your password and API keys saved while you use the app
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'app_password' not in st.session_state:
    st.session_state['app_password'] = "Arabella@30" # Initial Password
if 'openai_key' not in st.session_state:
    st.session_state['openai_key'] = ""
if 'tele_token' not in st.session_state:
    st.session_state['tele_token'] = ""
if 'tele_chat' not in st.session_state:
    st.session_state['tele_chat'] = ""

# --- 1. LOGIN SCREEN ---
def login_screen():
    st.markdown("<h1 style='text-align: center;'>🔒 ProTrader AI Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        password_input = st.text_input("Enter Password", type="password")
        if st.button("Login", use_container_width=True):
            if password_input == st.session_state['app_password']:
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("❌ Access Denied")

# --- 2. MAIN APPLICATION ---
def main_app():
    # Top Navigation Tabs
    main_tab, settings_tab = st.tabs(["📈 Dashboard", "⚙️ Settings"])

    # --- TAB A: DASHBOARD ---
    with main_tab:
        st.sidebar.header("🔍 Market Search")
        
        # 1. SMART SEARCH BAR
        search_query = st.sidebar.text_input("Search Company (e.g., Apple, Rolls Royce)")
        
        ticker = None
        if search_query:
            results = logic.search_ticker(search_query)
            if results:
                selected_label = st.sidebar.selectbox("Select Stock", list(results.keys()))
                ticker = results[selected_label] # Get the actual symbol (e.g., AAPL)
            else:
                st.sidebar.warning("No results found.")
        
        # Default if empty
        if not ticker:
            ticker = st.sidebar.text_input("Or enter Symbol manually", "AAPL").upper()

        # Risk Inputs
        st.sidebar.markdown("---")
        capital = st.sidebar.number_input("Capital ($)", 10000)
        risk = st.sidebar.slider("Risk (%)", 1, 5, 2)
        
        # --- DASHBOARD LOGIC ---
        if ticker:
            st.title(f"📊 Analysis: {ticker}")
            
            # Fetch Data
            with st.spinner("Analyzing Market Data..."):
                data = logic.get_data(ticker)
            
            if data is not None:
                market_status = logic.get_market_status()
                processed, features = logic.train_model(data)
                last_row = processed.iloc[-1]
                
                # Signal Logic
                conf = last_row['Confidence']
                signal = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${last_row['Close']:.2f}")
                m2.metric("Signal", signal)
                m3.metric("AI Confidence", f"{conf*100:.0f}%")
                m4.metric("Market Trend", market_status)
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=processed.index, 
                                             open=processed['Open'], high=processed['High'],
                                             low=processed['Low'], close=processed['Close'], name='Price'))
                # Add Buy Markers
                buys = processed[processed['Confidence'] > 0.6]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', 
                                         marker=dict(color='green', size=8, symbol='triangle-up'), name='Buy Zone'))
                
                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Action Buttons
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🤖 AI Opinion")
                    if st.button("Run AI Analysis"):
                        report, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                        st.info(report)
                
                with c2:
                    st.subheader("📢 Alerts")
                    if st.button("Send Telegram Alert"):
                        res = logic.send_telegram_alert(st.session_state['tele_token'], 
                                                        st.session_state['tele_chat'], 
                                                        ticker, signal, last_row['Close'])
                        st.success(res)

    # --- TAB B: SETTINGS ---
    with settings_tab:
        st.header("⚙️ Application Settings")
        
        st.subheader("🔑 API Keys")
        st.info("Keys are saved temporarily for this session.")
        
        # Input fields that update session state
        st.session_state['openai_key'] = st.text_input("OpenAI API Key", 
                                                       value=st.session_state['openai_key'], type="password")
        
        st.session_state['tele_token'] = st.text_input("Telegram Bot Token", 
                                                       value=st.session_state['tele_token'], type="password")
        
        st.session_state['tele_chat'] = st.text_input("Telegram Chat ID", 
                                                      value=st.session_state['tele_chat'], type="password")
        
        st.markdown("---")
        st.subheader("🔐 Security")
        new_pass = st.text_input("Change App Password", type="password")
        if st.button("Update Password"):
            if new_pass:
                st.session_state['app_password'] = new_pass
                st.success("Password Updated! Don't forget it.")

# --- APP FLOW CONTROL ---
if not st.session_state['authenticated']:
    login_screen()
else:
    main_app()