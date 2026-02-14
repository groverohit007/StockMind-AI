import streamlit as st
import plotly.graph_objects as go
import logic
import pandas as pd

st.set_page_config(layout="wide", page_title="HedgeFund Terminal", page_icon="🏦")

# --- SESSION & LOGIN ---
if 'auth' not in st.session_state: st.session_state['auth'] = False
if 'pwd' not in st.session_state: st.session_state['pwd'] = "Arabella@30"

def login():
    st.markdown("<h1 style='text-align: center;'>🏦 HedgeFund Terminal</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("Login", use_container_width=True) or st.text_input("Password", type="password") == st.session_state['pwd']:
            st.session_state['auth'] = True
            st.rerun()

if not st.session_state['auth']:
    login()
    st.stop()

# --- MAIN APP ---
def main():
    # SIDEBAR
    st.sidebar.header("🔍 Market Search")
    ticker = "AAPL"
    query = st.sidebar.text_input("Ticker / Company")
    if query:
        res = logic.search_ticker(query)
        if res: ticker = res[st.sidebar.selectbox("Results", list(res.keys()))]
    else:
        manual = st.sidebar.text_input("Or Symbol", "AAPL").upper()
        if manual: ticker = manual
    
    st.sidebar.markdown("---")
    st.sidebar.header("💷 Portfolio Settings")
    base_curr = st.sidebar.radio("Your Currency", ["GBP (£)", "USD ($)"])
    capital = st.sidebar.number_input(f"Capital ({base_curr[0]})", 10000)
    
    # LOAD SECRETS
    openai_key = st.secrets.get("api", {}).get("openai_key", "")
    
    # TABS
    tabs = st.tabs([
        "📈 Terminal", 
        "🦅 Scanner & Watchdog", 
        "🔙 Backtest", 
        "🌍 Macro & Sectors", 
        "💼 Multi-Currency Portfolio", 
        "⚙️ Settings"
    ])

    # --- TAB 1: TERMINAL ---
    with tabs[0]:
        if ticker:
            st.title(f"{ticker} Pro Analysis")
            
            with st.spinner("Running Multi-Model Consensus..."):
                data = logic.get_data(ticker)
                if data is not None and not data.empty:
                    processed, _, votes = logic.train_consensus_model(data)
                    
                    if processed is not None:
                        last = processed.iloc[-1]
                        conf = last['Confidence']
                        
                        # SIGNAL LOGIC
                        sig = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                        
                        # TOP METRICS
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Price", f"${last['Close']:.2f}")
                        m2.metric("Consensus Signal", sig)
                        m3.metric("AI Confidence", f"{conf*100:.0f}%")
                        m4.metric("Volatility (ATR)", f"{last['ATR']:.2f}")
                        
                        # CHART
                        fig = go.Figure(data=[go.Candlestick(x=processed.index, open=processed['Open'], 
                                              high=processed['High'], low=processed['Low'], close=processed['Close'])])
                        buys = processed[processed['Confidence'] > 0.6]
                        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', 
                                                 marker=dict(color='#00FF00', size=8, symbol='triangle-up'), name='Buy'))
                        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # CONSENSUS DETAILS & ACTION
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.subheader("🗳️ Model Votes")
                            for model, prob in votes.items():
                                st.progress(prob, text=f"{model}: {prob*100:.0f}% Bullish")
                            
                            st.markdown("---")
                            if st.button(f"🛒 Paper Buy {ticker}"):
                                rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                                usd_cap = capital / rate
                                shares = int((usd_cap * 0.02) / (1.5 * last['ATR'])) # 2% risk
                                res = logic.execute_trade(ticker, last['Close'], shares, "BUY", "USD")
                                st.success(f"Bought {shares} shares! ({res})")

                        with c2:
                            st.subheader("🤖 GPT-4 Insight")
                            if st.button("Analyze News"):
                                report, _ = logic.get_ai_analysis(ticker, openai_key)
                                st.info(report)
                else:
                    st.error("Data not available. Try another ticker.")

    # --- TAB 2: SCANNER & WATCHDOG ---
    with tabs[1]:
        st.header("🦅 Market Hunter")
        st.caption("Scan your watchlist for the highest confidence AI setups.")
        
        # SCANNER
        if st.button("🚀 Scan All Watchlist Stocks"):
            with st.spinner("Scanning markets..."):
                scan_res = logic.scan_market()
                if not scan_res.empty:
                    top = scan_res[scan_res['Signal'].str.contains("BUY")]
                    if not top.empty:
                        st.success(f"Found {len(top)} Opportunities!")
                        st.dataframe(top.style.format({"Price": "${:.2f}", "Confidence": "{:.1%}", "RSI": "{:.0f}"}))
                    else:
                        st.info("No strong Buy signals found.")
                    with st.expander("Full Results"):
                        st.dataframe(scan_res.style.format({"Price": "${:.2f}", "Confidence": "{:.1%}"}))
                else:
                    st.warning("Watchlist empty or data error.")
        
        st.markdown("---")
        
        # WATCHDOG MANAGER
        st.subheader("🐶 Watchlist Manager")
        c1, c2 = st.columns([3, 1])
        new_t = c1.text_input("Add Symbol (e.g., TSLA)")
        if c2.button("Add"):
            logic.add_to_watchlist(new_t.upper())
            st.rerun()
            
        watchlist = logic.get_watchlist()
        if watchlist:
            cols = st.columns(5)
            for i, t in enumerate(watchlist):
                if cols[i%5].button(f"❌ {t}"):
                    logic.remove_from_watchlist(t)
                    st.rerun()

    # --- TAB 3: BACKTESTING ---
    with tabs[2]:
        st.header(f"🔙 Time Machine: {ticker}")
        st.caption("Test how the AI would have performed over the last 2 years.")
        
        if st.button("Run Backtest Simulation"):
            with st.spinner("Simulating trades..."):
                # Convert user capital to USD for the test logic
                rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                usd_cap = capital / rate
                
                res, trades, ret = logic.run_backtest(ticker, usd_cap)
                
                if res is not None:
                    # Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Return", f"{ret:.2f}%")
                    c2.metric("Final Balance", f"${res['Equity'].iloc[-1]:.2f}")
                    c3.metric("Total Trades", len(trades) if not trades.empty else 0)
                    
                    # Equity Curve
                    st.line_chart(res['Equity'])
                    
                    # Trade Log
                    st.subheader("Trade Log")
                    st.dataframe(trades)
                else:
                    st.error("Backtest failed due to insufficient data.")

    # --- TAB 4: MACRO & SECTORS ---
    with tabs[3]:
        st.header("🌍 Global Dashboard")
        macro = logic.get_macro_data()
        cols = st.columns(len(macro))
        for i, (k, v) in enumerate(macro.items()):
            cols[i].metric(k, f"{v['Price']:.2f}", f"{v['Change']:.2f}%")
        
        st.markdown("---")
        st.subheader("🔥 US Sector Heatmap")
        sectors = logic.get_sector_heatmap()
        s_cols = st.columns(4)
        for i, (sect, chg) in enumerate(sectors.items()):
            color = "green" if chg > 0 else "red"
            s_cols[i % 4].markdown(f"""
                <div style="background-color: {'#1e3d1e' if chg>0 else '#3d1e1e'}; 
                            padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                    <strong>{sect}</strong><br>
                    <span style="color: {color}; font-size: 1.2em;">{chg:+.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

    # --- TAB 5: PORTFOLIO ---
    with tabs[4]:
        st.header(f"💼 Paper Portfolio ({base_curr})")
        df = logic.get_portfolio()
        if not df.empty and not df[df['Status']=='OPEN'].empty:
            rate_usd_to_base = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
            
            open_pos = df[df['Status']=='OPEN'].copy()
            total_val = 0
            
            st.subheader("Open Positions")
            header = st.columns([2, 1, 1, 1, 1])
            header[0].write("**Ticker**")
            header[1].write("**Shares**")
            header[2].write("**Avg Cost**")
            header[3].write("**Value**")
            header[4].write("**Action**")
            
            for i, row in open_pos.iterrows():
                live_data = logic.get_data(row['Ticker'], period="1d", interval="1m")
                curr_price = live_data['Close'].iloc[-1] if live_data is not None else row['Buy_Price_USD']
                
                val_usd = curr_price * row['Shares']
                val_base = val_usd * rate_usd_to_base
                cost_base = (row['Buy_Price_USD'] * row['Shares']) * rate_usd_to_base
                pnl = val_base - cost_base
                total_val += val_base
                
                c = st.columns([2, 1, 1, 1, 1])
                c[0].write(f"**{row['Ticker']}**")
                c[1].write(f"{row['Shares']}")
                c[2].write(f"{base_curr[0]}{cost_base/row['Shares']:.2f}")
                c[3].write(f"{base_curr[0]}{val_base:.2f} (: {'green' if pnl>0 else 'red'}[{pnl:+.2f}])")
                if c[4].button("Sell", key=f"s_{i}"):
                    logic.execute_trade(row['Ticker'], curr_price, 0, "SELL")
                    st.rerun()
            st.metric("Total Portfolio Value", f"{base_curr[0]}{total_val:.2f}")
        else:
            st.info("No open trades.")

    # --- TAB 6: SETTINGS ---
    with tabs[5]:
        st.header("⚙️ Settings")
        st.info("API Keys loaded from secrets.toml")
        if st.button("Logout"):
            st.session_state['auth'] = False
            st.rerun()

main()
