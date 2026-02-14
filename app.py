import streamlit as st
import plotly.graph_objects as go
import logic
import pandas as pd

st.set_page_config(layout="wide", page_title="HedgeFund Terminal", page_icon="🏦")

# --- SESSION & LOGIN ---
if 'auth' not in st.session_state: st.session_state['auth'] = False
if 'pwd' not in st.session_state: st.session_state['pwd'] = "Arabella@30"

# Secrets
if 'openai_key' not in st.session_state: 
    st.session_state['openai_key'] = st.secrets.get("api", {}).get("openai_key", "")
if 'tele_token' not in st.session_state: 
    st.session_state['tele_token'] = st.secrets.get("api", {}).get("telegram_token", "")
if 'tele_chat' not in st.session_state: 
    st.session_state['tele_chat'] = st.secrets.get("api", {}).get("telegram_chat_id", "")

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

def main():
    # --- SIDEBAR: SEARCH & FUNDAMENTALS ---
    st.sidebar.header("🔍 Market Search")
    ticker = "AAPL"
    query = st.sidebar.text_input("Ticker / Company")
    if query:
        res = logic.search_ticker(query)
        if res: ticker = res[st.sidebar.selectbox("Results", list(res.keys()))]
    else:
        manual = st.sidebar.text_input("Or Symbol", "AAPL").upper()
        if manual: ticker = manual

    # NEW: FUNDAMENTAL HEALTH CHECK WIDGET
    if ticker:
        st.sidebar.markdown("---")
        st.sidebar.header("🏢 Health Check")
        fund = logic.get_fundamentals(ticker)
        if fund:
            st.sidebar.caption(f"{fund['Industry']} | {fund['Sector']}")
            
            # Formatting Market Cap
            mc = fund['Market Cap']
            if isinstance(mc, (int, float)):
                if mc > 1e12: mc_str = f"${mc/1e12:.2f}T"
                elif mc > 1e9: mc_str = f"${mc/1e9:.2f}B"
                else: mc_str = f"${mc/1e6:.2f}M"
            else: mc_str = "N/A"
            
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Mkt Cap", mc_str)
            c2.metric("Beta", fund['Beta'])
            
            c3, c4 = st.sidebar.columns(2)
            c3.metric("P/E Ratio", f"{fund['P/E Ratio']}")
            
            div = fund['Dividend Yield']
            div_fmt = f"{div*100:.2f}%" if isinstance(div, (int, float)) else "0%"
            c4.metric("Div Yield", div_fmt)
            
            # Health Warning
            pe = fund['P/E Ratio']
            if isinstance(pe, (int, float)) and pe > 100:
                st.sidebar.warning("⚠️ Overvalued? (High P/E)")
            elif isinstance(pe, (int, float)) and pe < 0:
                st.sidebar.error("⚠️ Unprofitable (Neg P/E)")

    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ Trading Mode")
    mode = st.sidebar.radio("Strategy", ["Swing (Daily)", "Intraday"])
    interval = "1d"
    if mode == "Intraday":
        interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h"])

    st.sidebar.header("💷 Settings")
    base_curr = st.sidebar.radio("Currency", ["GBP (£)", "USD ($)"])
    capital = st.sidebar.number_input("Capital", 10000)

    # --- MAIN TABS ---
    tabs = st.tabs(["📈 Terminal", "🦅 Scanner", "🔙 Backtest", "🌍 Macro", "💼 Portfolio", "⚙️ Settings"])

    # --- TAB 1: TERMINAL ---
    with tabs[0]:
        if ticker:
            st.title(f"{ticker} Pro Analysis ({interval})")
            
            with st.spinner("Analyzing Market Data & Training Models..."):
                data = logic.get_data(ticker, interval=interval)
                
                if data is not None and not data.empty:
                    # 1. Add Technicals for Charting
                    data = logic.add_technical_overlays(data)
                    
                    # 2. Run Main AI (Short Term)
                    processed, _, votes = logic.train_consensus_model(data)
                    
                    # 3. Run Multi-Timeframe AI (Long Term) - NEW
                    long_term_preds = logic.predict_long_term_trends(data)
                    
                    if processed is not None:
                        last = processed.iloc[-1]
                        conf = last['Confidence']
                        sig = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                        
                        # --- TOP METRICS ---
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Current Price", f"${last['Close']:.2f}")
                        m2.metric("Short-Term Signal", sig)
                        m3.metric("AI Confidence", f"{conf*100:.0f}%")
                        m4.metric("Volatility (ATR)", f"{last['ATR']:.2f}")
                        
                        st.markdown("---")
                        
                        # --- NEW: TIMEFRAME PREDICTION DASHBOARD ---
                        st.subheader("📅 Multi-Timeframe Forecast")
                        t1, t2, t3, t4 = st.columns(4)
                        t1.info(f"**1 Week:** {long_term_preds.get('1 Week', 'N/A')}")
                        t2.info(f"**1 Month:** {long_term_preds.get('1 Month', 'N/A')}")
                        t3.info(f"**3 Months:** {long_term_preds.get('3 Months', 'N/A')}")
                        t4.info(f"**6 Months:** {long_term_preds.get('6 Months', 'N/A')}")
                        
                        st.markdown("---")

                        # --- PRO CHARTING ---
                        st.subheader("📊 Advanced Charting")
                        c1, c2, c3 = st.columns(3)
                        show_bb = c1.checkbox("Bollinger Bands", value=True)
                        show_ma = c2.checkbox("SMA 20/50", value=True)
                        show_macd = c3.checkbox("Show MACD", value=False)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=processed.index, open=processed['Open'], high=processed['High'], low=processed['Low'], close=processed['Close'], name="Price"))
                        
                        if show_bb:
                            fig.add_trace(go.Scatter(x=processed.index, y=processed['BB_High'], line=dict(color='rgba(173, 216, 230, 0.5)'), name='BB High'))
                            fig.add_trace(go.Scatter(x=processed.index, y=processed['BB_Low'], line=dict(color='rgba(173, 216, 230, 0.5)'), fill='tonexty', name='BB Low'))
                        
                        if show_ma:
                            fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
                            fig.add_trace(go.Scatter(x=processed.index, y=processed['SMA_50'], line=dict(color='blue', width=1), name='SMA 50'))

                        # Buy Signals
                        buys = processed[processed['Confidence'] > 0.6]
                        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(color='#00FF00', size=8, symbol='triangle-up'), name='Buy Signal'))

                        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if show_macd:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(x=processed.index, y=processed['MACD'], line=dict(color='cyan'), name='MACD'))
                            fig_macd.add_trace(go.Scatter(x=processed.index, y=processed['MACD_Signal'], line=dict(color='orange'), name='Signal'))
                            fig_macd.update_layout(height=200, template="plotly_dark", margin=dict(t=0))
                            st.plotly_chart(fig_macd, use_container_width=True)

                        # --- ACTIONS ---
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.subheader("🗳️ Consensus Votes")
                            for model, prob in votes.items():
                                st.progress(prob, text=f"{model}: {prob*100:.0f}% Bullish")
                            
                            st.markdown("---")
                            if st.button(f"🛒 Paper Buy {ticker}"):
                                rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                                usd_cap = capital / rate
                                shares = int((usd_cap * 0.02) / (1.5 * last['ATR'])) # 2% Risk Rule
                                res = logic.execute_trade(ticker, last['Close'], shares, "BUY", "USD")
                                st.success(f"Bought {shares} shares! ({res})")

                        with c2:
                            st.subheader("🤖 GPT Analysis")
                            if st.button("Analyze News"):
                                report, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                                st.info(report)
                            st.markdown("---")
                            if st.button("📱 Telegram Alert"):
                                msg = f"🚀 *{ticker} ({interval}) Update*\nSig: {sig}\nPrice: ${last['Close']:.2f}\nConf: {conf*100:.0f}%"
                                res = logic.send_telegram_alert(st.session_state['tele_token'], st.session_state['tele_chat'], msg)
                                st.success(res) if "✅" in res else st.error(res)

                else: st.error("Data Unavailable.")

    # --- TAB 2: SCANNER ---
    with tabs[1]:
        st.header("🦅 Market Hunter")
        if st.button("🚀 Scan Watchlist"):
            with st.spinner("Scanning..."):
                res = logic.scan_market()
                if not res.empty:
                    top = res[res['Signal'].str.contains("BUY")]
                    if not top.empty:
                        st.success(f"Found {len(top)} Buys!")
                        st.dataframe(top.style.format({"Price": "${:.2f}", "Confidence": "{:.1%}"}))
                    else: st.info("No strong buys.")
                    with st.expander("Full Results"): st.dataframe(res)
                else: st.warning("Watchlist empty.")
        
        st.subheader("🐶 Watchlist Manager")
        c1, c2 = st.columns([3, 1])
        new_t = c1.text_input("Add Symbol")
        if c2.button("Add"): logic.add_to_watchlist(new_t.upper()); st.rerun()
        wl = logic.get_watchlist()
        if wl:
            cols = st.columns(5)
            for i, t in enumerate(wl):
                if cols[i%5].button(f"❌ {t}"): logic.remove_from_watchlist(t); st.rerun()

    # --- TAB 3: BACKTEST ---
    with tabs[2]:
        st.header(f"🔙 Time Machine: {ticker}")
        if st.button("Run Simulation"):
            with st.spinner("Simulating..."):
                rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                res, trades, ret = logic.run_backtest(ticker, capital/rate)
                if res is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Return", f"{ret:.2f}%")
                    c2.metric("Final Balance", f"${res['Equity'].iloc[-1]:.2f}")
                    c3.metric("Trades", len(trades))
                    st.line_chart(res['Equity'])
                    st.dataframe(trades)

    # --- TAB 4: MACRO ---
    with tabs[3]:
        st.header("🌍 Global Dashboard")
        macro = logic.get_macro_data()
        cols = st.columns(len(macro))
        for i, (k, v) in enumerate(macro.items()): cols[i].metric(k, f"{v['Price']:.2f}", f"{v['Change']:.2f}%")
        st.markdown("---")
        st.subheader("🔥 Sector Heatmap")
        sectors = logic.get_sector_heatmap()
        cols = st.columns(4)
        for i, (s, c) in enumerate(sectors.items()):
            color = "green" if c > 0 else "red"
            cols[i%4].markdown(f"<div style='background:{'#1e3d1e' if c>0 else '#3d1e1e'};padding:10px;border-radius:5px;text-align:center'><b>{s}</b><br><span style='color:{color}'>{c:+.2f}%</span></div>", unsafe_allow_html=True)

    # --- TAB 5: PORTFOLIO ---
    with tabs[4]:
        st.header(f"💼 Paper Portfolio ({base_curr})")
        
        with st.expander("📓 Trading Journal"):
            note = st.text_area("Why did you trade?", placeholder="e.g. AI Conf 80%...")
            if st.button("Save Note"): st.success("Saved!")
        
        df = logic.get_portfolio()
        if not df.empty and not df[df['Status']=='OPEN'].empty:
            rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
            open_pos = df[df['Status']=='OPEN'].copy()
            total_val = 0
            
            st.subheader("Positions")
            hdr = st.columns([2,1,1,1,1])
            hdr[0].write("**Ticker**"); hdr[1].write("**Shares**"); hdr[2].write("**Cost**"); hdr[3].write("**Value**"); hdr[4].write("**Action**")
            
            for i, row in open_pos.iterrows():
                live = logic.get_data(row['Ticker'], period="1d", interval="1m")
                price = live['Close'].iloc[-1] if live is not None else row['Buy_Price_USD']
                val_base = (price * row['Shares']) * rate
                cost_base = (row['Buy_Price_USD'] * row['Shares']) * rate
                pnl = val_base - cost_base
                total_val += val_base
                
                c = st.columns([2,1,1,1,1])
                c[0].write(f"**{row['Ticker']}**")
                c[1].write(f"{row['Shares']}")
                c[2].write(f"{base_curr[0]}{cost_base/row['Shares']:.2f}")
                c[3].write(f"{base_curr[0]}{val_base:.2f} (: {'green' if pnl>0 else 'red'}[{pnl:+.2f}])")
                if c[4].button("Sell", key=f"s_{i}"):
                    logic.execute_trade(row['Ticker'], price, 0, "SELL")
                    st.rerun()
            
            st.metric("Total Value", f"{base_curr[0]}{total_val:.2f}")
            
            st.markdown("---")
            st.subheader("⚠️ Correlation Risk Matrix")
            corr = logic.get_correlation_matrix(open_pos)
            if corr is not None:
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmin=-1, zmax=1))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Add 2+ stocks to see correlation.")
        else: st.info("No active trades.")

    # --- TAB 6: SETTINGS ---
    with tabs[5]:
        st.header("⚙️ Settings")
        st.session_state['openai_key'] = st.text_input("OpenAI Key", st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Tele Token", st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Tele Chat ID", st.session_state['tele_chat'], type="password")
        if st.button("Logout"): st.session_state['auth'] = False; st.rerun()

main()
