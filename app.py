import streamlit as st
import plotly.graph_objects as go
import logic
import pandas as pd

st.set_page_config(layout="wide", page_title="HedgeFund Terminal", page_icon="🏦")

if 'auth' not in st.session_state: st.session_state['auth'] = False
if 'pwd' not in st.session_state: st.session_state['pwd'] = "Arabella@30"

# Secrets
for k in ['openai_key', 'tele_token', 'tele_chat']:
    if k not in st.session_state: st.session_state[k] = st.secrets.get("api", {}).get(k, "")

if not st.session_state['auth']:
    st.markdown("<h1 style='text-align: center;'>🏦 HedgeFund Terminal Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Login", use_container_width=True) or st.text_input("Password", type="password") == st.session_state['pwd']:
            st.session_state['auth'] = True
            st.rerun()
    st.stop()

def main():
    st.sidebar.header("🔍 Market Search")
    market_region = st.sidebar.selectbox("Select Market", ["All", "USA (NASDAQ/NYSE)", "UK (LSE)", "India (NSE/BSE)"])
    ticker = "AAPL"
    query = st.sidebar.text_input("Ticker / Company")

    if query:
        res = logic.search_ticker(query, region=market_region)
        if res: ticker = res[st.sidebar.selectbox("Results", list(res.keys()))]
    else:
        manual = st.sidebar.text_input("Or Symbol", "AAPL").upper()
        if manual: ticker = manual

    # Fundamentals
    if ticker:
        st.sidebar.markdown("---")
        st.sidebar.header("🏢 Health Check")
        fund = logic.get_fundamentals(ticker)
        if fund:
            st.sidebar.caption(f"{fund['Industry']} | {fund['Sector']}")
            mc = fund['Market Cap']
            mc_str = f"${mc/1e12:.2f}T" if mc > 1e12 else f"${mc/1e9:.2f}B" if mc > 1e9 else f"${mc/1e6:.2f}M"
            st.sidebar.metric("Mkt Cap", mc_str)
            st.sidebar.metric("P/E Ratio", fund['P/E Ratio'])
            st.sidebar.metric("Beta", fund['Beta'])

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Strategy", ["Swing (Daily)", "Intraday"])
    interval = "1d" if mode == "Swing (Daily)" else st.sidebar.selectbox("Interval", ["15m", "30m", "1h"])
    base_curr = st.sidebar.radio("Currency", ["GBP (£)", "USD ($)"])
    capital = st.sidebar.number_input("Capital", 10000)
    symbol = "£" if "GBP" in base_curr else "$"

    tabs = st.tabs(["📈 Terminal", "🦅 Scanner", "🔙 Backtest", "🌍 Macro", "💼 Portfolio", "⚙️ Settings"])

    # --- TAB 1: TERMINAL ---
    with tabs[0]:
        st.title(f"{ticker} Analysis ({interval})")
        if st.button(f"➕ Add {ticker} to Watchlist"):
            logic.add_to_watchlist(ticker)
            st.success("Added!")

        with st.spinner("Analyzing..."):
            data = logic.get_data(ticker, interval=interval)
            if data is not None and not data.empty:
                data = logic.add_technical_overlays(data)
                proc, _, votes = logic.train_consensus_model(data)
                long_preds = logic.predict_long_term_trends(data)

                if proc is not None:
                    last = proc.iloc[-1]
                    sig = "BUY 🟢" if last['Confidence'] > 0.6 else "SELL 🔴" if last['Confidence'] < 0.4 else "WAIT ⚪"
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"${last['Close']:.2f}")
                    m2.metric("Signal", sig)
                    m3.metric("Confidence", f"{last['Confidence']*100:.0f}%")
                    m4.metric("ATR", f"{last['ATR']:.2f}")

                    st.markdown("---")
                    st.subheader("📅 Forecast")
                    cols = st.columns(4)
                    for i, (k, v) in enumerate(long_preds.items()): cols[i].info(f"**{k}:** {v}")

                    fig = go.Figure(data=[go.Candlestick(x=proc.index, open=proc['Open'], high=proc['High'], low=proc['Low'], close=proc['Close'])])
                    fig.add_trace(go.Scatter(x=proc.index, y=proc['BB_High'], line=dict(color='rgba(173,216,230,0.5)'), name='BB High'))
                    fig.add_trace(go.Scatter(x=proc.index, y=proc['BB_Low'], line=dict(color='rgba(173,216,230,0.5)'), fill='tonexty', name='BB Low'))
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.subheader("🗳️ Votes")
                        for k, v in votes.items(): st.progress(v, text=f"{k}: {v*100:.0f}%")
                        if st.button(f"🛒 Paper Buy"):
                            r = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                            shares = int(((capital/r)*0.02)/(1.5*last['ATR']))
                            logic.execute_trade(ticker, last['Close'], shares, "BUY")
                            st.success("Executed!")
                    with c2:
                        st.subheader("🤖 AI Analysis")
                        if st.button("Analyze News"):
                            rep, _ = logic.get_ai_analysis(ticker, st.session_state['openai_key'])
                            st.info(rep)
                        if st.button("Telegram Alert"):
                            logic.send_telegram_alert(st.session_state['tele_token'], st.session_state['tele_chat'], f"Signal: {sig} {ticker}")
                            st.success("Sent!")

    # --- TAB 2: SCANNER ---
    with tabs[1]:
        st.header("🦅 Scanner")
        if st.button("Scan Watchlist"):
            with st.spinner("Scanning..."):
                res = logic.scan_market()
                if not res.empty: st.dataframe(res.style.format({"Price": "${:.2f}", "Confidence": "{:.1%}"}))
                else: st.warning("Watchlist empty or no data.")
        st.subheader("Manager")
        new_t = st.text_input("Add Symbol")
        if st.button("Add"): logic.add_to_watchlist(new_t.upper()); st.rerun()
        wl = logic.get_watchlist()
        if wl:
            cols = st.columns(5)
            for i, t in enumerate(wl):
                if cols[i%5].button(f"❌ {t}"): logic.remove_from_watchlist(t); st.rerun()

    # --- TAB 3: BACKTEST ---
    with tabs[2]:
        st.header("🔙 Backtest")
        if st.button("Run Simulation"):
            r = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
            p, t, ret = logic.run_backtest(ticker, capital/r)
            if p is not None:
                st.metric("Return", f"{ret:.2f}%")
                st.line_chart(p['Equity'])
                st.dataframe(t)

    # --- TAB 4: MACRO ---
    with tabs[3]:
        st.header("🌍 Global Dashboard")
        macro = logic.get_macro_data()
        cols = st.columns(len(macro))
        for i, (k, v) in enumerate(macro.items()): cols[i].metric(k, f"{v['Price']:.2f}", f"{v['Change']:.2f}%")
        st.subheader("🔥 Sectors")
        sects = logic.get_sector_heatmap()
        cols = st.columns(4)
        for i, (s, c) in enumerate(sects.items()):
            cols[i%4].markdown(f"<div style='background:{'#1e3d1e' if c>0 else '#3d1e1e'};padding:10px;text-align:center'>{s}<br>{c:+.2f}%</div>", unsafe_allow_html=True)

    # --- TAB 5: PORTFOLIO ---
    with tabs[4]:
        st.header(f"💼 Portfolio ({symbol})")
        c1, c2 = st.columns([2, 1])
        up = c1.file_uploader("Upload T212 PDF", type="pdf")
        if up and c2.button("Sync"):
            new_a = logic.process_t212_pdf(up)
            if logic.sync_portfolio_with_df(new_a): st.success("Synced!"); st.rerun()
            else: st.error("Failed.")

        df = logic.get_portfolio()
        if not df.empty:
            rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
            # Display logic
            disp_df = df.copy()
            disp_df['Value'] = disp_df['Shares'] * disp_df['Buy_Price_USD'] * rate
            st.dataframe(disp_df)
            
            var = logic.calculate_portfolio_var(df) * rate
            st.warning(f"⚠️ **VaR (95%):** {symbol}{var:.2f}")
            
            corr = logic.get_correlation_matrix(df)
            if corr is not None: 
                st.subheader("Correlation")
                st.plotly_chart(go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r')), use_container_width=True)
            
            if len(df) > 1 and st.button("Optimize Weights"):
                opt = logic.optimize_portfolio(df['Ticker'].unique().tolist())
                if opt: 
                    st.write(opt)
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=list(opt.keys()), values=list(opt.values()))]), use_container_width=True)

    # --- TAB 6: SETTINGS ---
    with tabs[5]:
        st.header("⚙️ Settings")
        st.session_state['openai_key'] = st.text_input("OpenAI Key", st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Tele Token", st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Tele Chat ID", st.session_state['tele_chat'], type="password")
        if st.button("Logout"): st.session_state['auth'] = False; st.rerun()

if __name__ == "__main__":
    main()
