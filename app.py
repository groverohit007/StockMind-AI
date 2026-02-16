import streamlit as st
import plotly.graph_objects as go
import logic
import io
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
    # --- SIDEBAR: SEARCH & MARKETS ---
    st.sidebar.header("🔍 Market Search")
    
    # MARKET REGION SELECTOR
    market_region = st.sidebar.selectbox("Select Market", 
        ["All", "USA (NASDAQ/NYSE)", "UK (LSE)", "India (NSE/BSE)"])
    
    ticker = "AAPL"
    query = st.sidebar.text_input("Ticker / Company")
    
    if query:
        res = logic.search_ticker(query, region=market_region)
        if res: ticker = res[st.sidebar.selectbox("Results", list(res.keys()))]
    else:
        manual = st.sidebar.text_input("Or Symbol", "AAPL").upper()
        if manual: ticker = manual

    # --- SIDEBAR: FUNDAMENTAL HEALTH ---
    if ticker:
        st.sidebar.markdown("---")
        st.sidebar.header("🏢 Health Check")
        fund = logic.get_fundamentals(ticker)
        if fund:
            st.sidebar.caption(f"{fund['Industry']} | {fund['Sector']}")
            
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
            
            # TOP ACTION BAR
            c_top1, c_top2 = st.columns([3, 1])
            with c_top2:
                if st.button(f"➕ Add {ticker} to Watchlist"):
                    logic.add_to_watchlist(ticker)
                    st.success(f"Added {ticker} to Watchdog!")

            with st.spinner("Analyzing Market Data & Training Models..."):
                data = logic.get_data(ticker, interval=interval)
                
                if data is not None and not data.empty:
                    # 1. Technicals
                    data = logic.add_technical_overlays(data)
                    # 2. AI Short Term (Robust)
                    processed, _, votes = logic.train_consensus_model(data)
                    # 3. AI Long Term
                    long_term_preds = logic.predict_long_term_trends(data)
                    
                    if processed is not None:
                        last = processed.iloc[-1]
                        conf = last['Confidence']
                        sig = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Current Price", f"${last['Close']:.2f}")
                        m2.metric("Short-Term Signal", sig)
                        m3.metric("AI Confidence", f"{conf*100:.0f}%")
                        m4.metric("Volatility (ATR)", f"{last['ATR']:.2f}")
                        
                        st.markdown("---")
                        
                        st.subheader("📅 Multi-Timeframe Forecast")
                        t1, t2, t3, t4 = st.columns(4)
                        t1.info(f"**1 Week:** {long_term_preds.get('1 Week', 'N/A')}")
                        t2.info(f"**1 Month:** {long_term_preds.get('1 Month', 'N/A')}")
                        t3.info(f"**3 Months:** {long_term_preds.get('3 Months', 'N/A')}")
                        t4.info(f"**6 Months:** {long_term_preds.get('6 Months', 'N/A')}")
                        
                        st.markdown("---")

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

                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.subheader("🗳️ Consensus Votes")
                            for model, prob in votes.items():
                                st.progress(prob, text=f"{model}: {prob*100:.0f}% Bullish")
                            
                            st.markdown("---")
                            if st.button(f"🛒 Paper Buy {ticker}"):
                                rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
                                usd_cap = capital / rate
                                shares = int((usd_cap * 0.02) / (1.5 * last['ATR'])) 
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

# --- TAB 5: PORTFOLIO (Enhanced with Excel Import) ---
with tabs[4]:
    st.header(f"💼 Portfolio Manager ({base_curr})")
    
    # --- IMPORT/EXPORT SECTION ---
    st.subheader("📤 Import / Export Portfolio")
    
    import_col1, import_col2, import_col3 = st.columns([2, 2, 1])
    
    # Template Download
    with import_col1:
        st.caption("**Download Template**")
        template_df = logic.generate_portfolio_template()
        template_excel = io.BytesIO()
        template_df.to_excel(template_excel, index=False, engine='openpyxl')
        template_excel.seek(0)
        
        st.download_button(
            label="📥 Download Excel Template",
            data=template_excel,
            file_name="portfolio_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download a sample Excel file with the correct format"
        )
    
    # Export Current Portfolio
    with import_col2:
        st.caption("**Export Current Portfolio**")
        if st.button("📤 Export to Excel"):
            export_path = "/home/claude/portfolio_export.xlsx"
            if logic.export_portfolio_to_excel(export_path):
                with open(export_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Export",
                        data=f,
                        file_name=f"portfolio_export_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                st.success("Portfolio exported successfully!")
            else:
                st.error("No portfolio data to export")
    
    st.markdown("---")
    
    # Excel Upload Section
    st.subheader("📊 Upload Excel Portfolio")
    
    upload_col1, upload_col2 = st.columns([3, 1])
    
    with upload_col1:
        excel_file = st.file_uploader(
            "Upload your portfolio Excel file",
            type=['xlsx', 'xls'],
            help="Accepts Excel files with columns: Ticker/ISIN, Shares/Units, Price, Currency (optional), Date (optional)"
        )
    
    with upload_col2:
        merge_mode = st.selectbox(
            "Import Mode",
            ["Add", "Update", "Replace"],
            help="""
            - Add: Keep existing positions, add new ones
            - Update: Merge positions by ticker (average prices)
            - Replace: Clear existing and import fresh
            """
        )
    
    # Process Excel Upload
    if excel_file is not None:
        try:
            with st.spinner("Reading Excel file..."):
                excel_df = logic.read_excel_portfolio(excel_file)
            
            if excel_df is not None:
                st.success(f"✅ Loaded {len(excel_df)} positions from Excel")
                
                # Show preview
                with st.expander("📋 Preview Imported Data", expanded=True):
                    st.dataframe(excel_df[['Ticker', 'Shares', 'Buy_Price_USD', 'Currency', 'Date']])
                
                # Validate data
                with st.spinner("Validating data..."):
                    validation = logic.validate_portfolio_data(excel_df)
                
                # Show validation results
                if validation['issues']:
                    st.error("**Issues Found:**")
                    for issue in validation['issues']:
                        st.write(f"- {issue}")
                
                if validation['warnings']:
                    st.warning("**Warnings:**")
                    for warning in validation['warnings']:
                        st.write(f"- {warning}")
                
                # Show summary
                st.info(f"""
                **Import Summary:**
                - Total positions: {validation['total_positions']}
                - Unique tickers: {validation['unique_tickers']}
                - Total cost basis: ${validation['total_value']:,.2f}
                """)
                
                # Import button
                if st.button("🚀 Confirm Import", type="primary"):
                    with st.spinner(f"Importing portfolio ({merge_mode.lower()} mode)..."):
                        success = logic.sync_excel_portfolio(
                            excel_df, 
                            merge_mode=merge_mode.lower()
                        )
                        
                        if success:
                            st.success("Portfolio imported successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Import failed. Please check the logs.")
        
        except Exception as e:
            st.error(f"**Error reading Excel file:** {str(e)}")
            st.caption("Make sure your Excel file has the correct format. Download the template above for reference.")
    
    st.markdown("---")
    
    # --- TRADING 212 PDF IMPORT (Existing) ---
    st.subheader("📤 Import Trading 212 PDF Statement")
    
    pdf_col1, pdf_col2 = st.columns([2, 1])
    
    uploaded_pdf = pdf_col1.file_uploader("Upload Trading 212 PDF Statement", type="pdf")
    
    if uploaded_pdf is not None:
        if pdf_col2.button("Sync PDF"):
            with st.spinner("Reading PDF and syncing assets..."):
                new_assets = logic.process_t212_pdf(uploaded_pdf)
                
                if new_assets is not None and not new_assets.empty:
                    success = logic.sync_portfolio_with_df(new_assets)
                    
                    if success:
                        st.success(f"Successfully synced {len(new_assets)} assets!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("No assets imported. Please check the PDF format.")
    
    st.markdown("---")
    
    # --- TRADING JOURNAL ---
    with st.expander("📓 Trading Journal"):
        note = st.text_area("Why did you trade?", placeholder="e.g. AI Conf 80%...")
        if st.button("Save Note"): 
            st.success("Saved!")
    
    st.markdown("---")
    
    # --- PORTFOLIO DISPLAY (Existing, enhanced) ---
    df = logic.get_portfolio()
    
    if not df.empty and not df[df['Status']=='OPEN'].empty:
        rate = logic.get_exchange_rate("GBP") if "GBP" in base_curr else 1.0
        open_pos = df[df['Status']=='OPEN'].copy()
        
        # Calculate current values
        open_pos['Current_Price'] = 0.0
        open_pos['Current_Value'] = 0.0
        open_pos['PnL'] = 0.0
        open_pos['PnL_Pct'] = 0.0
        
        total_cost = 0
        total_value = 0
        
        for idx, row in open_pos.iterrows():
            live = logic.get_data(row['Ticker'], period="1d", interval="1m")
            price = live['Close'].iloc[-1] if live is not None and not live.empty else row['Buy_Price_USD']
            
            cost_base = row['Buy_Price_USD'] * row['Shares']
            current_val = price * row['Shares']
            pnl = current_val - cost_base
            pnl_pct = ((price / row['Buy_Price_USD']) - 1) * 100
            
            open_pos.at[idx, 'Current_Price'] = price
            open_pos.at[idx, 'Current_Value'] = current_val
            open_pos.at[idx, 'PnL'] = pnl
            open_pos.at[idx, 'PnL_Pct'] = pnl_pct
            
            total_cost += cost_base
            total_value += current_val
        
        # Portfolio Summary Cards
        st.subheader("💰 Portfolio Summary")
        
        total_pnl = total_value - total_cost
        total_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        sum_col1.metric(
            "Total Value",
            f"{base_curr[0]}{total_value * rate:,.2f}",
            help="Current market value of all positions"
        )
        
        sum_col2.metric(
            "Total Cost",
            f"{base_curr[0]}{total_cost * rate:,.2f}",
            help="Total amount invested"
        )
        
        sum_col3.metric(
            "Unrealized P&L",
            f"{base_curr[0]}{total_pnl * rate:,.2f}",
            delta=f"{total_pnl_pct:+.2f}%",
            delta_color="normal"
        )
        
        sum_col4.metric(
            "Positions",
            f"{len(open_pos)} / {open_pos['Ticker'].nunique()} unique",
            help="Total positions / Unique tickers"
        )
        
        st.markdown("---")
        
        # Position List with Enhanced Display
        st.subheader("📊 Positions")
        
        # Add sorting options
        sort_col1, sort_col2 = st.columns([3, 1])
        sort_by = sort_col1.selectbox(
            "Sort by",
            ["P&L %", "P&L $", "Value", "Ticker"],
            key="sort_positions"
        )
        
        sort_order = sort_col2.radio("Order", ["Desc", "Asc"], horizontal=True, key="sort_order")
        
        # Sort dataframe
        sort_map = {
            "P&L %": "PnL_Pct",
            "P&L $": "PnL",
            "Value": "Current_Value",
            "Ticker": "Ticker"
        }
        
        open_pos = open_pos.sort_values(
            by=sort_map[sort_by],
            ascending=(sort_order == "Asc")
        )
        
        # Display positions
        for idx, row in open_pos.iterrows():
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                
                # Ticker with live price
                col1.markdown(f"**{row['Ticker']}**")
                col1.caption(f"{row['Shares']} shares @ {base_curr[0]}{row['Current_Price']*rate:.2f}")
                
                # Purchase info
                col2.metric("Cost Basis", f"{base_curr[0]}{row['Buy_Price_USD']*rate:.2f}")
                
                # Current value
                col3.metric("Value", f"{base_curr[0]}{row['Current_Value']*rate:.2f}")
                
                # P&L
                pnl_color = "green" if row['PnL'] > 0 else "red" if row['PnL'] < 0 else "gray"
                col4.markdown(f"<div style='color:{pnl_color}'><b>{base_curr[0]}{row['PnL']*rate:+,.2f}</b><br><small>{row['PnL_Pct']:+.1f}%</small></div>", unsafe_allow_html=True)
                
                # Purchase date
                col5.caption(f"Bought: {row['Date'].strftime('%Y-%m-%d')}")
                
                # Actions
                if col6.button("🗑️ Sell", key=f"sell_{idx}"):
                    logic.execute_trade(row['Ticker'], row['Current_Price'], 0, "SELL")
                    st.rerun()
                
                st.markdown("---")
        
        # --- RISK & OPTIMIZATION (Existing) ---
        st.markdown("---")
        st.subheader("📊 Risk Analytics")
        
        # Value at Risk
        var_val = logic.calculate_portfolio_var(open_pos) * rate
        st.warning(f"⚠️ **Value at Risk (Daily 95%):** {base_curr[0]}{var_val:.2f}")
        st.caption("You have a 95% confidence that you will not lose more than this amount in a single day.")
        
        # Correlation Matrix
        st.markdown("---")
        st.subheader("🔗 Asset Correlation")
        corr = logic.get_correlation_matrix(open_pos)
        if corr is not None:
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=corr.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            fig.update_layout(height=400, title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add 2+ stocks to see correlation.")
        
        # Portfolio Optimizer
        st.markdown("---")
        st.subheader("⚖️ AI Portfolio Optimizer")
        st.caption("Uses Modern Portfolio Theory (Markowitz) to find the optimal allocation.")
        
        if len(open_pos['Ticker'].unique()) >= 2:
            if st.button("🚀 Run Optimizer"):
                with st.spinner("Calculating Efficient Frontier..."):
                    optimal_weights = logic.optimize_portfolio(open_pos['Ticker'].unique().tolist())
                    
                    if optimal_weights:
                        st.success("✅ Optimization Complete!")
                        
                        opt_df = pd.DataFrame(
                            list(optimal_weights.items()),
                            columns=["Ticker", "Ideal Weight"]
                        )
                        opt_df['Ideal Weight'] = opt_df['Ideal Weight'].apply(lambda x: f"{x*100:.1f}%")
                        
                        c1, c2 = st.columns(2)
                        c1.dataframe(opt_df, use_container_width=True)
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=list(optimal_weights.keys()),
                            values=list(optimal_weights.values()),
                            hole=.3
                        )])
                        fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                        c2.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("You need at least 2 different stocks to run the optimizer.")
    
    else:
        st.info("💡 No active positions. Import your portfolio using Excel or PDF above!")

    # --- TAB 6: SETTINGS ---
    with tabs[5]:
        st.header("⚙️ Settings")
        st.session_state['openai_key'] = st.text_input("OpenAI Key", st.session_state['openai_key'], type="password")
        st.session_state['tele_token'] = st.text_input("Tele Token", st.session_state['tele_token'], type="password")
        st.session_state['tele_chat'] = st.text_input("Tele Chat ID", st.session_state['tele_chat'], type="password")
        st.markdown("---")
        st.subheader("🐶 Portfolio Watchdog")
        st.caption("Scan your synced portfolio for AI signals and get Telegram alerts.")
        
        c1, c2 = st.columns([1, 1])
        threshold = c1.slider("AI Confidence Threshold", 0.5, 0.95, 0.75, help="Only alert if AI is this sure.")
        
        if c2.button("🚀 Run Watchdog Scan Now"):
            if not st.session_state['tele_token'] or not st.session_state['tele_chat']:
                st.error("⚠️ Please save Telegram Token & Chat ID first!")
            else:
                with st.spinner("🐶 Sniffing the market..."):
                    result = logic.run_watchdog_scan(
                        st.session_state['tele_token'], 
                        st.session_state['tele_chat'], 
                        threshold
                    )
                    if "✅" in result: st.success(result)
                    else: st.info(result)
        if st.button("Logout"): st.session_state['auth'] = False; st.rerun()

main()



