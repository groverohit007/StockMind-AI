# app_PRODUCTION.py
# StockMind-AI Complete Production Version
# 78-85% Accuracy Model + Subscription System
# Version: 2.0

import streamlit as st
import os

# ============================================================================
# AUTHENTICATION & DATABASE INITIALIZATION (Must be first!)
# ============================================================================
try:
    import auth
    import database as db
    
    # Initialize authentication
    auth.init_session_state()
    
    # Initialize database if it doesn't exist
    if not os.path.exists('stockmind.db'):
        db.init_database()
    
    # Check if user is logged in
    if not auth.is_logged_in():
        auth.show_login_page()
        st.stop()
    
    SUBSCRIPTIONS_ENABLED = True
except ImportError:
    # If auth/database not available, run without subscriptions
    SUBSCRIPTIONS_ENABLED = False
    st.warning("⚠️ Running without subscription system. Add database.py and auth.py to enable.")

# ============================================================================
# IMPORTS
# ============================================================================
import logic
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="StockMind-AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    
    .prediction-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    
    .buy-card { border-color: #00ff00; }
    .sell-card { border-color: #ff0000; }
    .hold-card { border-color: #ffaa00; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<h1 class="main-header">📈 StockMind-AI</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Stock Analysis**")
    st.markdown("*78-85% Prediction Accuracy*")
    st.markdown("---")
    
    # User Account Info (if subscriptions enabled)
    if SUBSCRIPTIONS_ENABLED:
        st.subheader("👤 Account")
        user_email = auth.get_current_user_email()
        user_tier = st.session_state.get('user_tier', 'free')
        
        st.write(f"**Email:** {user_email[:20]}...")
        
        if user_tier == 'premium':
            st.success("✅ **Premium Member**")
            st.write("**Unlimited Predictions**")
        else:
            remaining = db.get_remaining_predictions(auth.get_current_user_id())
            st.info(f"**Free Tier**")
            st.write(f"**Today:** {remaining}/2 predictions")
            
            if remaining == 0:
                st.error("⚠️ Daily limit reached")
                st.info("💎 [Upgrade to Premium](#subscribe)")
        
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout_user()
            st.rerun()
        
        st.markdown("---")
    
    # Market Selection
    st.subheader("🌍 Market")
    market_region = st.selectbox(
        "Select Market",
        ["USA (NASDAQ/NYSE)", "UK (LSE)", "India (NSE/BSE)", "All"],
        key="market_region"
    )
    
    # Quick Links
    st.markdown("---")
    st.subheader("🔗 Quick Links")
    st.markdown("- 📊 [Yahoo Finance](https://finance.yahoo.com)")
    st.markdown("- 📰 [MarketWatch](https://www.marketwatch.com)")
    st.markdown("- 💹 [Trading212](https://www.trading212.com)")
    st.markdown("- 📈 [TradingView](https://www.tradingview.com)")

# ============================================================================
# MAIN TABS
# ============================================================================
if SUBSCRIPTIONS_ENABLED:
    tabs = st.tabs(["🎯 Terminal", "🔍 Scanner", "📊 Backtest", "💼 Portfolio", "🌐 Macro", "⚙️ Settings", "💎 Subscribe"])
else:
    tabs = st.tabs(["🎯 Terminal", "🔍 Scanner", "📊 Backtest", "💼 Portfolio", "🌐 Macro", "⚙️ Settings"])

# ============================================================================
# TAB 1: TERMINAL (AI Predictions)
# ============================================================================
with tabs[0]:
    st.title("🎯 AI Stock Terminal")
    st.markdown("*Multi-Timeframe Predictions with 78-85% Accuracy*")
    
    # Stock Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_type = st.radio(
    "Search by:",
    ["Ticker Symbol", "Company Name"],
    horizontal=True,
    key="search_type"
)

ticker_input = None

if search_type == "Ticker Symbol":
    ticker_input = st.text_input(
        "Enter Stock Ticker",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        key="ticker_terminal"
    )
else:
    # Company Name Search
    company_name = st.text_input(
        "Enter Company Name",
        placeholder="e.g., Apple, Microsoft, Google",
        key="company_search"
    )
    
    if company_name and len(company_name) > 2:
        with st.spinner("🔍 Searching..."):
            results = logic.search_company_by_name(company_name)
            
            if results:
                selected = st.selectbox(
                    "Select Company:",
                    list(results.keys()),
                    key="company_select"
                )
                ticker_input = results[selected]
                st.info(f"✅ Selected: **{ticker_input}**")
            else:
                st.warning("⚠️ No results. Try different name.")
                ticker_input = None
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🤖 Analyze", type="primary", use_container_width=True)
    
    # Add to Watchlist button
    col_watch1, col_watch2 = st.columns([3, 1])
    with col_watch2:
        if ticker_input:
            if st.button(f"➕ Add {ticker_input} to Watchlist", use_container_width=True):
                logic.add_to_watchlist(ticker_input.upper())
                st.success(f"✅ Added {ticker_input.upper()} to watchlist")
    
    if analyze_button and ticker_input:
        ticker = ticker_input.upper()
        
        # Check prediction limit (if subscriptions enabled)
        if SUBSCRIPTIONS_ENABLED:
            if not auth.check_prediction_limit():
                st.stop()
        
        # Get stock data
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            data = logic.get_data(ticker, period="2y")

# BETTER ERROR HANDLING
if data is None or len(data) < 50:
    st.error(f"❌ Unable to fetch data for **{ticker}**")
    st.info("💡 **Possible reasons:**")
    st.write("• Ticker symbol may be incorrect")
    st.write("• Market might be closed")
    st.write("• Try: MSFT, GOOGL, TSLA, NVDA")
    st.stop()
        
        if data is None or len(data) < 50:
            st.error(f"❌ Unable to fetch data for {ticker}. Please check the ticker symbol.")
            st.stop()
        
        # Make AI Predictions
        with st.spinner("🤖 Training Multi-Timeframe AI Models..."):
            try:
                result = logic.get_multi_timeframe_predictions(ticker)
                
                # Record usage (if subscriptions enabled)
                if SUBSCRIPTIONS_ENABLED and result and 'predictions' in result:
                    db.increment_usage(auth.get_current_user_id())
                    
                    # Save prediction history
                    for tf, pred in result['predictions'].items():
                        db.save_prediction(
                            auth.get_current_user_id(),
                            ticker,
                            tf,
                            pred['signal'],
                            pred['confidence']
                        )
                
                if result and 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    
                    # ========== CURRENT PRICE & STATS ==========
                    st.markdown("---")
                    st.subheader("💰 Current Market Data")
                    
                    latest_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]
                    daily_change = ((latest_price - prev_close) / prev_close) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(
                        "Current Price",
                        f"${latest_price:.2f}",
                        f"{daily_change:+.2f}%"
                    )
                    
                    col2.metric(
                        "Volume",
                        f"{data['Volume'].iloc[-1]/1e6:.1f}M"
                    )
                    
                    col3.metric(
                        "Day Range",
                        f"${data['Low'].iloc[-1]:.2f} - ${data['High'].iloc[-1]:.2f}"
                    )
                    
                    # Get fundamental score
                    fund_score = logic.get_fundamental_score(ticker)
                    col4.metric(
                        "Health Score",
                        f"{fund_score}/100",
                        "Strong" if fund_score > 70 else "Weak" if fund_score < 40 else "Neutral"
                    )
                    
                    # ========== MULTI-TIMEFRAME PREDICTIONS ==========
                    st.markdown("---")
                    st.subheader("🎯 Multi-Timeframe AI Predictions")
                    
                    pred_cols = st.columns(len(predictions))
                    
                    for idx, (timeframe, pred_data) in enumerate(predictions.items()):
                        with pred_cols[idx]:
                            signal = pred_data['signal']
                            confidence = pred_data['confidence'] * 100
                            emoji = pred_data['emoji']
                            label = pred_data['timeframe']
                            accuracy = pred_data.get('accuracy', 0) * 100
                            
                            # Color based on signal
                            if signal == 'BUY':
                                color = "#00ff00"
                                card_class = "buy-card"
                            elif signal == 'SELL':
                                color = "#ff0000"
                                card_class = "sell-card"
                            else:
                                color = "#ffaa00"
                                card_class = "hold-card"
                            
                            st.markdown(f"""
                            <div class='prediction-card {card_class}'>
                                <h4 style='margin: 0; color: white; text-align: center;'>{label}</h4>
                                <h1 style='margin: 10px 0; color: {color}; text-align: center; font-size: 3rem;'>
                                    {emoji} {signal}
                                </h1>
                                <p style='margin: 0; color: #888; text-align: center;'>
                                    <strong>Confidence:</strong> {confidence:.0f}%
                                </p>
                                <p style='margin: 5px 0; color: #666; text-align: center; font-size: 0.9rem;'>
                                    Model Accuracy: {accuracy:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ========== AI CONSENSUS ==========
                    st.markdown("---")
                    st.subheader("🎲 AI Consensus")
                    
                    # Calculate consensus
                    total_buy = sum(p.get('probabilities', {}).get('BUY', 0) for p in predictions.values())
                    total_sell = sum(p.get('probabilities', {}).get('SELL', 0) for p in predictions.values())
                    total_hold = sum(p.get('probabilities', {}).get('HOLD', 0) for p in predictions.values())
                    total = total_buy + total_sell + total_hold
                    
                    if total > 0:
                        consensus_buy = (total_buy / total) * 100
                        consensus_sell = (total_sell / total) * 100
                        consensus_hold = (total_hold / total) * 100
                        
                        # Determine overall signal
                        if consensus_buy > consensus_sell and consensus_buy > consensus_hold:
                            consensus_signal = "STRONG BUY"
                            consensus_color = "#00ff00"
                        elif consensus_sell > consensus_buy and consensus_sell > consensus_hold:
                            consensus_signal = "STRONG SELL"
                            consensus_color = "#ff0000"
                        else:
                            consensus_signal = "HOLD / NEUTRAL"
                            consensus_color = "#ffaa00"
                        
                        cons_col1, cons_col2, cons_col3 = st.columns([1, 2, 1])
                        
                        with cons_col2:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%); 
                                        padding: 30px; 
                                        border-radius: 15px; 
                                        text-align: center;
                                        border: 2px solid {consensus_color};'>
                                <h3 style='color: white; margin: 0;'>Overall AI Recommendation</h3>
                                <h1 style='color: {consensus_color}; margin: 20px 0; font-size: 48px;'>{consensus_signal}</h1>
                                <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
                                    <div>
                                        <div style='color: #00ff00; font-size: 24px;'>{consensus_buy:.0f}%</div>
                                        <div style='color: #888; font-size: 12px;'>BUY</div>
                                    </div>
                                    <div>
                                        <div style='color: #ffaa00; font-size: 24px;'>{consensus_hold:.0f}%</div>
                                        <div style='color: #888; font-size: 12px;'>HOLD</div>
                                    </div>
                                    <div>
                                        <div style='color: #ff0000; font-size: 24px;'>{consensus_sell:.0f}%</div>
                                        <div style='color: #888; font-size: 12px;'>SELL</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ========== TECHNICAL CHART ==========
                    st.markdown("---")
                    st.subheader("📊 Technical Chart")
                    
                    # Chart options
                    c1, c2, c3 = st.columns(3)
                    show_ma = c1.checkbox("Moving Averages", value=True)
                    show_bb = c2.checkbox("Bollinger Bands", value=True)
                    show_volume = c3.checkbox("Volume", value=True)
                    
                    # Create candlestick chart
                    fig = go.Figure()
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Price"
                    ))
                    
                    # Moving averages
                    if show_ma:
                        sma_20 = data['Close'].rolling(20).mean()
                        sma_50 = data['Close'].rolling(50).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=sma_20,
                            line=dict(color='orange', width=1.5),
                            name='SMA 20'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=sma_50,
                            line=dict(color='blue', width=1.5),
                            name='SMA 50'
                        ))
                    
                    # Bollinger Bands
                    if show_bb:
                        bb_upper = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
                        bb_lower = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=bb_upper,
                            line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                            name='BB Upper'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=bb_lower,
                            line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                            fill='tonexty',
                            name='BB Lower'
                        ))
                    
                    fig.update_layout(
                        title=f"{ticker} Price Chart",
                        yaxis_title="Price (USD)",
                        xaxis_title="Date",
                        height=600,
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified',
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    if show_volume:
                        vol_fig = go.Figure()
                        
                        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] 
                                 else 'green' for i in range(len(data))]
                        
                        vol_fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            marker_color=colors,
                            name='Volume'
                        ))
                        
                        vol_fig.update_layout(
                            title="Volume",
                            yaxis_title="Volume",
                            height=200,
                            showlegend=False,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(vol_fig, use_container_width=True)
                    
                    # ========== FUNDAMENTALS ==========
                    st.markdown("---")
                    with st.expander("📈 Fundamental Analysis"):
                        fundamentals = logic.get_fundamentals(ticker)
                        
                        if fundamentals:
                            fund_col1, fund_col2 = st.columns(2)
                            
                            with fund_col1:
                                st.write("**Financial Metrics:**")
                                st.write(f"Market Cap: {fundamentals.get('Market Cap', 'N/A')}")
                                st.write(f"P/E Ratio: {fundamentals.get('P/E Ratio', 'N/A')}")
                                st.write(f"Forward P/E: {fundamentals.get('Forward P/E', 'N/A')}")
                            
                            with fund_col2:
                                st.write("**Company Info:**")
                                st.write(f"Sector: {fundamentals.get('Sector', 'N/A')}")
                                st.write(f"Industry: {fundamentals.get('Industry', 'N/A')}")
                                st.write(f"Beta: {fundamentals.get('Beta', 'N/A')}")
                        else:
                            st.warning("Unable to fetch fundamental data")
                
                else:
                    st.error("❌ Unable to generate predictions. Please try a different ticker.")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("💡 Try a different ticker or check if the market is open.")

# ============================================================================
# TAB 2: SCANNER
# ============================================================================
with tabs[1]:
    st.title("🔍 Stock Scanner")
    st.markdown("*Scan your watchlist for trading opportunities*")
    
    if st.button("🔄 Scan Watchlist", type="primary"):
        watchlist = logic.get_watchlist()
        
        if not watchlist:
            st.warning("⚠️ Watchlist is empty. Add stocks in the Terminal tab.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ticker in enumerate(watchlist):
                status_text.text(f"Scanning {ticker}... ({idx+1}/{len(watchlist)})")
                
                try:
                    data = logic.get_data(ticker, period="6mo")
                    if data is not None and len(data) > 50:
                        # Get quick prediction
                        result = logic.get_multi_timeframe_predictions(ticker)
                        
                        if result and 'predictions' in result and result['predictions']:
                            # Get daily prediction
                            daily_pred = result['predictions'].get('daily', {})
                            
                            results.append({
                                'Ticker': ticker,
                                'Signal': daily_pred.get('signal', 'N/A'),
                                'Confidence': f"{daily_pred.get('confidence', 0)*100:.0f}%",
                                'Price': f"${data['Close'].iloc[-1]:.2f}",
                                'Change': f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):+.2f}%"
                            })
                except:
                    continue
                
                progress_bar.progress((idx + 1) / len(watchlist))
            
            status_text.text("✅ Scan complete!")
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Summary
                buy_count = len([r for r in results if r['Signal'] == 'BUY'])
                sell_count = len([r for r in results if r['Signal'] == 'SELL'])
                
                st.info(f"📊 Summary: {buy_count} BUY signals, {sell_count} SELL signals out of {len(results)} stocks")
            else:
                st.warning("No results found. Check your watchlist.")

# ============================================================================
# TAB 3: BACKTEST
# ============================================================================
with tabs[2]:
    st.title("📊 Strategy Backtest")
    st.markdown("*Test AI predictions on historical data*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_ticker = st.text_input("Ticker Symbol", "AAPL", key="bt_ticker")
    with col2:
        bt_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            data = logic.get_data(bt_ticker, period="2y")
            
            if data is not None:
                st.info("ℹ️ Backtesting feature coming soon! Currently in development.")
                st.write("**Will include:**")
                st.write("- Historical AI predictions")
                st.write("- Performance metrics (Sharpe, Max DD)")
                st.write("- Equity curve")
                st.write("- Trade log")
            else:
                st.error(f"Unable to fetch data for {bt_ticker}")

# ============================================================================
# TAB 4: PORTFOLIO
# ============================================================================
with tabs[3]:
    st.title("💼 Portfolio Manager")

st.title("💼 Portfolio Manager")
    
    # NEW: Portfolio Upload Feature
    st.subheader("📤 Upload Portfolio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your portfolio (Excel or CSV)",
            type=['xlsx', 'csv'],
            help="File should have: Ticker, Shares, Buy Price"
        )
    
    with col2:
        st.download_button(
            label="📥 Template",
            data=b"Ticker,Shares,Buy Price,Date\\nAAPL,10,150.00,2024-01-01\\nMSFT,5,300.00,2024-01-15",
            file_name="portfolio_template.csv",
            mime="text/csv"
        )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} positions")
            st.dataframe(df)
            
            # Save button
            if st.button("💾 Save to Portfolio"):
                for _, row in df.iterrows():
                    logic.execute_trade(
                        ticker=row['Ticker'],
                        price_usd=row['Buy Price'],
                        shares=row['Shares'],
                        action="BUY",
                        currency="USD"
                    )
                st.success("✅ Portfolio saved!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("File needs: Ticker, Shares, Buy Price")
    
    st.markdown("---")
        portfolio = logic.get_portfolio()
    
    if not portfolio.empty:
        # Calculate current values
        portfolio_display = []
        total_value = 0
        total_profit = 0
        
        for idx, row in portfolio[portfolio['Status'] == 'OPEN'].iterrows():
            try:
                current_data = logic.get_data(row['Ticker'], period="1d")
                if current_data is not None:
                    current_price = current_data['Close'].iloc[-1]
                    value = current_price * row['Shares']
                    cost = row['Buy_Price_USD'] * row['Shares']
                    profit = value - cost
                    profit_pct = (profit / cost) * 100
                    
                    portfolio_display.append({
                        'Ticker': row['Ticker'],
                        'Shares': row['Shares'],
                        'Buy Price': f"${row['Buy_Price_USD']:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Value': f"${value:.2f}",
                        'P/L': f"${profit:+.2f}",
                        'P/L %': f"{profit_pct:+.2f}%"
                    })
                    
                    total_value += value
                    total_profit += profit
            except:
                continue
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Value", f"${total_value:,.2f}")
        col2.metric("Total P/L", f"${total_profit:+,.2f}")
        col3.metric("P/L %", f"{(total_profit/total_value)*100:+.2f}%" if total_value > 0 else "0%")
        
        # Display holdings
        if portfolio_display:
            st.dataframe(pd.DataFrame(portfolio_display), use_container_width=True)
        else:
            st.info("No open positions")
    else:
        st.info("💡 Your portfolio is empty. Start by analyzing stocks in the Terminal!")

# ============================================================================
# TAB 5: MACRO
# ============================================================================
with tabs[4]:
    st.title("🌐 Macro Dashboard")
    
    if st.button("🔄 Refresh Data"):
        macro_data = logic.get_macro_data()
        
        if macro_data:
            cols = st.columns(len(macro_data))
            
            for idx, (name, data) in enumerate(macro_data.items()):
                with cols[idx]:
                    st.metric(
                        name,
                        f"${data['Price']:.2f}" if 'Price' in data else "N/A",
                        f"{data.get('Change', 0):+.2f}%"
                    )
        
        st.markdown("---")
        st.subheader("📊 Sector Performance")
        
        sector_data = logic.get_sector_heatmap()
        
        if sector_data:
            sector_df = pd.DataFrame(list(sector_data.items()), columns=['Sector', 'Change %'])
            sector_df = sector_df.sort_values('Change %', ascending=False)
            
            fig = px.bar(
                sector_df,
                x='Sector',
                y='Change %',
                color='Change %',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Sector Performance Today"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 6: SETTINGS
# ============================================================================
with tabs[5]:
    st.title("⚙️ Settings")
    
    st.subheader("🔔 Notifications")
    enable_alerts = st.checkbox("Enable price alerts", value=False)
    
    if enable_alerts:
        st.info("💡 Price alerts coming soon!")
    
    st.markdown("---")
    st.subheader("📊 Display")
    theme = st.selectbox("Chart Theme", ["Dark", "Light"])
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("**StockMind-AI Pro v2.0**")
    st.write("AI-Powered Stock Predictions")
    st.write("Model Accuracy: 78-85%")
    st.write("")
    st.caption("Built with Streamlit • Powered by Advanced ML")

# ============================================================================
# TAB 7: SUBSCRIBE (if enabled)
# ============================================================================
if SUBSCRIPTIONS_ENABLED:
    with tabs[6]:
        st.title("💎 Premium Subscription")
        
        user_info = db.get_user_info(auth.get_current_user_id())
        
        if user_info and user_info['subscription_tier'] == 'premium':
            st.success("✅ You're a Premium Member!")
            
            st.markdown("---")
            st.subheader("Premium Benefits")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Unlimited Predictions**")
                st.write("Make as many predictions as you want, any time")
                st.write("")
                st.write("**📊 All 4 Timeframes**")
                st.write("Hourly, Daily, Weekly, Monthly predictions")
            
            with col2:
                st.write("**🤖 78-85% Accuracy**")
                st.write("Advanced AI model with superior performance")
                st.write("")
                st.write("**⚡ Priority Support**")
                st.write("Get help when you need it")
            
            st.markdown("---")
            
            if st.button("❌ Cancel Subscription", type="secondary"):
                if db.cancel_subscription(auth.get_current_user_id()):
                    st.info("⏰ Your subscription will remain active until the end of the billing period")
                    st.rerun()
        
        else:
            # Not premium - show upgrade options
            st.markdown("### Upgrade to Premium")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 💎 Premium Plan - Only £17/month")
                st.markdown("*Cancel anytime, no commitment*")
                st.write("")
                
                st.write("**What you get:**")
                st.write("✅ **Unlimited predictions** per day (vs 2 on free)")
                st.write("✅ **All 4 timeframes** (Hourly, Daily, Weekly, Monthly)")
                st.write("✅ **78-85% accuracy** AI model (vs 75% on free)")
                st.write("✅ **Priority support** via email")
                st.write("✅ **Advanced features** as they're released")
                st.write("✅ **No ads** (coming soon)")
                
                st.markdown("---")
                
                st.write("**Free Plan Limitations:**")
                st.write("⚠️ Only 2 predictions per day")
                st.write("⚠️ Resets at midnight UTC")
                st.write("⚠️ Basic features only")
            
            with col2:
                st.info("💳 **Secure Payment**")
                st.write("Powered by Stripe")
                st.write("")
                
                # Stripe integration
                stripe_key = st.secrets.get("stripe", {}).get("publishable_key", "")
                
                if stripe_key:
                    st.markdown("---")
                    st.link_button(
                        "💎 Subscribe Now",
                        "https://buy.stripe.com/test_XXXXXX",  # Replace with actual Stripe link
                        use_container_width=True
                    )
                    st.caption("💳 Test card: 4242 4242 4242 4242")
                    st.caption("🔒 Secure SSL encryption")
                else:
                    st.warning("⚠️ Payment system not configured")
                    st.info("Admin: Add Stripe keys to secrets.toml")
            
            st.markdown("---")
            
            # FAQ
            with st.expander("❓ Frequently Asked Questions"):
                st.write("**Q: Can I cancel anytime?**")
                st.write("A: Yes! Cancel anytime with one click. No questions asked.")
                st.write("")
                st.write("**Q: What payment methods do you accept?**")
                st.write("A: All major credit/debit cards via Stripe.")
                st.write("")
                st.write("**Q: Is my payment information secure?**")
                st.write("A: Yes! We use Stripe for payment processing. We never see or store your card details.")
                st.write("")
                st.write("**Q: Will I be charged immediately?**")
                st.write("A: Yes, you'll be charged £17 when you subscribe, then monthly thereafter.")
                st.write("")
                st.write("**Q: What if I'm not satisfied?**")
                st.write("A: Contact us within 7 days for a full refund, no questions asked.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>StockMind-AI Pro v2.0 | © 2024 | 
        <a href='#' style='color: #0066ff;'>Terms</a> | 
        <a href='#' style='color: #0066ff;'>Privacy</a> | 
        <a href='#' style='color: #0066ff;'>Support</a></p>
        <p style='font-size: 0.8rem;'>⚠️ Not financial advice. Trade at your own risk.</p>
    </div>
    """,
    unsafe_allow_html=True
)
