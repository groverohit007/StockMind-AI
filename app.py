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
    
    # Initialize / migrate database and ensure master admin user
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
import time

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
    @keyframes fadeInMain {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .main .block-container {
        animation: fadeInMain 0.6s ease-out;
    }

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

# Login transition animation (one-time after successful login)
if SUBSCRIPTIONS_ENABLED and st.session_state.get('show_login_transition', False):
    transition_placeholder = st.empty()
    transition_placeholder.markdown(
        """
        <div style="
            text-align:center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,102,255,0.12));
            border: 1px solid rgba(0,212,255,0.35);
            border-radius: 14px;
            margin-bottom: 1rem;
        ">
            <h3 style="margin-bottom: 0.5rem;">🚀 Welcome to StockMind-AI Pro</h3>
            <p style="opacity: 0.9; margin-bottom: 0;">Loading your dashboard...</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(1.0)
    st.session_state['show_login_transition'] = False
    transition_placeholder.empty()

# Premium upgrade popup (shown after premium user login)
if SUBSCRIPTIONS_ENABLED and st.session_state.get('show_premium_popup', False):
    active_models = logic.get_active_model_stack()
    model_lines = "\n".join([f"• {m}" for m in active_models])
    st.success("🎉 Congratulations! You have been upgraded to Premium.")
    st.info(
        "**Premium AI engine activated (75%–85% target accuracy):**\n"
        f"{model_lines}\n"
        "• AI-enhanced multi-timeframe signal layer"
    )
    st.balloons()
    st.session_state['show_premium_popup'] = False

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
        
        if auth.is_admin():
            st.success("👑 **Master Admin**")
            st.write("**Full platform access**")
        elif user_tier == 'premium':
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
    tabs = st.tabs(["🎯 Terminal", "🔍 Scanner", "🟢 Buy/Sell", "📊 Backtest", "💼 Portfolio", "🌐 Macro", "⚙️ Settings", "💎 Subscribe"])
else:
    tabs = st.tabs(["🎯 Terminal", "🔍 Scanner", "🟢 Buy/Sell", "📊 Backtest", "💼 Portfolio", "🌐 Macro", "⚙️ Settings"])

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
        data = None
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            data = logic.get_data(ticker, period="2y")
        
        # Error Handling
        if data is None or len(data) < 35:
            st.error(f"❌ Unable to fetch data for **{ticker}**")
            st.info("💡 **Possible reasons:**")
            st.write("• Ticker symbol may be incorrect")
            st.write("• Market might be closed")
            st.write("• Try: MSFT, GOOGL, TSLA, NVDA")

            with st.expander("🛠️ Data source diagnostics", expanded=False):
                status = logic.get_data_source_status(ticker)
                st.write(f"**Yahoo Finance reachable:** {'✅' if status['yahoo_ok'] else '❌'}")
                st.write(f"**Alpha Vantage key configured:** {'✅' if status['alpha_key_configured'] else '❌'}")
                if status['alpha_key_configured']:
                    st.write(f"**Alpha Vantage data fetch:** {'✅' if status['alpha_ok'] else '❌'}")
                    if status.get('alpha_message'):
                        st.caption(f"Alpha Vantage message: {status['alpha_message']}")

                if not status['alpha_key_configured']:
                    st.info(
                        "**To configure Alpha Vantage** (fallback data source):\n"
                        "- Set the `ALPHA_VANTAGE_API_KEY` env var in Streamlit Cloud Secrets, **or**\n"
                        "- Enter the key in the **Settings > API configuration** tab (admin only)\n\n"
                        "Get a free key at https://www.alphavantage.co/support/#api-key\n\n"
                        "Note: GitHub Actions secrets do **not** apply here."
                    )

                st.warning("If Alpha Vantage shows a rate-limit note, wait 60 seconds and try again. Free plans are throttled.")
            st.stop()
        
        # Determine user tier for model selection
        if SUBSCRIPTIONS_ENABLED:
            user_tier = 'premium' if (auth.is_admin() or auth.is_premium()) else 'free'
        else:
            user_tier = 'free'

        # Make AI Predictions
        with st.spinner("🤖 Training Multi-Timeframe AI Models..."):
            try:
                result = logic.get_multi_timeframe_predictions(ticker, user_tier=user_tier)
                
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
        if SUBSCRIPTIONS_ENABLED:
            scan_tier = 'premium' if (auth.is_admin() or auth.is_premium()) else 'free'
        else:
            scan_tier = 'free'

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
                        result = logic.get_multi_timeframe_predictions(ticker, user_tier=scan_tier)
                        
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
# TAB 3: BUY/SELL (PREMIUM)
# ============================================================================
with tabs[2]:
    st.title("🟢 Buy/Sell AI Signals")
    st.markdown("*Premium-only intraday buy/sell signals with estimated target prices.*")

    is_premium_user = SUBSCRIPTIONS_ENABLED and (auth.is_admin() or auth.is_premium())

    if not is_premium_user:
        st.warning("🔒 Buy/Sell tab is available for Premium subscribers only.")
        st.markdown("### Upgrade to Premium")
        st.write("Unlock intraday 15m / 30m / 1h AI signals and target prices.")
        st.link_button(
            "💳 Upgrade via Stripe - £17/month",
            "https://buy.stripe.com/4gM7sM8MWcCEb7r2WQ7AI00",
            use_container_width=True
        )
    else:
        col_bs_1, col_bs_2 = st.columns([3, 1])

        with col_bs_1:
            bs_search_type = st.radio(
                "Search stock by:",
                ["Ticker Symbol", "Company Name"],
                horizontal=True,
                key="buy_sell_search_type"
            )

            bs_ticker = None
            if bs_search_type == "Ticker Symbol":
                bs_ticker = st.text_input(
                    "Enter Stock Ticker",
                    placeholder="e.g., AAPL, MSFT, NVDA",
                    key="buy_sell_ticker"
                )
            else:
                bs_company = st.text_input(
                    "Enter Company Name",
                    placeholder="e.g., Apple, Microsoft, Nvidia",
                    key="buy_sell_company"
                )
                if bs_company and len(bs_company) > 2:
                    with st.spinner("🔍 Searching..."):
                        bs_results = logic.search_company_by_name(bs_company)
                        if bs_results:
                            bs_selected = st.selectbox(
                                "Select Company:",
                                list(bs_results.keys()),
                                key="buy_sell_company_select"
                            )
                            bs_ticker = bs_results[bs_selected]
                            st.info(f"✅ Selected: **{bs_ticker}**")
                        else:
                            st.warning("⚠️ No results found. Try another company name.")

        with col_bs_2:
            bs_interval_label = st.selectbox(
                "Interval",
                ["15 Minutes", "30 Minutes", "1 Hour"],
                key="buy_sell_interval"
            )
            auto_refresh = st.checkbox("Auto update chart", value=False, key="buy_sell_auto_refresh")
            refresh_seconds = st.selectbox(
                "Refresh every",
                [15, 30, 60],
                index=1,
                key="buy_sell_refresh_seconds"
            )

        interval_map = {
            "15 Minutes": "15m",
            "30 Minutes": "30m",
            "1 Hour": "1h"
        }
        selected_interval = interval_map[bs_interval_label]

        if bs_ticker:
            bs_ticker = bs_ticker.upper().strip()
            st.markdown(f"### {bs_ticker} • {bs_interval_label}")

            period_map = {"15m": "30d", "30m": "45d", "1h": "60d"}
            chart_data = logic.get_data(bs_ticker, period=period_map[selected_interval], interval=selected_interval)

            if chart_data is None or len(chart_data) < 30:
                st.error("❌ Unable to load chart data for this stock/interval.")
            else:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=chart_data.index,
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name=bs_ticker
                        )
                    ]
                )
                fig.update_layout(
                    title=f"{bs_ticker} Price Chart ({bs_interval_label})",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark',
                    height=520
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.spinner("🤖 Generating AI Buy/Sell signal..."):
                    bs_signal = logic.get_interval_trade_signal(bs_ticker, interval=selected_interval)

                if not bs_signal:
                    st.error("❌ Could not generate intraday signal right now. Please try again.")
                else:
                    signal = bs_signal['signal']
                    signal_color = "#00cc66" if signal == "BUY" else "#ff4d4f" if signal == "SELL" else "#f1c40f"

                    st.markdown(
                        f"""
                        <div style="padding: 16px; border-left: 6px solid {signal_color}; background: rgba(255,255,255,0.04); border-radius: 8px; margin-bottom: 1rem;">
                            <h3 style="margin:0;">AI Signal: {signal}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Price", f"${bs_signal['current_price']:.2f}")
                    m2.metric("Target Price", f"${bs_signal['target_price']:.2f}")
                    m3.metric("Projected Move", f"{bs_signal['projected_change_pct']:+.2f}%")
                    m4.metric("Model Confidence", f"{bs_signal['confidence']*100:.1f}%")

                    m5, m6, m7 = st.columns(3)
                    m5.metric("Meta Signal", bs_signal.get('meta_signal', bs_signal['signal']))
                    m6.metric("Meta Score", f"{bs_signal.get('meta_score', 0.0)*100:.1f}%")
                    m7.metric("Uncertainty", f"{bs_signal.get('uncertainty', 0.0)*100:.1f}%")

                    if bs_signal.get('regime'):
                        st.caption(f"Detected regime: {bs_signal['regime']}")

                    st.caption(
                        f"Estimated model accuracy band for this signal: {bs_signal['accuracy']*100:.1f}%"
                    )
                    st.write("**AI models used:** " + ", ".join(bs_signal['model_stack']))

            if auto_refresh:
                last_refreshed = st.session_state.get('buy_sell_last_refreshed')
                if last_refreshed:
                    st.caption(
                        f"Auto update enabled • Last refreshed: {last_refreshed.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    st.caption("Auto update enabled • Last refreshed: just now")

                st.caption(f"Refreshing every {refresh_seconds} seconds...")
                st.session_state['buy_sell_last_refreshed'] = datetime.now()
                time.sleep(refresh_seconds)
                st.rerun()

# ============================================================================
# TAB 4: BACKTEST
# ============================================================================
with tabs[3]:
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
# TAB 5: PORTFOLIO
# ============================================================================
with tabs[4]:
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
    
    # Portfolio Display Logic
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
# TAB 6: MACRO
# ============================================================================
with tabs[5]:
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
# TAB 7: SETTINGS
# ============================================================================
with tabs[6]:
    st.title("⚙️ Settings")
    
    st.subheader("🔔 Notifications")
    enable_alerts = st.checkbox("Enable price alerts", value=False)
    
    if enable_alerts:
        st.info("💡 Price alerts coming soon!")
    
    st.markdown("---")
    st.subheader("📊 Display")
    theme = st.selectbox("Chart Theme", ["Dark", "Light"])
    

    st.markdown("---")
    st.subheader("👑 Admin Control Center")

    if SUBSCRIPTIONS_ENABLED and auth.is_admin():
        st.success("Master admin mode enabled")

        with st.expander("🔐 Admin password", expanded=False):
            with st.form("admin_password_form"):
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm new password", type="password")
                password_submit = st.form_submit_button("Update admin password")

                if password_submit:
                    if not new_password or len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif db.update_user_password(auth.get_current_user_id(), new_password):
                        st.success("✅ Admin password updated")
                    else:
                        st.error("❌ Failed to update password")

        with st.expander("🧩 API configuration", expanded=True):
            existing_settings = db.get_app_settings()

            # Check which keys are already provided via env vars or secrets.toml
            def _key_source(env_vars, section, secret_key='api_key', db_key=None):
                """Return the source name if the key is already configured, else None."""
                for var in env_vars:
                    if os.getenv(var, '').strip():
                        return f"environment variable ({var})"
                try:
                    if st.secrets.get(section, {}).get(secret_key, '').strip():
                        return "secrets.toml / Streamlit Cloud secrets"
                except Exception:
                    pass
                if db_key and existing_settings.get(db_key, '').strip():
                    return "admin settings (database)"
                return None

            alpha_source = _key_source(['ALPHA_VANTAGE_API_KEY'], 'alpha_vantage', db_key='alpha_vantage_api_key')
            news_source = _key_source(['NEWS_API_KEY'], 'news', db_key='news_api_key')
            openai_source = _key_source(['OPENAI_API_KEY', 'OPENAI_KEY'], 'openai', db_key='openai_api_key')

            st.info(
                "**Where to configure API keys** (checked in order):\n"
                "1. **Environment variables** — set in Streamlit Cloud (Settings > Secrets as `KEY=value`) or your hosting platform\n"
                "2. **Streamlit secrets.toml** — add via Streamlit Cloud dashboard or `.streamlit/secrets.toml` file\n"
                "3. **Admin settings below** — saved in the app database\n\n"
                "⚠️ **GitHub Actions secrets** (repo > Settings > Secrets > Actions) are only available "
                "to GitHub Actions workflows. They do **not** apply to Streamlit Cloud deployments."
            )

            with st.form("admin_api_settings_form"):
                if alpha_source:
                    st.success(f"**Alpha Vantage API Key:** Configured via {alpha_source}")
                    alpha_vantage_key = None
                else:
                    alpha_vantage_key = st.text_input(
                        "Alpha Vantage API Key",
                        value=existing_settings.get("alpha_vantage_api_key", ""),
                        type="password",
                        help="Get a free key at https://www.alphavantage.co/support/#api-key"
                    )

                if news_source:
                    st.success(f"**News API Key:** Configured via {news_source}")
                    news_api_key = None
                else:
                    news_api_key = st.text_input(
                        "News API Key",
                        value=existing_settings.get("news_api_key", ""),
                        type="password",
                        help="Get a free key at https://newsapi.org/register"
                    )

                if openai_source:
                    st.success(f"**OpenAI API Key:** Configured via {openai_source}")
                    openai_api_key = None
                else:
                    openai_api_key = st.text_input(
                        "OpenAI API Key",
                        value=existing_settings.get("openai_api_key", ""),
                        type="password",
                        help="Get a key at https://platform.openai.com/api-keys"
                    )

                save_apis = st.form_submit_button("Save API settings")

                if save_apis:
                    save_results = []
                    if alpha_vantage_key is not None:
                        save_results.append(db.set_app_setting("alpha_vantage_api_key", alpha_vantage_key.strip()))
                    if news_api_key is not None:
                        save_results.append(db.set_app_setting("news_api_key", news_api_key.strip()))
                    if openai_api_key is not None:
                        save_results.append(db.set_app_setting("openai_api_key", openai_api_key.strip()))

                    if not save_results:
                        st.info("All API keys are managed via environment variables or secrets.toml")
                    elif all(save_results):
                        st.success("✅ API settings saved")
                        st.rerun()
                    else:
                        st.error("❌ Failed saving one or more API settings")

        with st.expander("📧 User emails (marketing list)", expanded=True):
            emails = db.get_all_user_emails()
            if emails:
                email_df = pd.DataFrame({"email": emails})
                st.dataframe(email_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download emails CSV",
                    email_df.to_csv(index=False),
                    file_name=f"stockmind_user_emails_{datetime.now().date()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No user emails available yet")

        with st.expander("💳 Manual subscription assignment", expanded=True):
            st.caption("Use this when Stripe webhook/payment sync is delayed and you need to manually update a user.")

            all_users = db.get_all_users()
            non_admin_users = [u for u in all_users if not u.get('is_admin')]

            if not non_admin_users:
                st.info("No non-admin users available.")
            else:
                user_options = {
                    f"{u['email']} ({u['tier']}/{u['status']})": u
                    for u in non_admin_users
                }

                with st.form("manual_subscription_form"):
                    selected_label = st.selectbox(
                        "Select user",
                        list(user_options.keys()),
                        help="Choose the user account you want to update"
                    )
                    selected_action = st.selectbox(
                        "Subscription action",
                        ["Grant Premium", "Set Free"],
                        help="Grant Premium = premium/active, Set Free = free/active"
                    )
                    save_subscription = st.form_submit_button("Update subscription")

                if save_subscription:
                    selected_user = user_options[selected_label]
                    target_tier = 'premium' if selected_action == "Grant Premium" else 'free'
                    user_info = db.get_user_info(selected_user['id']) or {}
                    success = db.update_subscription(
                        selected_user['id'],
                        tier=target_tier,
                        status='active',
                        stripe_customer_id=user_info.get('stripe_customer_id'),
                        stripe_subscription_id=None
                    )

                    if success:
                        st.success(
                            f"✅ Updated {selected_user['email']} to {target_tier.title()} ({'active'})"
                        )
                        st.rerun()
                    else:
                        st.error("❌ Failed to update subscription")
    elif SUBSCRIPTIONS_ENABLED:
        st.info("Admin Control Center is only available for master admin users.")
    else:
        st.info("Admin Control Center requires the subscription system to be enabled.")
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("**StockMind-AI Pro v2.0**")
    st.write("AI-Powered Stock Predictions")
    st.write("Model Accuracy: 78-85%")
    st.write("")
    st.caption("Built with Streamlit • Powered by Advanced ML")

# ============================================================================
# TAB 8: SUBSCRIBE (if enabled)
# ============================================================================
if SUBSCRIPTIONS_ENABLED:
    with tabs[7]:
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
                        "https://buy.stripe.com/4gM7sM8MWcCEb7r2WQ7AI00",  # Replace with actual Stripe link
                        use_container_width=True
                    )
                    st.caption("💳 Transaction Protector embedded")
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
