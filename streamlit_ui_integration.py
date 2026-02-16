# Streamlit UI Integration for Multi-Timeframe Predictions
# Add this to your app.py to replace the current Terminal tab

# --- TAB 1: ENHANCED TERMINAL WITH MULTI-TIMEFRAME PREDICTIONS ---
with tabs[0]:
    if ticker:
        st.title(f"📊 {ticker} AI Analysis Dashboard")
        
        # TOP ACTION BAR
        c_top1, c_top2, c_top3 = st.columns([2, 1, 1])
        with c_top2:
            if st.button(f"➕ Add to Watchlist"):
                logic.add_to_watchlist(ticker)
                st.success(f"Added {ticker}")
        
        with c_top3:
            refresh = st.button("🔄 Refresh Analysis")
        
        # MAIN ANALYSIS
        with st.spinner("🤖 Training Multi-Timeframe AI Models..."):
            # Get comprehensive predictions
            result = logic.get_multi_timeframe_predictions(ticker)
            
            if result and 'predictions' in result:
                predictions = result['predictions']
                processed_data = result.get('processed_data', {})
                
                # ==================== CURRENT PRICE & QUICK STATS ====================
                st.markdown("---")
                st.subheader("💰 Current Market Data")
                
                # Get latest data for current price
                latest_data = None
                if 'daily' in processed_data and not processed_data['daily'].empty:
                    latest_data = processed_data['daily'].iloc[-1]
                
                if latest_data is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(
                        "Current Price",
                        f"${latest_data['Close']:.2f}",
                        f"{latest_data['Returns']*100:+.2f}%"
                    )
                    
                    col2.metric(
                        "Volume",
                        f"{latest_data['Volume']/1e6:.1f}M",
                        f"{latest_data['Volume_Change']*100:+.1f}%" if 'Volume_Change' in latest_data else None
                    )
                    
                    col3.metric(
                        "RSI (14)",
                        f"{latest_data['RSI_14']:.1f}",
                        "Overbought" if latest_data['RSI_14'] > 70 else "Oversold" if latest_data['RSI_14'] < 30 else "Neutral"
                    )
                    
                    col4.metric(
                        "Volatility (ATR%)",
                        f"{latest_data['ATR_Percent']:.2f}%",
                        "High" if latest_data['ATR_Percent'] > 3 else "Low"
                    )
                
                # ==================== MULTI-TIMEFRAME PREDICTIONS ====================
                st.markdown("---")
                st.subheader("🎯 Multi-Timeframe AI Predictions")
                
                # Create a nice table view
                timeframe_labels = {
                    'hourly': '24-Hour Outlook',
                    'daily': 'Weekly Outlook',
                    'weekly': 'Monthly Outlook',
                    'monthly': 'Quarterly Outlook'
                }
                
                # Display predictions in columns
                pred_cols = st.columns(len(predictions))
                
                for idx, (tf, pred_data) in enumerate(predictions.items()):
                    with pred_cols[idx]:
                        label = timeframe_labels.get(tf, tf.title())
                        
                        # Card-like display
                        signal = pred_data['signal']
                        confidence = pred_data['confidence'] * 100
                        
                        # Color coding
                        if signal == 'BUY':
                            color = "#00ff00"
                            emoji = "🟢"
                        elif signal == 'SELL':
                            color = "#ff0000"
                            emoji = "🔴"
                        else:
                            color = "#ffaa00"
                            emoji = "⚪"
                        
                        st.markdown(f"""
                        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {color}'>
                            <h4 style='margin: 0; color: white;'>{label}</h4>
                            <h2 style='margin: 10px 0; color: {color};'>{emoji} {signal}</h2>
                            <p style='margin: 0; color: #888;'>Confidence: {confidence:.1f}%</p>
                            <hr style='margin: 10px 0; border-color: #333;'>
                            <small style='color: #666;'>
                                Buy: {pred_data['probabilities']['BUY']*100:.0f}% | 
                                Hold: {pred_data['probabilities']['HOLD']*100:.0f}% | 
                                Sell: {pred_data['probabilities']['SELL']*100:.0f}%
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show model accuracy
                        if 'metrics' in pred_data:
                            accuracy = pred_data['metrics'].get('test_accuracy', 0) * 100
                            st.caption(f"Model Accuracy: {accuracy:.1f}%")
                
                # ==================== CONSENSUS SIGNAL ====================
                st.markdown("---")
                st.subheader("🎲 AI Consensus")
                
                # Calculate weighted consensus
                total_buy = sum(p['probabilities']['BUY'] for p in predictions.values())
                total_sell = sum(p['probabilities']['SELL'] for p in predictions.values())
                total_hold = sum(p['probabilities']['HOLD'] for p in predictions.values())
                
                total = total_buy + total_sell + total_hold
                
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
                
                # ==================== MODEL PERFORMANCE ====================
                st.markdown("---")
                st.subheader("📈 Model Performance Metrics")
                
                perf_data = []
                for tf, pred_data in predictions.items():
                    if 'metrics' in pred_data:
                        metrics = pred_data['metrics']
                        perf_data.append({
                            'Timeframe': timeframe_labels.get(tf, tf.title()),
                            'Accuracy': f"{metrics.get('test_accuracy', 0)*100:.1f}%",
                            'Precision': f"{metrics.get('precision', 0)*100:.1f}%",
                            'Recall': f"{metrics.get('recall', 0)*100:.1f}%",
                            'F1 Score': f"{metrics.get('f1', 0)*100:.1f}%",
                            'CV Score': f"{metrics.get('cv_mean', 0)*100:.1f}% ± {metrics.get('cv_std', 0)*100:.1f}%"
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
                
                # ==================== FEATURE IMPORTANCE ====================
                st.markdown("---")
                
                with st.expander("🔍 Top Predictive Features"):
                    engine = result.get('engine')
                    if engine and hasattr(engine, 'feature_importance'):
                        for tf, importance_df in engine.feature_importance.items():
                            st.write(f"**{timeframe_labels.get(tf, tf.title())} Model:**")
                            
                            # Show top 10 features
                            top_features = importance_df.head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=top_features['importance'],
                                y=top_features['feature'],
                                orientation='h',
                                marker=dict(color=top_features['importance'], colorscale='Viridis')
                            ))
                            
                            fig.update_layout(
                                title=f"Top 10 Features - {timeframe_labels.get(tf, tf)}",
                                xaxis_title="Importance",
                                yaxis_title="Feature",
                                height=400,
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("---")
                
                # ==================== ADVANCED CHARTING ====================
                st.markdown("---")
                st.subheader("📊 Advanced Technical Chart")
                
                # Use daily data for chart
                if 'daily' in processed_data and not processed_data['daily'].empty:
                    chart_data = processed_data['daily']
                    
                    # Chart options
                    c1, c2, c3, c4 = st.columns(4)
                    show_bb = c1.checkbox("Bollinger Bands", value=True)
                    show_ma = c2.checkbox("Moving Averages", value=True)
                    show_signals = c3.checkbox("AI Signals", value=True)
                    show_volume = c4.checkbox("Volume", value=True)
                    
                    # Create candlestick chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name="Price"
                    ))
                    
                    if show_bb:
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['BB_High'],
                            line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                            name='BB Upper'
                        ))
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['BB_Low'],
                            line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                            fill='tonexty',
                            name='BB Lower'
                        ))
                    
                    if show_ma:
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['SMA_20'],
                            line=dict(color='orange', width=1.5),
                            name='SMA 20'
                        ))
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['SMA_50'],
                            line=dict(color='blue', width=1.5),
                            name='SMA 50'
                        ))
                    
                    fig.update_layout(
                        title=f"{ticker} Price Chart",
                        yaxis_title="Price (USD)",
                        xaxis_title="Date",
                        height=600,
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    if show_volume:
                        vol_fig = go.Figure()
                        
                        colors = ['red' if chart_data['Close'].iloc[i] < chart_data['Open'].iloc[i] 
                                 else 'green' for i in range(len(chart_data))]
                        
                        vol_fig.add_trace(go.Bar(
                            x=chart_data.index,
                            y=chart_data['Volume'],
                            marker_color=colors,
                            name='Volume'
                        ))
                        
                        vol_fig.update_layout(
                            title="Volume",
                            yaxis_title="Volume",
                            height=200,
                            showlegend=False
                        )
                        
                        st.plotly_chart(vol_fig, use_container_width=True)
                
                # ==================== RISK ASSESSMENT ====================
                st.markdown("---")
                st.subheader("⚠️ Risk Assessment")
                
                if latest_data is not None:
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    # Volatility Risk
                    atr_pct = latest_data.get('ATR_Percent', 0)
                    vol_risk = "High" if atr_pct > 3 else "Medium" if atr_pct > 1.5 else "Low"
                    risk_col1.metric("Volatility Risk", vol_risk, f"{atr_pct:.2f}%")
                    
                    # Momentum Risk
                    rsi = latest_data.get('RSI_14', 50)
                    mom_risk = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    risk_col2.metric("Momentum", mom_risk, f"RSI: {rsi:.1f}")
                    
                    # Trend Risk
                    adx = latest_data.get('ADX', 0)
                    trend_strength = "Strong" if adx > 25 else "Weak"
                    risk_col3.metric("Trend Strength", trend_strength, f"ADX: {adx:.1f}")
            
            else:
                st.error("Unable to generate predictions. Please try again.")


# ==================== ADDITIONAL: MODEL COMPARISON TAB ====================
# You can add this as a new tab to compare different model architectures

with st.expander("🔬 Model Architecture Comparison"):
    st.markdown("""
    ### Current Multi-Timeframe System
    
    **Models Used:**
    - XGBoost (Primary)
    - LightGBM (Fast)
    - CatBoost (Robust)
    - Random Forest (Ensemble)
    - Extra Trees (Diversity)
    - Hist Gradient Boosting (Speed)
    
    **Ensemble Method:** Stacked with Logistic Regression meta-learner
    
    **Features:** 80+ technical indicators including:
    - Momentum: RSI, Stochastic, Williams %R, ROC
    - Trend: MACD, ADX, CCI, Aroon, EMAs
    - Volatility: Bollinger Bands, ATR, Keltner, Donchian
    - Volume: OBV, CMF, MFI, VWAP
    - Advanced: Market regime, fractal patterns, correlations
    
    **Validation:** Time-series cross-validation with 5 splits
    
    **Performance:** Check the metrics table above for detailed accuracy
    """)
