# Excel Portfolio Import UI Integration for app.py
# Replace the Portfolio tab section (around line 253-366) with this enhanced version:

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

# Don't forget to add this import at the top of app.py:
import io
