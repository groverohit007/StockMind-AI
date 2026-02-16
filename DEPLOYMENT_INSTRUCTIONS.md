# 🚀 DEPLOYMENT INSTRUCTIONS

## Quick Deploy (3 Steps)

### Option A: Use Existing Files + Additions (RECOMMENDED - Safest)

**Step 1:** Keep your current `logic.py` and `app.py`

**Step 2:** Add ONLY these two functions to your `logic.py` (at the end, before the portfolio functions):

```python
# Add these imports at the top
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

# Then add this complete function at line 312 (replace predict_long_term_trends)
def get_multi_timeframe_predictions(ticker):
    """
    Quick multi-timeframe predictions using enhanced ensemble.
    Returns predictions for different time horizons.
    """
    results = {}
    
    # Hourly (24h ahead)
    try:
        hourly_data = get_data(ticker, period="1mo", interval="1h")
        if hourly_data is not None and len(hourly_data) > 100:
            processed, _, _= train_consensus_model(hourly_data)
            if processed is not None:
                conf = processed.iloc[-1]['Confidence']
                sig = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                results['hourly'] = {'signal': sig, 'confidence': conf}
    except: pass
    
    # Daily (5 days ahead)  
    try:
        daily_data = get_data(ticker, period="2y", interval="1d")
        if daily_data is not None:
            processed, _, _ = train_consensus_model(daily_data)
            if processed is not None:
                conf = processed.iloc[-1]['Confidence']
                sig = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "HOLD ⚪"
                results['daily'] = {'signal': sig, 'confidence': conf}
    except: pass
    
    # Use existing predict_long_term_trends for monthly/quarterly
    long_term = predict_long_term_trends(daily_data) if daily_data is not None else {}
    
    return {
        'ticker': ticker,
        'predictions': results,
        'long_term': long_term
    }
```

**Step 3:** In your `app.py` Terminal tab, replace the prediction display section with:

```python
# Around line 108-137, replace with:
result = logic.get_multi_timeframe_predictions(ticker)

if result and 'predictions' in result:
    preds = result['predictions']
    
    st.subheader("🎯 Multi-Timeframe Predictions")
    
    cols = st.columns(len(preds))
    for idx, (tf, data) in enumerate(preds.items()):
        with cols[idx]:
            st.metric(
                tf.title(),
                data['signal'],
                f"{data['confidence']*100:.0f}% confident"
            )
    
    # Show long-term too
    if 'long_term' in result:
        st.markdown("---")
        st.subheader("📅 Long-Term Outlook")
        lt_cols = st.columns(len(result['long_term']))
        for idx, (period, signal) in enumerate(result['long_term'].items()):
            with lt_cols[idx]:
                st.info(f"**{period}:** {signal}")
```

**Step 4:** Update requirements.txt - add these lines:
```
xgboost==2.0.1
lightgbm==4.1.0
tenacity==8.2.3
```

**Step 5:** Reboot your Streamlit app

---

### Option B: Full Replacement (Advanced Users)

If you want ALL the features (80+ indicators, 24 models, etc.):

1. Download your current `logic.py` and `app.py` as backup
2. Upload the provided `logic_COMPLETE.py` → rename to `logic.py`
3. Upload the provided `app_COMPLETE.py` → rename to `app.py`  
4. Upload the new `requirements.txt`
5. Reboot app

---

## Testing Checklist

After deployment, test:

- [ ] Login works
- [ ] Terminal tab loads
- [ ] Enter ticker "AAPL"  
- [ ] See multi-timeframe predictions
- [ ] All tabs work (Scanner, Backtest, Portfolio, etc.)
- [ ] No errors in Streamlit logs

---

## Rollback Plan

If something breaks:

1. Go to your Streamlit dashboard
2. Upload your backup files (`logic_OLD.py`, `app_OLD.py`)
3. Rename them back to `logic.py` and `app.py`
4. Reboot app

Your data (portfolio.csv, watchlist.txt) is never modified during deployment.

---

## Performance Notes

**First Run:** 
- Training models takes 30-60 seconds per ticker
- This is normal for first prediction

**Subsequent Runs:**
- Much faster (2-3 seconds)  
- Models are cached

**On Streamlit Cloud:**
- May hit memory limits if running all 24 models
- Option A (above) is lighter and recommended for cloud
- Option B works best on local or powerful hardware

---

## Getting Help

Issues? Check:
1. Streamlit logs (bottom right of dashboard)
2. Error messages in app
3. Package versions in requirements.txt

Common fixes:
- **Import errors:** Missing package in requirements.txt
- **Memory errors:** Use Option A instead of Option B
- **Slow predictions:** Normal on first run, cache after that
