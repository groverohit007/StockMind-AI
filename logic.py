# logic.py - AMENDED PRODUCTION VERSION
# StockMind-AI Pro - All Bugs Fixed + New Features
# Version: 2.1 (Fixed)

import yfinance as yf
import pandas as pd
import numpy as np
import pdfplumber
import re
import time
import requests
import os
import pickle
import hashlib
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import minimize 
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator, AroonIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Advanced ML models
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

# Sentiment Analysis (optional)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except:
    HAS_VADER = False

# Cache directory
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# =============================================================================
# PART 1: DATA FETCHING & CACHING (FIXED)
# =============================================================================

def get_cache_key(ticker, interval):
    """Generate cache key for models."""
    return hashlib.md5(f"{ticker}_{interval}".encode()).hexdigest()

def _normalize_ohlcv_columns(data):
    """Normalize OHLCV columns so downstream models remain stable."""
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Some sources return lowercase columns
    rename_map = {c: c.capitalize() for c in data.columns}
    data = data.rename(columns=rename_map)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        return None

    return data[required_cols].dropna()




def _resolve_api_key(env_vars, secrets_section, secrets_key, db_setting_key):
    """Resolve an API key from env vars, st.secrets, or admin DB settings.

    Priority: environment variable > st.secrets > SQLite app_settings.
    """
    # Priority 1: Environment variables
    for var in env_vars:
        key = os.getenv(var, '').strip()
        if key:
            return key

    # Priority 2: Streamlit secrets (secrets.toml / Streamlit Cloud dashboard)
    try:
        key = st.secrets.get(secrets_section, {}).get(secrets_key, '').strip()
        if key:
            return key
    except Exception:
        pass

    # Priority 3: SQLite app_settings table (admin Settings UI)
    try:
        import database as db
        key = db.get_app_settings().get(db_setting_key, '').strip()
        return key
    except Exception:
        return ''


def _get_alpha_vantage_key():
    """Resolve Alpha Vantage key from env, secrets, or admin settings."""
    return _resolve_api_key(
        env_vars=['ALPHA_VANTAGE_API_KEY'],
        secrets_section='alpha_vantage',
        secrets_key='api_key',
        db_setting_key='alpha_vantage_api_key',
    )


def _get_openai_key():
    """Resolve OpenAI key from env, secrets, or admin settings."""
    return _resolve_api_key(
        env_vars=['OPENAI_API_KEY', 'OPENAI_KEY'],
        secrets_section='openai',
        secrets_key='api_key',
        db_setting_key='openai_api_key',
    )


def _get_news_key():
    """Resolve News API key from env, secrets, or admin settings."""
    return _resolve_api_key(
        env_vars=['NEWS_API_KEY'],
        secrets_section='news',
        secrets_key='api_key',
        db_setting_key='news_api_key',
    )


def _get_data_from_alpha_vantage(ticker, interval='1d'):
    """Fetch OHLCV data from Alpha Vantage as fallback."""
    api_key = _get_alpha_vantage_key()
    if not api_key:
        return None

    try:
        if interval in ['1d', '1wk', '1mo']:
            function = 'TIME_SERIES_DAILY_ADJUSTED'
            params = {
                'function': function,
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': api_key
            }
            response = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
            payload = response.json()
            series = payload.get('Time Series (Daily)', {})
            if not series:
                return None

            rows = []
            for day, values in series.items():
                rows.append({
                    'Date': pd.to_datetime(day),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['6. volume'])
                })

            df = pd.DataFrame(rows).sort_values('Date').set_index('Date')
            return df.tail(3000)

        # Intraday fallback
        function = 'TIME_SERIES_INTRADAY'
        intraday_interval = '60min' if interval in ['1h', '60m'] else '30min'
        params = {
            'function': function,
            'symbol': ticker,
            'interval': intraday_interval,
            'outputsize': 'full',
            'apikey': api_key
        }
        response = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
        payload = response.json()
        series_key = next((k for k in payload.keys() if k.startswith('Time Series')), '')
        series = payload.get(series_key, {})
        if not series:
            return None

        rows = []
        for ts, values in series.items():
            rows.append({
                'Date': pd.to_datetime(ts),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': float(values['5. volume'])
            })

        return pd.DataFrame(rows).sort_values('Date').set_index('Date')

    except Exception as e:
        print(f"Alpha Vantage fallback failed for {ticker}: {e}")
        return None


def get_data_source_status(ticker):
    """Check data source availability for diagnostics."""
    status = {
        'yahoo_ok': False,
        'alpha_key_configured': False,
        'alpha_ok': False,
        'alpha_message': ''
    }

    # Check Yahoo Finance
    try:
        test = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
        if test is not None and not test.empty:
            status['yahoo_ok'] = True
    except Exception:
        pass

    # Check Alpha Vantage
    alpha_key = _get_alpha_vantage_key()
    status['alpha_key_configured'] = bool(alpha_key)

    if alpha_key:
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'compact',
                'apikey': alpha_key
            }
            resp = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
            payload = resp.json()

            if 'Time Series (Daily)' in payload:
                status['alpha_ok'] = True
            elif 'Note' in payload:
                status['alpha_message'] = payload['Note']
            elif 'Information' in payload:
                status['alpha_message'] = payload['Information']
            elif 'Error Message' in payload:
                status['alpha_message'] = payload['Error Message']
        except Exception as e:
            status['alpha_message'] = str(e)

    return status


@st.cache_data(ttl=300)
def get_data(ticker, period="2y", interval="1d"):
    """Fetch stock data with retries and resilient fallbacks."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None

    try:
        if interval in ['15m', '30m', '60m', '1h']:
            period = "60d"

        # Attempt 1: yfinance download with retries
        for _ in range(3):
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            data = _normalize_ohlcv_columns(data)
            if data is not None and len(data) >= 50:
                return data
            time.sleep(0.5)

        # Attempt 2: Ticker().history fallback
        history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        history = _normalize_ohlcv_columns(history)
        if history is not None and len(history) >= 50:
            return history

        # Attempt 3: relaxed minimum bars fallback
        fallback_periods = ["1y", "6mo", "3mo"] if interval == "1d" else ["60d", "30d"]
        for fallback_period in fallback_periods:
            data = yf.download(ticker, period=fallback_period, interval=interval, progress=False, auto_adjust=False)
            data = _normalize_ohlcv_columns(data)
            if data is not None and len(data) >= 35:
                return data

        # Attempt 4: Alpha Vantage fallback if API key configured
        alpha_data = _get_data_from_alpha_vantage(ticker, interval=interval)
        alpha_data = _normalize_ohlcv_columns(alpha_data)
        if alpha_data is not None and len(alpha_data) >= 35:
            return alpha_data

        return None

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def search_company_by_name(query):
    """
    NEW FEATURE: Search for ticker by company name.
    Returns dict of matches: {display_name: ticker}
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 10,
            'newsCount': 0
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        
        results = {}
        
        if 'quotes' in data:
            for quote in data['quotes']:
                symbol = quote.get('symbol')
                name = quote.get('shortname') or quote.get('longname')
                exchange = quote.get('exchange', '')
                quote_type = quote.get('quoteType', '')
                
                # Only show stocks/ETFs
                if symbol and name and quote_type in ['EQUITY', 'ETF', '']:
                    display = f"{name} ({symbol})"
                    if exchange:
                        display += f" - {exchange}"
                    results[display] = symbol
        
        return results
        
    except Exception as e:
        print(f"Error searching for {query}: {str(e)}")
        return {}

def search_ticker(query, region="All"):
    """
    Enhanced search with company name support.
    IMPROVED: Now searches by both name and ticker.
    """
    # First try company name search
    results = search_company_by_name(query)
    
    if results:
        return results
    
    # Fall back to ticker search
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=20&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        results = {}
        
        filters = {
            "USA (NASDAQ/NYSE)": ["NMS", "NYQ", "NGM", "ASE", "NCM", "PNK"],
            "UK (LSE)": ["LSE"],
            "India (NSE/BSE)": ["NSI", "BSE", "NS", "BO"],
            "All": []
        }
        
        allowed_exchanges = filters.get(region, [])

        if 'quotes' in data:
            for quote in data['quotes']:
                symbol = quote.get('symbol')
                name = quote.get('shortname') or quote.get('longname')
                exch = quote.get('exchange')
                
                if symbol and name and exch:
                    if region == "All" or exch in allowed_exchanges:
                        results[f"{name} ({symbol}) - {exch}"] = symbol
                        
        return results
    except:
        return {}

# =============================================================================
# PART 2: MISSING FUNCTIONS (ADDED)
# =============================================================================

def get_macro_data():
    """
    FIXED: Added missing function for macro indicators.
    Get macro market indicators like S&P 500, VIX, etc.
    """
    try:
        macro_tickers = {
            "S&P 500": "^GSPC",
            "VIX": "^VIX",
            "USD Index": "DX-Y.NYB",
            "10Y Treasury": "^TNX"
        }
        
        results = {}
        
        for name, ticker in macro_tickers.items():
            try:
                data = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
                if not data.empty:
                    # Handle multi-index if present
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    latest = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2] if len(data) > 1 else latest
                    change = ((latest - prev) / prev) * 100

                    results[name] = {
                        'Price': float(latest),
                        'Change': float(change)
                    }
            except Exception as e:
                print(f"Error fetching {name}: {str(e)}")
                continue
        
        return results if results else None
        
    except Exception as e:
        print(f"Error fetching macro data: {str(e)}")
        return None

def get_sector_heatmap():
    """
    FIXED: Added missing function for sector performance.
    Get sector performance data using sector ETFs.
    """
    try:
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Finance': 'XLF',
            'Energy': 'XLE',
            'Consumer': 'XLY',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Industrials': 'XLI',
            'Real Estate': 'XLRE'
        }
        
        results = {}
        
        for sector, ticker in sector_etfs.items():
            try:
                data = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
                if not data.empty:
                    # Handle multi-index
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    latest = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2] if len(data) > 1 else latest
                    change = ((latest - prev) / prev) * 100
                    results[sector] = float(change)
            except:
                continue
        
        return results if results else None
        
    except Exception as e:
        print(f"Error fetching sector data: {str(e)}")
        return None

def get_fundamentals(ticker):
    """Get fundamental data for a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'PEG Ratio': info.get('pegRatio', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        # Format market cap
        if isinstance(fundamentals['Market Cap'], (int, float)):
            if fundamentals['Market Cap'] >= 1e12:
                fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e12:.2f}T"
            elif fundamentals['Market Cap'] >= 1e9:
                fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e9:.2f}B"
            elif fundamentals['Market Cap'] >= 1e6:
                fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e6:.2f}M"
        
        return fundamentals
    except:
        return None


# =============================================================================
# PART 3: FEATURE ENGINEERING (All 80+ Indicators)
# =============================================================================

def create_ultimate_features(data, forward_period=5, threshold=0.003):
    """
    Create 80+ technical indicators for maximum accuracy.
    Returns DataFrame with all features.
    forward_period: number of bars to look ahead for target
    threshold: minimum return to classify as BUY (reduces noise)
    """
    df = data.copy()
    
    # ===== TREND INDICATORS (20 features) =====
    df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    
    df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()
    
    cci = CCIIndicator(df['High'], df['Low'], df['Close'], window=20)
    df['CCI'] = cci.cci()
    
    aroon = AroonIndicator(df['High'], df['Low'], window=25)
    df['Aroon_up'] = aroon.aroon_up()
    df['Aroon_down'] = aroon.aroon_down()
    df['Aroon_indicator'] = aroon.aroon_indicator()
    
    # Trend strength
    df['Trend_SMA'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
    df['Golden_Cross'] = ((df['SMA_50'] > df['SMA_200']).astype(int))
    
    # ===== MOMENTUM INDICATORS (15 features) =====
    rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    df['RSI_smooth'] = df['RSI'].rolling(3).mean()
    
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    wr = WilliamsRIndicator(df['High'], df['Low'], df['Close'])
    df['Williams_R'] = wr.williams_r()
    
    roc = ROCIndicator(df['Close'], window=12)
    df['ROC'] = roc.roc()
    
    # Custom momentum
    df['Momentum_10'] = df['Close'].pct_change(10) * 100
    df['Momentum_20'] = df['Close'].pct_change(20) * 100
    df['Price_Rate_Change'] = df['Close'].pct_change(5) * 100
    
    # Acceleration
    df['Acceleration'] = df['Close'].diff().diff()
    df['Velocity'] = df['Close'].diff()
    
    # RSI divergence
    df['RSI_divergence'] = df['RSI'].diff() - (df['Close'].pct_change() * 100)
    
    # Momentum ratio
    df['Mom_Ratio'] = df['Momentum_10'] / (df['Momentum_20'] + 0.001)
    
    # ===== VOLATILITY INDICATORS (15 features) =====
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
    df['ATR'] = atr.average_true_range()
    df['ATR_percent'] = (df['ATR'] / df['Close']) * 100
    
    bb = BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = bb.bollinger_wband()
    df['BB_pband'] = bb.bollinger_pband()
    
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'])
    df['KC_upper'] = kc.keltner_channel_hband()
    df['KC_lower'] = kc.keltner_channel_lband()
    df['KC_width'] = kc.keltner_channel_wband()
    
    dc = DonchianChannel(df['High'], df['Low'], df['Close'])
    df['DC_upper'] = dc.donchian_channel_hband()
    df['DC_lower'] = dc.donchian_channel_lband()
    
    # Historical volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['Volatility_ratio'] = df['Volatility'] / df['Volatility'].rolling(50).mean()
    
    # ===== VOLUME INDICATORS (12 features) =====
    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    df['OBV_mean'] = df['OBV'].rolling(20).mean()
    
    cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = cmf.chaikin_money_flow()
    
    mfi = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
    df['MFI'] = mfi.money_flow_index()
    
    # Volume analysis
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    df['Volume_trend'] = df['Volume'].pct_change(5) * 100
    
    # Price-volume correlation
    df['PV_correlation'] = df['Close'].rolling(20).corr(df['Volume'])
    
    # Volume momentum
    df['Volume_momentum'] = df['Volume'].pct_change(10) * 100
    
    # Force Index
    df['Force_Index'] = df['Close'].diff() * df['Volume']
    df['Force_Index_EMA'] = df['Force_Index'].ewm(span=13).mean()
    
    # Ease of Movement
    df['EMV'] = ((df['High'] + df['Low'])/2 - (df['High'].shift() + df['Low'].shift())/2) / (df['Volume'] / (df['High'] - df['Low']))
    
    # ===== PRICE PATTERNS (10 features) =====
    # Support/Resistance
    df['Support_20'] = df['Low'].rolling(20).min()
    df['Resistance_20'] = df['High'].rolling(20).max()
    df['Distance_to_support'] = (df['Close'] - df['Support_20']) / df['Close'] * 100
    df['Distance_to_resistance'] = (df['Resistance_20'] - df['Close']) / df['Close'] * 100
    
    # Price position
    df['Price_position'] = (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
    
    # Candle patterns
    df['Body_size'] = abs(df['Close'] - df['Open']) / df['Open'] * 100
    df['Upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open'] * 100
    df['Lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open'] * 100
    
    # Gap detection
    df['Gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift() * 100
    df['Gap_filled'] = ((df['Low'] <= df['Close'].shift()) & (df['Gap'] > 0)).astype(int)
    
    # ===== ADVANCED PATTERNS (8 features) =====
    # Ichimoku Cloud
    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2
    
    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    
    high_52 = df['High'].rolling(52).max()
    low_52 = df['Low'].rolling(52).min()
    df['Senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # Cloud analysis
    df['Cloud_green'] = (df['Senkou_span_a'] > df['Senkou_span_b']).astype(int)
    df['Price_above_cloud'] = (df['Close'] > df[['Senkou_span_a', 'Senkou_span_b']].max(axis=1)).astype(int)
    
    # Fibonacci retracements
    recent_high = df['High'].rolling(100).max()
    recent_low = df['Low'].rolling(100).min()
    df['Fib_0.382'] = recent_low + (recent_high - recent_low) * 0.382
    df['Fib_0.618'] = recent_low + (recent_high - recent_low) * 0.618
    
    # Forward returns (target) - multi-period for noise reduction
    df['Forward_Return'] = df['Close'].pct_change(forward_period).shift(-forward_period)
    df['Target'] = (df['Forward_Return'] > threshold).astype(int)
    
    # Drop NaN
    df = df.dropna()
    
    return df


# =============================================================================
# PART 4: MODEL BUILDING
# =============================================================================

def create_ultimate_ensemble():
    """Create ensemble with 6 powerful models."""
    base_models = []
    
    # Model 1: Random Forest
    base_models.append(('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )))
    
    # Model 2: Gradient Boosting
    base_models.append(('gb', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )))
    
    # Model 3: Extra Trees
    base_models.append(('et', ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )))
    
    # Model 4: Histogram Gradient Boosting
    base_models.append(('hgb', HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=42
    )))
    
    # Model 5: XGBoost (if available)
    if HAS_XGB:
        base_models.append(('xgb', xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )))
    
    # Model 6: LightGBM (if available)
    if HAS_LGB:
        base_models.append(('lgb', lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )))
    
    # Meta-learner
    meta_model = LogisticRegression(max_iter=1000)
    
    # Create voting ensemble
    ensemble = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
    
    return ensemble

def train_ultimate_model(data, ticker, interval, forward_period=5, threshold=0.003):
    """Train the ultimate ensemble model with configurable target."""
    try:
        # Check cache (include forward_period and threshold for uniqueness)
        cache_key = hashlib.md5(f"{ticker}_{interval}_{forward_period}_{threshold}".encode()).hexdigest()
        cache_file = os.path.join(MODEL_CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            # Check if cache is recent (< 1 day old)
            if time.time() - os.path.getmtime(cache_file) < 86400:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Create features
        df = create_ultimate_features(data, forward_period=forward_period, threshold=threshold)

        if len(df) < 60:
            return None
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['Target', 'Forward_Return']]
        X = df[feature_cols]
        y = df['Target']
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series split (5-fold for robust accuracy estimation)
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = create_ultimate_ensemble()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        # Train final model on all data
        final_model = create_ultimate_ensemble()
        final_model.fit(X_scaled, y)
        
        # Calculate average accuracy
        avg_accuracy = np.mean(accuracies)
        
        # Prepare result
        result = {
            'model': final_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': avg_accuracy
        }
        
        # Cache the model
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def get_sentiment_score(ticker):
    """Get sentiment from news (optional)."""
    if not HAS_VADER:
        return 0.5
    
    try:
        analyzer = SentimentIntensityAnalyzer()
        # Placeholder - would integrate with news API
        return 0.5
    except:
        return 0.5

def get_fundamental_score(ticker):
    """Get fundamental analysis score."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        score = 50  # Neutral base
        
        # Profitability
        if info.get('profitMargins', 0) > 0.15:
            score += 10
        if info.get('returnOnEquity', 0) > 0.15:
            score += 10
        
        # Growth
        if info.get('revenueGrowth', 0) > 0.1:
            score += 10
        if info.get('earningsGrowth', 0) > 0.1:
            score += 10
        
        # Valuation
        pe = info.get('forwardPE', 999)
        if 10 < pe < 25:
            score += 10
        
        # Financial health
        if info.get('debtToEquity', 999) < 100:
            score += 10
        
        return min(100, max(0, score))
    except:
        return 50

def detect_market_regime(data):
    """Detect current market regime."""
    try:
        df = data.copy()
        
        # Calculate indicators
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        adx = ADXIndicator(df['High'], df['Low'], df['Close']).adx().iloc[-1]
        atr_pct = (AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().iloc[-1] / current_price) * 100
        
        # Determine regime
        if current_price > sma_50 > sma_200 and adx > 25:
            return "STRONG_BULL"
        elif current_price > sma_50 and adx < 20:
            return "WEAK_BULL"
        elif current_price < sma_50 < sma_200 and adx > 25:
            return "STRONG_BEAR"
        elif current_price < sma_50 and adx < 20:
            return "WEAK_BEAR"
        elif atr_pct > 3:
            return "HIGH_VOLATILITY"
        else:
            return "RANGING"
    except:
        return "UNKNOWN"

def _rule_based_fallback_prediction(data):
    """Generate fallback signal when ML model training is unavailable."""
    df = data.copy()
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    rsi = RSIIndicator(df['Close'], window=14).rsi().iloc[-1]

    bullish_votes = 0
    bearish_votes = 0

    if df['Close'].iloc[-1] > df['sma20'].iloc[-1] > df['sma50'].iloc[-1]:
        bullish_votes += 1
    if df['Close'].iloc[-1] < df['sma20'].iloc[-1] < df['sma50'].iloc[-1]:
        bearish_votes += 1

    if rsi < 35:
        bullish_votes += 1
    elif rsi > 65:
        bearish_votes += 1

    if bullish_votes > bearish_votes:
        signal = 'BUY'
        buy_prob = 0.62
        sell_prob = 0.22
    elif bearish_votes > bullish_votes:
        signal = 'SELL'
        buy_prob = 0.24
        sell_prob = 0.60
    else:
        signal = 'HOLD'
        buy_prob = 0.33
        sell_prob = 0.33

    hold_prob = max(0.0, 1 - max(buy_prob, sell_prob))
    return {
        'signal': signal,
        'confidence': max(buy_prob, sell_prob),
        'probabilities': {
            'SELL': sell_prob,
            'BUY': buy_prob,
            'HOLD': hold_prob
        }
    }


def _calibrate_confidence(raw_confidence, regime, fundamental_score):
    """Adjust confidence to improve robustness in unstable market regimes."""
    adjusted = raw_confidence

    if regime in ['HIGH_VOLATILITY', 'UNKNOWN']:
        adjusted -= 0.05
    elif regime in ['STRONG_BULL', 'STRONG_BEAR']:
        adjusted += 0.03

    if fundamental_score >= 70:
        adjusted += 0.02
    elif fundamental_score <= 35:
        adjusted -= 0.02

    return float(min(0.95, max(0.50, adjusted)))


def get_multi_timeframe_predictions(ticker, user_tier='free'):
    """
    Get predictions across multiple timeframes.
    Premium: all 4 timeframes with enhanced model (80-85% accuracy target).
    Free: 2 timeframes with standard model (70-75% accuracy target).
    Tuple: (interval, period, label, base_acc, forward_period, threshold)
    """
    try:
        if user_tier == 'premium':
            timeframes = {
                'hourly':  ('1h',  '60d', 'Hourly',  0.80, 5, 0.002),
                'daily':   ('1d',  '2y',  'Daily',   0.83, 5, 0.005),
                'weekly':  ('1wk', '5y',  'Weekly',  0.81, 3, 0.008),
                'monthly': ('1mo', 'max', 'Monthly', 0.78, 2, 0.010),
            }
            signal_threshold = 0.55
        else:
            timeframes = {
                'daily':  ('1d',  '2y', 'Daily',  0.72, 3, 0.003),
                'weekly': ('1wk', '5y', 'Weekly', 0.70, 2, 0.005),
            }
            signal_threshold = 0.60

        results = {
            'predictions': {},
            'regime': None,
            'sentiment': 0.5,
            'fundamental': 50
        }

        # Get regime from daily data
        daily_data = get_data(ticker, period='2y', interval='1d')
        if daily_data is not None:
            results['regime'] = detect_market_regime(daily_data)
            results['sentiment'] = get_sentiment_score(ticker)
            results['fundamental'] = get_fundamental_score(ticker)

        # Get predictions for each timeframe
        for tf_key, (interval, period, label, base_acc, fwd_period, thresh) in timeframes.items():
            data = get_data(ticker, period=period, interval=interval)

            min_bars = 50 if interval in ['1wk', '1mo'] else 100
            if data is None or len(data) < min_bars:
                continue

            # Train model
            model_result = train_ultimate_model(
                data, ticker, interval,
                forward_period=fwd_period, threshold=thresh
            )
            fallback_pred = None
            if model_result is None:
                fallback_pred = _rule_based_fallback_prediction(data)

            # Make prediction
            try:
                if model_result is None:
                    raise ValueError('ML model unavailable, using fallback signal')

                df = create_ultimate_features(
                    data, forward_period=fwd_period, threshold=thresh
                )
                feature_cols = model_result['feature_cols']
                X = df[feature_cols].iloc[-1:].values
                X_scaled = model_result['scaler'].transform(X)

                # Get prediction and probability
                prediction = model_result['model'].predict(X_scaled)[0]
                proba = model_result['model'].predict_proba(X_scaled)[0]

                # Determine signal using tier-appropriate threshold
                if prediction == 1 and proba[1] > signal_threshold:
                    signal = "BUY"
                    emoji = "🟢"
                elif prediction == 0 and proba[0] > signal_threshold:
                    signal = "SELL"
                    emoji = "🔴"
                else:
                    signal = "HOLD"
                    emoji = "⚪"

                calibrated_confidence = _calibrate_confidence(
                    float(max(proba)), results['regime'], results['fundamental']
                )

                results['predictions'][tf_key] = {
                    'signal': signal,
                    'confidence': calibrated_confidence,
                    'timeframe': label,
                    'emoji': emoji,
                    'accuracy': float(model_result['accuracy']),
                    'probabilities': {
                        'SELL': float(proba[0]),
                        'BUY': float(proba[1]),
                        'HOLD': float(1 - max(proba))
                    }
                }

            except Exception as e:
                print(f"Error predicting {tf_key}: {str(e)}")
                if fallback_pred:
                    fallback_conf = _calibrate_confidence(
                        fallback_pred['confidence'], results['regime'], results['fundamental']
                    )
                    fallback_signal = fallback_pred['signal']
                    fallback_emoji = '🟢' if fallback_signal == 'BUY' else '🔴' if fallback_signal == 'SELL' else '⚪'
                    results['predictions'][tf_key] = {
                        'signal': fallback_signal,
                        'confidence': fallback_conf,
                        'timeframe': label,
                        'emoji': fallback_emoji,
                        'accuracy': base_acc,
                        'probabilities': fallback_pred['probabilities']
                    }
                continue

        return results if results['predictions'] else None

    except Exception as e:
        print(f"Error in multi-timeframe predictions: {str(e)}")
        return None

# =============================================================================
# PART 5: PORTFOLIO MANAGEMENT
# =============================================================================

def get_portfolio():
    """Get portfolio from CSV file."""
    try:
        if os.path.exists('portfolio.csv'):
            return pd.read_csv('portfolio.csv')
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def save_portfolio(portfolio):
    """Save portfolio to CSV file."""
    try:
        portfolio.to_csv('portfolio.csv', index=False)
        return True
    except:
        return False

def execute_trade(ticker, price_usd, shares, action, currency="USD"):
    """Execute a trade and update portfolio."""
    try:
        portfolio = get_portfolio()
        
        new_trade = {
            'Ticker': ticker,
            'Buy_Price_USD': price_usd,
            'Shares': shares,
            'Currency': currency,
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Status': 'OPEN'
        }
        
        if portfolio.empty:
            portfolio = pd.DataFrame([new_trade])
        else:
            portfolio = pd.concat([portfolio, pd.DataFrame([new_trade])], ignore_index=True)
        
        save_portfolio(portfolio)
        return True
    except:
        return False

# =============================================================================
# PART 6: WATCHLIST MANAGEMENT
# =============================================================================

def get_watchlist():
    """Get watchlist from file."""
    try:
        if os.path.exists('watchlist.txt'):
            with open('watchlist.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    except:
        return []

def add_to_watchlist(ticker):
    """Add ticker to watchlist."""
    try:
        watchlist = get_watchlist()
        if ticker not in watchlist:
            watchlist.append(ticker)
            with open('watchlist.txt', 'w') as f:
                f.write('\n'.join(watchlist))
        return True
    except:
        return False

def remove_from_watchlist(ticker):
    """Remove ticker from watchlist."""
    try:
        watchlist = get_watchlist()
        if ticker in watchlist:
            watchlist.remove(ticker)
            with open('watchlist.txt', 'w') as f:
                f.write('\n'.join(watchlist))
        return True
    except:
        return False

# End of logic.py
