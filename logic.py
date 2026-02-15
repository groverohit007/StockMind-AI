import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
import os
from scipy.optimize import minimize  # NEW IMPORT FOR OPTIMIZER
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# --- 1. DATA & CACHING ---
def search_ticker(query, region="All"):
    """
    Searches Yahoo Finance for tickers, filtered by region/market.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=20&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
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

@st.cache_data(ttl=300)
def get_data(ticker, period="2y", interval="1d"):
    try:
        if interval in ['15m', '30m', '60m', '1h']:
            period = "1mo"
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def add_technical_overlays(df):
    """Adds Bollinger Bands and MACD for plotting."""
    df = df.copy()

    # Bollinger Bands
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()

    # MACD
    indicator_macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = indicator_macd.macd()
    df['MACD_Signal'] = indicator_macd.macd_signal()

    return df

import pdfplumber
import io
import re

def process_t212_pdf(uploaded_file):
    """
    Robust parser for Trading 212 'Confirmation of Holdings' PDFs.

    Returns a DataFrame with columns:
    Ticker, Buy_Price_USD, Shares, Date, Status, Currency
    """
    def _norm(s):
        return re.sub(r"\s+", " ", str(s or "")).strip()

    def _to_float(num_str):
        s = _norm(num_str)
        if not s:
            return None
        s = s.replace(":", ".")           # sometimes extracted as 27:431
        s = s.replace(",", "")            # thousands separators
        s = re.sub(r"[^0-9\.\-]", "", s)
        if s in ("", ".", "-", "-."):
            return None
        try:
            return float(s)
        except:
            return None

    def _parse_price(price_cell):
        txt = _norm(price_cell).upper()
        if not txt:
            return (None, None)

        currency = None
        if "GBX" in txt:
            currency = "GBP"
        elif "GBP" in txt:
            currency = "GBP"
        elif "USD" in txt:
            currency = "USD"
        elif "EUR" in txt:
            currency = "EUR"

        val = _to_float(txt)

        # Convert GBX (pence) -> GBP
        if "GBX" in txt and val is not None:
            val = val / 100.0

        return (val, currency)

    def _infer_ticker(instrument):
        name = _norm(instrument)
        m = re.search(r"\(([A-Z0-9\.\-]{1,15})\)", name.upper())
        if m:
            return m.group(1)
        # fallback: first token (best-effort)
        return re.split(r"\s+", name.upper())[0] if name else ""

    extracted = []

    try:
        # Make sure we're at start (Streamlit may reuse the file object)
        try:
            uploaded_file.seek(0)
        except:
            pass

        pdf_bytes = uploaded_file.read()
        pdf_stream = io.BytesIO(pdf_bytes)

        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Find the header row anywhere in the first few rows
                    header_row_idx = None
                    header = None
                    for i, row in enumerate(table[:10]):
                        joined = " | ".join(_norm(x).upper() for x in row)
                        if ("INSTRUMENT" in joined or "INSTRUMENT NAME" in joined) and ("ISIN" in joined) and ("QUANTITY" in joined or "QTY" in joined):
                            header_row_idx = i
                            header = [_norm(x).upper() for x in row]
                            break

                    if header_row_idx is None or header is None:
                        continue

                    def find_col(keywords):
                        for idx, col in enumerate(header):
                            c = (col or "").upper()
                            if any(k in c for k in keywords):
                                return idx
                        return None

                    c_instrument = find_col(["INSTRUMENT", "INSTRUMENT NAME", "NAME"])
                    c_isin = find_col(["ISIN"])  # optional
                    c_qty = find_col(["QUANTITY", "QTY"])
                    c_price = find_col(["PRICE", "AVG PRICE", "AVERAGE PRICE", "PURCHASE PRICE"])

                    if c_instrument is None or c_qty is None:
                        continue

                    for row in table[header_row_idx + 1:]:
                        if not row:
                            continue

                        # Safely access columns
                        instrument = _norm(row[c_instrument]) if len(row) > c_instrument else ""
                        if not instrument or instrument.upper() in ("TOTAL",):
                            continue

                        qty = _to_float(row[c_qty]) if len(row) > c_qty else None
                        if qty is None or qty <= 0:
                            continue

                        price_val, currency = (None, None)
                        if c_price is not None and len(row) > c_price:
                            price_val, currency = _parse_price(row[c_price])

                        if currency is None:
                            row_txt = " ".join(_norm(x).upper() for x in row)
                            if "GBX" in row_txt or "GBP" in row_txt:
                                currency = "GBP"
                            elif "EUR" in row_txt:
                                currency = "EUR"
                            else:
                                currency = "USD"

                        ticker = _infer_ticker(instrument)
                        if not ticker:
                            continue

                        extracted.append({
                            "Ticker": ticker,
                            "Buy_Price_USD": float(price_val) if price_val is not None else 0.0,
                            "Shares": float(qty),
                            "Date": pd.Timestamp.now(),
                            "Status": "OPEN",
                            "Currency": currency
                        })

        if not extracted:
            return pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency"])

        return pd.DataFrame(extracted)

    except Exception as e:
        print(f"process_t212_pdf error: {e}")
        return pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency"])


def sync_portfolio_with_df(new_data_df):
    """Overwrites or appends the portfolio with uploaded data."""
    if new_data_df is not None and not new_data_df.empty:
        new_data_df.to_csv(PORTFOLIO_FILE, index=False)
        return True
    return False

# --- 2. FUNDAMENTAL HEALTH CHECK ---
def get_fundamentals(ticker):
    """Fetches key financial ratios."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "Beta": info.get("beta", "N/A")
        }
    except:
        return None

# --- 3. CONSENSUS AI (ROBUST VERSION) ---
def train_consensus_model(data):
    """
    Trains an ensemble (RF + GB + LR) on technical features.

    Returns a 3-tuple to match app.py:
      (processed_df_or_None, features_list, votes_dict)

    processed_df includes indicators and a 'Confidence' column (bullish probability).
    votes includes per-model bullish probability for the latest bar.
    """
    if data is None or getattr(data, "empty", True):
        return None, [], {}

    df = data.copy()

    required_cols = {"Close", "High", "Low"}
    if not required_cols.issubset(set(df.columns)):
        return None, [], {}

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    # Indicators
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

    # Lagged returns
    df["Return"] = df["Close"].pct_change()
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["Lag_5"] = df["Return"].shift(5)

    # Target (noise-filtered)
    threshold = 0.002
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > threshold).astype(int)

    # Enough samples?
    if len(df) < 60:
        return None, [], {}

    df = df.dropna()
    if df.empty or len(df) < 50:
        return None, [], {}

    features = ["RSI", "SMA_20", "SMA_50", "ATR", "Lag_1", "Lag_2", "Lag_5"]
    X = df[features]
    y = df["Target"]

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft"
    )

    try:
        ensemble.fit(X, y)
    except Exception:
        return None, [], {}

    # Per-row confidence
    try:
        df["Confidence"] = ensemble.predict_proba(X)[:, 1]
    except Exception:
        df["Confidence"] = np.nan

    # Latest votes (for UI)
    votes = {}
    try:
        x_last = X.iloc[[-1]]
        for name, est in ensemble.named_estimators_.items():
            try:
                votes[name.upper()] = float(est.predict_proba(x_last)[0][1])
            except Exception:
                pass
        try:
            votes["ENSEMBLE"] = float(ensemble.predict_proba(x_last)[0][1])
        except Exception:
            pass
    except Exception:
        votes = {}

    return df, features, votes

def get_ai_signal(data):
    processed, features, _votes = train_consensus_model(data)
    if processed is None or processed.empty:
        return "HOLD"

    # Latest probability from processed (already computed)
    prob = float(processed["Confidence"].iloc[-1]) if "Confidence" in processed.columns else None
    if prob is None or np.isnan(prob):
        return "HOLD"

    if prob > 0.60:
        return "BUY"
    elif prob < 0.40:
        return "SELL"
    else:
        return "HOLD"



# --- 4. TRADING STRATEGIES ---
def rsi_strategy(df, buy_rsi=30, sell_rsi=70):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['Signal'] = 0
    df.loc[df['RSI'] < buy_rsi, 'Signal'] = 1
    df.loc[df['RSI'] > sell_rsi, 'Signal'] = -1
    return df

def sma_strategy(df, short_window=20, long_window=50):
    df = df.copy()
    df['SMA_Short'] = SMAIndicator(df['Close'], window=short_window).sma_indicator()
    df['SMA_Long'] = SMAIndicator(df['Close'], window=long_window).sma_indicator()
    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1
    return df

def macd_strategy(df):
    df = df.copy()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1
    df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1
    return df

def bb_strategy(df):
    df = df.copy()
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Signal'] = 0
    df.loc[df['Close'] < df['BB_Low'], 'Signal'] = 1
    df.loc[df['Close'] > df['BB_High'], 'Signal'] = -1
    return df

def atr_breakout_strategy(df, multiplier=2):
    df = df.copy()
    df['ATR'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    df['Upper'] = df['Close'].shift(1) + multiplier * df['ATR']
    df['Lower'] = df['Close'].shift(1) - multiplier * df['ATR']
    df['Signal'] = 0
    df.loc[df['Close'] > df['Upper'], 'Signal'] = 1
    df.loc[df['Close'] < df['Lower'], 'Signal'] = -1
    return df

def calculate_backtest(df, initial_capital=10000):
    df = df.copy()
    if 'Signal' not in df.columns:
        return None

    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0) * df['Market_Return']
    df['Equity'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

    total_return = df['Equity'].iloc[-1] - initial_capital
    roi = (df['Equity'].iloc[-1] / initial_capital - 1) * 100

    return df, total_return, roi


# --- 5. PORTFOLIO (PAPER TRADING) ---
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "paper_portfolio.csv")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_PORTFOLIO = pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency"])

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df = pd.read_csv(PORTFOLIO_FILE)
            if df.empty:
                return DEFAULT_PORTFOLIO.copy()
            return df
        except:
            return DEFAULT_PORTFOLIO.copy()
    return DEFAULT_PORTFOLIO.copy()

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

def add_trade_to_portfolio(ticker, buy_price, shares, currency="USD"):
    df = load_portfolio()
    df.loc[len(df)] = [ticker, buy_price, shares, pd.Timestamp.now(), "OPEN", currency]
    save_portfolio(df)

def close_trade(ticker):
    df = load_portfolio()
    df.loc[(df["Ticker"] == ticker) & (df["Status"] == "OPEN"), "Status"] = "CLOSED"
    save_portfolio(df)

def portfolio_value():
    df = load_portfolio()
    if df.empty:
        return 0, df

    open_trades = df[df["Status"] == "OPEN"].copy()
    if open_trades.empty:
        return 0, df

    total_value = 0
    for idx, row in open_trades.iterrows():
        data = get_data(row["Ticker"], period="5d", interval="1d")
        if data is None or data.empty:
            continue
        current_price = float(data["Close"].iloc[-1])
        total_value += current_price * float(row["Shares"])

    return total_value, df


# --- 6. OPENAI (AI INSIGHTS) ---
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def ai_market_summary(ticker, data):
    client = get_openai_client()
    if client is None:
        return "OpenAI API Key missing in secrets."

    latest_close = float(data["Close"].iloc[-1])
    latest_rsi = float(RSIIndicator(close=data["Close"], window=14).rsi().iloc[-1])

    prompt = f"""
You are a trading assistant. Summarize the market situation for {ticker}.
Latest close: {latest_close}
Latest RSI: {latest_rsi}

Give a short human-friendly summary, and end with a simple action: BUY / SELL / HOLD.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


# --- 7. OPTIMIZER (PORTFOLIO ALLOCATION) ---
def optimize_portfolio(tickers, period="1y"):
    prices = pd.DataFrame()
    for t in tickers:
        df = get_data(t, period=period, interval="1d")
        if df is None:
            continue
        prices[t] = df["Close"]

    prices = prices.dropna()
    if prices.empty or prices.shape[1] < 2:
        return None

    returns = prices.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)

    def portfolio_performance(weights):
        ret = np.sum(mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = ret / vol if vol != 0 else 0
        return ret, vol, sharpe

    def negative_sharpe(weights):
        return -portfolio_performance(weights)[2]

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    optimized = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = optimized.x

    ret, vol, sharpe = portfolio_performance(best_weights)

    return {
        "tickers": list(mean_returns.index),
        "weights": best_weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe
    }
