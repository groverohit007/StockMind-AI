import yfinance as yf
import pandas as pd
import numpy as np
import pdfplumber
import re
import time
import streamlit as st
import requests
import os
from scipy.optimize import minimize 
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
        if interval in ['15m', '30m', '60m', '1h']: period = "1mo"
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty: return None
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

# --- IMPROVED PDF PARSER FOR TRADING 212 ---
def process_t212_pdf(file):
    extracted_data = []
    current_account_type = "Invest" # Default
    
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                
                # 1. Detect Account Type Section
                if "Trading 212 Stocks ISA" in text:
                    current_account_type = "ISA"
                elif "Trading 212 Invest" in text:
                    current_account_type = "Invest"
                
                # 2. Parsing Strategy: Regex for Quoted CSV format
                # T212 PDF often hides data as: "Name","ISIN","Qty","Price"
                # We use a robust regex to capture this even if it spans lines
                
                # Regex Explanation:
                # "([^"]+)"  -> Group 1: Name (anything except quotes)
                # \s*,\s* -> Comma with optional whitespace
                # "([A-Z0-9]{9,12})" -> Group 2: ISIN (9-12 alphanum chars)
                # "([\d:.,]+)" -> Group 3: Quantity (digits, colons, dots, commas)
                # "([A-Z]{3}\s?[\d.]+)" -> Group 4: Price (Currency + Value)
                
                pattern = re.compile(r'"([^"]+)"\s*,\s*"([A-Z0-9]{9,12})"\s*,\s*"([\d:.,]+)"\s*,\s*"([A-Z]{3}\s?[\d.]+)"')
                
                matches = pattern.findall(text)
                
                if matches:
                    for match in matches:
                        raw_name, isin_code, raw_qty, raw_price = match
                        
                        # Clean Name
                        full_name = raw_name.replace('\n', '').strip()
                        if "INSTRUMENT" in full_name: continue # Skip header
                        
                        # Clean ISIN
                        isin_code = isin_code.strip()
                        
                        # Clean Quantity (Fix typos like '27:43' -> '27.43')
                        qty_str = raw_qty.replace(':', '.').replace(',', '')
                        try:
                            qty = float(qty_str)
                        except:
                            continue # Skip bad data
                            
                        # Clean Price & Currency
                        price_parts = raw_price.replace('\n', '').strip().split()
                        if len(price_parts) < 2: 
                            # Handle case where space might be missing "USD100"
                            # Simple regex to split alpha from numeric
                            p_match = re.match(r"([A-Z]{3})([\d.]+)", raw_price)
                            if p_match:
                                currency, price_val_str = p_match.groups()
                            else:
                                continue
                        else:
                            currency, price_val_str = price_parts[0], price_parts[1]
                        
                        try:
                            price_val = float(price_val_str)
                            # Convert Pence (GBX) to Pounds (GBP)
                            if currency == "GBX":
                                price_val /= 100
                                currency = "GBP"
                        except:
                            price_val = 0.0

                        # --- TICKER RESOLUTION ---
                        ticker = None
                        
                        # A. Try ISIN Lookup (Best)
                        if isin_code:
                            res = search_ticker(isin_code)
                            if res: ticker = list(res.values())[0]
                        
                        # B. Try Name Lookup (Fallback)
                        if not ticker:
                            res = search_ticker(full_name)
                            if res: ticker = list(res.values())[0]
                        
                        # C. Last Resort: First word of name
                        if not ticker:
                            ticker = full_name.split()[0].upper()

                        if qty > 0 and ticker:
                            extracted_data.append({
                                "Ticker": ticker,
                                "Name": full_name,
                                "ISIN": isin_code,
                                "Buy_Price_USD": price_val, # Storing as generic price column
                                "Shares": qty,
                                "Date": pd.Timestamp.now(),
                                "Status": "OPEN",
                                "Currency": currency,
                                "Account": current_account_type
                            })
                else:
                    # FALLBACK: Try standard table extraction if Regex fails
                    tables = page.extract_tables()
                    # (Simplified fallback logic can go here if needed, 
                    # but the Regex above is highly specific to the file you provided)
                    pass

        if not extracted_data:
            print("Debug: No assets found in PDF.")
            return None
            
        return pd.DataFrame(extracted_data)

    except Exception as e:
        print(f"Critical PDF Error: {e}")
        return None

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
    df = data.copy()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].fillna(0)
    
    # 1. Standard Indicators
    df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
    df['SMA_20'] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df['ATR'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    
    # 2. NEW: Lagged Features (Memory)
    df['Return'] = df['Close'].pct_change()
    df['Lag_1'] = df['Return'].shift(1) # Yesterday
    df['Lag_2'] = df['Return'].shift(2) # 2 Days ago
    df['Lag_5'] = df['Return'].shift(5) # Weekly trend

    # 3. NEW: Robust Target (Noise Filter)
    # Only Buy if price rises > 0.2% (covers fees/spread)
    threshold = 0.002 
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > threshold).astype(int)
    
    if len(df) < 50: return None, [], {}
    df.dropna(inplace=True)
    if df.empty: return None, [], {}
    
    # Updated Features List
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'Volume', 'Lag_1', 'Lag_2', 'Lag_5']
    features = [f for f in features if f in df.columns]
    
    train_size = int(len(df) * 0.90)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    lr = LogisticRegression(solver='liblinear')
    
    ensemble = VotingClassifier(estimators=[('RF', rf), ('GB', gb), ('LR', lr)], voting='soft')
    ensemble.fit(train[features], train['Target'])
    
    probs = ensemble.predict_proba(test[features])[:, 1]
    test = test.copy()
    test['Confidence'] = probs
    
    # Votes for display
    rf.fit(train[features], train['Target'])
    gb.fit(train[features], train['Target'])
    lr.fit(train[features], train['Target'])
    
    last_row = test.iloc[[-1]]
    votes = {
        "Random Forest": rf.predict_proba(last_row[features])[:, 1][0],
        "Gradient Boost": gb.predict_proba(last_row[features])[:, 1][0],
        "Logistic Reg": lr.predict_proba(last_row[features])[:, 1][0]
    }
    
    return test, features, votes

# --- 4. MULTI-TIMEFRAME PREDICTIONS ---
def predict_long_term_trends(data):
    """
    Trains lightweight models for 1W, 1M, 3M, 6M horizons.
    """
    df = data.copy()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].fillna(0)
    
    # Simple Technicals
    df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
    df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=df["Close"], window=200).sma_indicator()
    
    features = ['RSI', 'SMA_50', 'SMA_200', 'Volume']
    features = [f for f in features if f in df.columns]
    
    horizons = {
        "1 Week": 5,
        "1 Month": 20,
        "3 Months": 60,
        "6 Months": 120
    }
    
    predictions = {}
    
    for name, shift_val in horizons.items():
        # Create specific target for this horizon
        temp_df = df.copy()
        temp_df['Target'] = (temp_df['Close'].shift(-shift_val) > temp_df['Close']).astype(int)
        temp_df.dropna(inplace=True)
        
        if len(temp_df) > 100:
            train_size = int(len(temp_df) * 0.9)
            train = temp_df.iloc[:train_size]
            
            # Fast model
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(train[features], train['Target'])
            
            # Predict on latest data
            last_row = temp_df.iloc[[-1]]
            prob = model.predict_proba(last_row[features])[:, 1][0]
            
            if prob > 0.6: predictions[name] = "Bullish 🟢"
            elif prob < 0.4: predictions[name] = "Bearish 🔴"
            else: predictions[name] = "Neutral ⚪"
        else:
            predictions[name] = "N/A"
            
    return predictions

# --- 5. MACRO, SECTORS & RISK ---
def get_macro_data():
    tickers = {
        "🇺🇸 10Y Yield": "^TNX",
        "💵 Dollar Index": "DX-Y.NYB",
        "🎢 VIX (Fear)": "^VIX",
        "🛢️ Oil": "CL=F",
        "🏆 Gold": "GC=F"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            d = yf.Ticker(sym).history(period="5d")
            if not d.empty:
                change = ((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
                data[name] = {"Price": d['Close'].iloc[-1], "Change": change}
        except: pass
    return data

def get_sector_heatmap():
    sectors = {
        "Tech": "XLK", "Finance": "XLF", "Health": "XLV", "Energy": "XLE",
        "Consumer": "XLY", "Staples": "XLP", "Real Estate": "XLRE"
    }
    performance = {}
    for name, tick in sectors.items():
        try:
            d = yf.Ticker(tick).history(period="2d")
            if len(d) >= 2:
                change = ((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
                performance[name] = change
            else: performance[name] = 0.0
        except: performance[name] = 0.0
    return performance

def get_correlation_matrix(portfolio_df):
    tickers = portfolio_df['Ticker'].unique().tolist()
    if len(tickers) < 2: return None
    closes = {}
    for t in tickers:
        d = get_data(t, period="6mo")
        if d is not None: closes[t] = d['Close']
    return pd.DataFrame(closes).corr()

# NEW: VALUE AT RISK (VaR)
def calculate_portfolio_var(portfolio_df, confidence_level=0.95):
    """Calculates historical Value at Risk (95% confidence)."""
    try:
        tickers = portfolio_df['Ticker'].unique().tolist()
        weights = portfolio_df['Shares'] * portfolio_df['Buy_Price_USD'] # Approx weights
        total_value = weights.sum()
        if total_value == 0: return 0.0
        weights = weights / total_value
        
        # Get historical returns
        data = pd.DataFrame()
        for t in tickers:
            df = get_data(t, period="1y")
            if df is not None:
                data[t] = df['Close'].pct_change()
        
        data.dropna(inplace=True)
        if data.empty: return 0.0
        
        # Portfolio historical returns
        data['Portfolio'] = data.dot(weights.values)
        
        # Calculate VaR (5th percentile of returns)
        var_percent = np.percentile(data['Portfolio'], (1 - confidence_level) * 100)
        var_value = abs(var_percent * total_value)
        
        return var_value
    except:
        return 0.0

# --- FIX START: ROBUST PORTFOLIO OPTIMIZER ---
def optimize_portfolio(tickers):
    """
    Calculates the best allocation weights (Sharpe Ratio) for a list of tickers.
    Fixes: shape mismatches when tickers fail to download.
    """
    # 1. Filter out empty tickers
    tickers = [t for t in tickers if t]
    if len(tickers) < 2: 
        return None
    
    # 2. Get Data
    data = pd.DataFrame()
    for t in tickers:
        df = get_data(t, period="1y")
        if df is not None:
            data[t] = df['Close']
            
    # 3. Clean Data - Drop columns with all NaNs or insufficient data
    # This prevents the shape mismatch error
    data = data.dropna(axis=1, how='all')
    
    if data.empty or data.shape[1] < 2: 
        return None
    
    # 4. Calculate Returns
    returns = data.pct_change().dropna()
    if returns.empty: 
        return None

    # 5. ALIGN TICKERS with actual data
    valid_tickers = returns.columns.tolist()
    num_assets = len(valid_tickers)
    
    if num_assets < 2:
        return None

    mean_returns = returns.mean() * 252 # Annualized
    cov_matrix = returns.cov() * 252
    
    # 6. Define Optimization Function
    def negative_sharpe(weights):
        p_ret = np.sum(weights * mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if p_vol == 0: return 0
        return -p_ret / p_vol
    
    # 7. Constraints (Weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 8. Bounds and Init Guess using CORRECT num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = [1.0 / num_assets] * num_assets
    
    # 9. Run Optimization
    try:
        result = minimize(
            negative_sharpe, 
            init_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        return dict(zip(valid_tickers, result.x))
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None
# --- FIX END ---

# --- 6. PORTFOLIO & ALERTS ---
PORTFOLIO_FILE = "portfolio.csv"
WATCHLIST_FILE = "watchlist.txt"

def get_exchange_rate(base_currency="GBP"):
    if base_currency == "USD": return 1.0
    try:
        pair = f"{base_currency}USD=X"
        d = yf.Ticker(pair).history(period="1d")
        if not d.empty: return 1.0 / d['Close'].iloc[-1]
    except: return 0.78
    return 1.0

def get_portfolio():
    if os.path.exists(PORTFOLIO_FILE): return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency", "Account"])

def execute_trade(ticker, price_usd, shares, action, currency="USD"):
    df = get_portfolio()
    if action == "BUY":
        new = pd.DataFrame([{"Ticker": ticker, "Buy_Price_USD": price_usd, "Shares": shares, "Date": pd.Timestamp.now(), "Status": "OPEN", "Currency": currency}])
        df = pd.concat([df, new], ignore_index=True)
    elif action == "SELL":
        rows = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
        if not rows.empty:
            idx = rows.index[0]
            df.at[idx, 'Status'] = 'CLOSED'
            df.at[idx, 'Sell_Price_USD'] = price_usd
            df.at[idx, 'Profit_USD'] = (price_usd - df.at[idx, 'Buy_Price_USD']) * df.at[idx, 'Shares']
    df.to_csv(PORTFOLIO_FILE, index=False)
    return "✅ Executed"

def get_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f: return [line.strip() for line in f.readlines() if line.strip()]
    return []

def add_to_watchlist(ticker):
    current = get_watchlist()
    if ticker not in current:
        with open(WATCHLIST_FILE, "a") as f: f.write(f"{ticker}\n")

def remove_from_watchlist(ticker):
    current = get_watchlist()
    if ticker in current:
        current.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            for t in current: f.write(f"{t}\n")

def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing.", []
    if "sk-" not in api_key: return "⚠️ Invalid Key.", []
    
    try:
        # 1. Get News safely
        stock = yf.Ticker(ticker)
        news_list = stock.news
        headlines = []
        
        # 2. Extract Titles (Bulletproof Method)
        if news_list:
            for n in news_list[:3]:
                # Method A: Standard 'title' key
                if 'title' in n:
                    headlines.append(n['title'])
                # Method B: Nested inside 'content' (New Yahoo Format)
                elif 'content' in n and 'title' in n['content']:
                    headlines.append(n['content']['title'])
        
        if not headlines: 
            return f"ℹ️ No news headlines found for {ticker}.", []

        # 3. Call AI
        client = OpenAI(api_key=api_key)
        prompt = f"Analyze these headlines for {ticker}: {headlines}. Summary & Sentiment?"
        
        res = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role":"user","content":prompt}]
        )
        return res.choices[0].message.content, headlines

    except Exception as e:
        return f"AI Error: {str(e)}", []

def send_telegram_alert(token, chat_id, msg):
    try:
        requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg})
        return "✅ Sent"
    except: return "❌ Failed"

def scan_market():
    tickers = get_watchlist()
    if not tickers: return pd.DataFrame()
    results = []
    for ticker in tickers:
        try:
            data = get_data(ticker, period="1y")
            if data is not None:
                processed, _, _ = train_consensus_model(data)
                if processed is not None:
                    last = processed.iloc[-1]
                    sig = "BUY 🟢" if last['Confidence'] > 0.6 else "SELL 🔴" if last['Confidence'] < 0.4 else "WAIT ⚪"
                    results.append({"Ticker": ticker, "Price": last['Close'], "Signal": sig, "Confidence": last['Confidence']})
        except: continue
    return pd.DataFrame(results).sort_values(by="Confidence", ascending=False) if results else pd.DataFrame()

def run_backtest(ticker, initial_capital=10000):
    data = get_data(ticker, period="2y")
    if data is None: return None
    processed, _, _ = train_consensus_model(data)
    if processed is None: return None
    
    balance = initial_capital
    shares = 0
    equity = []
    trades = []
    
    for date, row in processed.iterrows():
        price = row['Close']
        if row['Confidence'] > 0.65 and shares == 0:
            shares = int(balance / price)
            balance -= shares * price
            trades.append({"Date": date, "Action": "BUY", "Price": price})
        elif row['Confidence'] < 0.40 and shares > 0:
            balance += shares * price
            shares = 0
            trades.append({"Date": date, "Action": "SELL", "Price": price})
        equity.append(balance + (shares * price))
        
    processed['Equity'] = equity
    final = equity[-1]
    return processed, pd.DataFrame(trades), ((final-initial_capital)/initial_capital)*100
    # --- [NEW] WATCHDOG BACKGROUND SYSTEM ---
import time
import threading

def run_watchdog_scan(tele_token, tele_chat, threshold=0.7):
    """
    Scans the portfolio. If AI Confidence > threshold, sends a Telegram alert.
    """
    df = get_portfolio()
    if df.empty or 'Ticker' not in df.columns: return "❌ Portfolio Empty"
    
    # Filter for OPEN positions only
    active_tickers = df[df['Status'] == 'OPEN']['Ticker'].unique()
    alerts_sent = 0
    
    msg_buffer = "🚨 *Watchdog Report*\n"
    
    for ticker in active_tickers:
        try:
            # Run AI Model
            data = get_data(ticker, period="6mo", interval="1d")
            if data is not None:
                processed, _, _ = train_consensus_model(data)
                if processed is not None:
                    last = processed.iloc[-1]
                    conf = last['Confidence']
                    
                    # ALERT LOGIC:
                    # 1. Buy Signal with High Confidence
                    if conf > threshold:
                        msg_buffer += f"\n🟢 *{ticker}*: BUY (Conf: {conf*100:.0f}%)"
                        alerts_sent += 1
                    # 2. Sell Signal (Panic Button)
                    elif conf < (1.0 - threshold):
                        msg_buffer += f"\n🔴 *{ticker}*: SELL/DUMP (Conf: {(1-conf)*100:.0f}%)"
                        alerts_sent += 1
        except: continue
        
    if alerts_sent > 0:
        send_telegram_alert(tele_token, tele_chat, msg_buffer)
        return f"✅ Sent {alerts_sent} alerts to Telegram!"
    else:
        return "🐶 Watchdog scanned. No significant moves detected."
