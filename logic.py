import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
import requests
import os

# --- 1. DATA & SEARCH ---
def search_ticker(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                symbol = quote.get('symbol')
                name = quote.get('shortname') or quote.get('longname')
                exch = quote.get('exchange')
                if symbol and name:
                    results[f"{name} ({symbol}) - {exch}"] = symbol
        return results
    except:
        return {}

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

# --- 2. MULTI-MODEL CONSENSUS AI ---
def train_consensus_model(data):
    """
    Trains 3 models (Random Forest, XGBoost, LogReg) and takes a vote.
    """
    df = data.copy()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].fillna(0)
    
    # Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    if len(df) < 50: return None, [], {}
    df.dropna(inplace=True)
    if df.empty: return None, [], {}
    
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'Volume']
    features = [f for f in features if f in df.columns]
    
    train_size = int(len(df) * 0.90)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # 1. Define the 3 Models
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    lr = LogisticRegression(solver='liblinear')
    
    # 2. Create Voting Consensus (Soft Vote = Average Probability)
    ensemble = VotingClassifier(estimators=[('RF', rf), ('GB', gb), ('LR', lr)], voting='soft')
    ensemble.fit(train[features], train['Target'])
    
    # 3. Predict
    probs = ensemble.predict_proba(test[features])[:, 1]
    test = test.copy()
    test['Confidence'] = probs
    
    # Get individual votes for the "Details" view
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

# --- 3. MACRO & SECTORS ---
def get_macro_data():
    """Fetches key economic indicators."""
    tickers = {
        "🇺🇸 10Y Yield": "^TNX",
        "💵 Dollar Index": "DX-Y.NYB",
        "🎢 VIX (Fear)": "^VIX",
        "🛢️ Oil": "CL=F",
        "🏆 Gold": "GC=F",
        "🇬🇧 FTSE 100": "^FTSE"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            d = yf.Ticker(sym).history(period="5d")
            if not d.empty:
                change = ((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
                data[name] = {"Price": d['Close'].iloc[-1], "Change": change}
        except:
            pass
    return data

def get_sector_heatmap():
    """Fetches performance of major US Sector ETFs."""
    sectors = {
        "Tech": "XLK", "Finance": "XLF", "Health": "XLV", "Energy": "XLE",
        "Consumer": "XLY", "Staples": "XLP", "Industrial": "XLI", "Comms": "XLC",
        "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"
    }
    performance = {}
    for name, tick in sectors.items():
        try:
            d = yf.Ticker(tick).history(period="2d")
            if len(d) >= 2:
                change = ((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
                performance[name] = change
            else:
                performance[name] = 0.0
        except:
            performance[name] = 0.0
    return performance

# --- 4. MULTI-CURRENCY PORTFOLIO ---
PORTFOLIO_FILE = "portfolio.csv"

def get_exchange_rate(base_currency="GBP"):
    if base_currency == "USD": return 1.0
    try:
        pair = f"{base_currency}USD=X" 
        d = yf.Ticker(pair).history(period="1d")
        if not d.empty:
            rate = d['Close'].iloc[-1] 
            return 1.0 / rate 
    except:
        return 0.78 
    return 1.0

def get_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency"])

def execute_trade(ticker, price_usd, shares, action, currency="USD"):
    df = get_portfolio()
    if action == "BUY":
        new = pd.DataFrame([{
            "Ticker": ticker, "Buy_Price_USD": price_usd, "Shares": shares, 
            "Date": pd.Timestamp.now(), "Status": "OPEN", "Currency": currency
        }])
        df = pd.concat([df, new], ignore_index=True)
    elif action == "SELL":
        rows = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
        if not rows.empty:
            idx = rows.index[0]
            df.at[idx, 'Status'] = 'CLOSED'
            df.at[idx, 'Sell_Price_USD'] = price_usd
            pnl = (price_usd - df.at[idx, 'Buy_Price_USD']) * df.at[idx, 'Shares']
            df.at[idx, 'Profit_USD'] = pnl
            
    df.to_csv(PORTFOLIO_FILE, index=False)
    return "✅ Executed"

# --- 5. ALERTS & AI ---
def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing.", []
    try:
        headlines = [n['title'] for n in yf.Ticker(ticker).news[:3]]
        client = OpenAI(api_key=api_key)
        prompt = f"Analyze {ticker} based on: {headlines}. Max 50 words. Sentiment?"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content, headlines
    except: return "AI Error", []

def send_telegram_alert(token, chat_id, msg):
    try:
        requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg})
        return "✅ Sent"
    except: return "❌ Failed"

# --- 6. WATCHDOG & WATCHLIST ---
WATCHLIST_FILE = "watchlist.txt"

def get_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

def add_to_watchlist(ticker):
    current = get_watchlist()
    if ticker not in current:
        with open(WATCHLIST_FILE, "a") as f:
            f.write(f"{ticker}\n")

def remove_from_watchlist(ticker):
    current = get_watchlist()
    if ticker in current:
        current.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            for t in current: f.write(f"{t}\n")

# --- 7. MARKET SCANNER ---
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
                    last_row = processed.iloc[-1]
                    conf = last_row['Confidence']
                    signal = "BUY 🟢" if conf > 0.6 else "SELL 🔴" if conf < 0.4 else "WAIT ⚪"
                    results.append({
                        "Ticker": ticker,
                        "Price": last_row['Close'],
                        "Signal": signal,
                        "Confidence": conf,
                        "RSI": last_row['RSI']
                    })
        except: continue
            
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values(by="Confidence", ascending=False)
    return df

# --- 8. BACKTESTING ENGINE ---
def run_backtest(ticker, initial_capital=10000):
    """
    Simulates trading over the last 1-2 years using the Consensus Model.
    """
    data = get_data(ticker, period="2y") # Fetch max history for test
    if data is None: return None
    
    # Train/Predict on whole dataset to simulate "Past Performance"
    # Note: In a rigorous academic backtest, you would use a Walk-Forward method.
    # For a dashboard tool, we use the standard model to show theoretical fit.
    processed, _, _ = train_consensus_model(data)
    
    if processed is None: return None
    
    # Simulation Loop
    balance = initial_capital
    shares = 0
    equity_curve = []
    trades = []
    
    for date, row in processed.iterrows():
        price = row['Close']
        conf = row['Confidence']
        
        # BUY LOGIC (High Confidence & No Position)
        if conf > 0.65 and shares == 0:
            shares = int(balance / price)
            balance -= shares * price
            trades.append({"Date": date, "Action": "BUY", "Price": price})
            
        # SELL LOGIC (Low Confidence & Have Position)
        elif conf < 0.40 and shares > 0:
            balance += shares * price
            shares = 0
            trades.append({"Date": date, "Action": "SELL", "Price": price})
            
        # Record Daily Equity
        current_equity = balance + (shares * price)
        equity_curve.append(current_equity)
        
    processed['Equity'] = equity_curve
    
    # Calculate Stats
    final_value = equity_curve[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    return processed, pd.DataFrame(trades), total_return
