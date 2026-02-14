import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import requests

# --- 1. SMART TICKER SEARCH ---
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
                if symbol and name:
                    results[f"{name} ({symbol})"] = symbol
        return results
    except:
        return {}

# --- 2. DATA ACQUISITION (UPDATED FOR DAY TRADING) ---
def get_data(ticker, period="2y", interval="1d"):
    """
    Fetches data. For Day Trading (intraday), period must be shorter (e.g., '5d' or '1mo').
    """
    try:
        # Day trading requires fetching data differently to get recent intraday moves
        if interval in ['15m', '30m', '60m', '1h']:
            period = "1mo" # Max buffer for intraday data
        
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"Data Error: {e}")
        return None

def get_market_status():
    spy = get_data("SPY", period="6mo", interval="1d")
    if spy is None: return "Unknown"
    spy['SMA_50'] = ta.sma(spy['Close'], length=50)
    current = spy['Close'].iloc[-1]
    sma = spy['SMA_50'].iloc[-1]
    return "Bullish 🐂" if current > sma else "Bearish 🐻"

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Market Cap": info.get('marketCap', 'N/A'),
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "Beta": info.get('beta', 'N/A'),
            "Vol (10Day)": info.get('averageVolume10days', 'N/A')
        }
    except:
        return {}

# --- 3. ML & TECHNICALS (UPDATED STRATEGY) ---
def train_model(data):
    df = data.copy()
    
    # Standard Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Target: 1 if Price Rises in the NEXT candle
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'Volume']
    
    # Train
    train_size = int(len(df) * 0.90)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(train[features], train['Target'])
    
    # Predict
    probs = model.predict_proba(test[features])[:, 1]
    test = test.copy()
    test['Confidence'] = probs
    
    return test, features

# --- 4. CALCULATORS ---
def calculate_position_size(account_size, risk_pct, current_price, atr):
    if atr == 0: return 0
    risk_amount = account_size * (risk_pct / 100)
    stop_loss_dist = 1.5 * atr # Tighter stop for Day Trading
    shares = int(risk_amount / stop_loss_dist)
    return shares, stop_loss_dist

# --- 5. FIXED AI NEWS ANALYSIS ---
def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing in Settings.", []
    
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news[:3]
        
        # FIX: Robust way to handle different Yahoo news formats
        headlines = []
        for n in raw_news:
            # Check if 'title' is direct or inside 'content'
            if 'title' in n:
                headlines.append(n['title'])
            elif 'content' in n and 'title' in n['content']:
                headlines.append(n['content']['title'])
        
        if not headlines:
            return "No recent news found to analyze.", []

        client = OpenAI(api_key=api_key)
        prompt = f"Analyze these headlines for {ticker} (Day Trading Context): {headlines}. Return: Sentiment (Bullish/Bearish) & 1 sentence summary."
        
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, headlines
    
    except Exception as e:
        return f"AI Error: {e}", []

def send_telegram_alert(token, chat_id, ticker, signal, price, timeframe):
    if not token or not chat_id: return "⚠️ Telegram details missing."
    msg = f"🚀 *{ticker} ({timeframe}) Alert*\nSignal: {signal}\nPrice: ${price:.2f}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.get(url, params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
        return "✅ Alert Sent!"
    except:
        return "❌ Failed."
        
        # --- 6. WATCHLIST MANAGER ---
WATCHLIST_FILE = "watchlist.txt"

def get_watchlist():
    """Reads the watchlist from a text file."""
    try:
        with open(WATCHLIST_FILE, "r") as f:
            # Read lines, strip whitespace, remove empty lines
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        return []

def add_to_watchlist(ticker):
    """Adds a ticker to the file if not exists."""
    current = get_watchlist()
    if ticker not in current:
        with open(WATCHLIST_FILE, "a") as f:
            f.write(f"{ticker}\n")
        return f"✅ {ticker} added to Watchdog."
    return f"⚠️ {ticker} is already in Watchdog."

def remove_from_watchlist(ticker):
    """Removes a ticker from the file."""
    current = get_watchlist()
    if ticker in current:
        current.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            for t in current:
                f.write(f"{t}\n")
        return f"🗑️ {ticker} removed."
    return "⚠️ Ticker not found."

