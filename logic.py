import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import requests
import os

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
                exch = quote.get('exchange')
                if symbol and name:
                    results[f"{name} ({symbol}) - {exch}"] = symbol
        return results
    except:
        return {}

# --- 2. DATA ACQUISITION ---
def get_data(ticker, period="2y", interval="1d"):
    try:
        # Day trading requires fetching data differently
        if interval in ['15m', '30m', '60m', '1h']:
            period = "1mo"
        
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def get_market_status():
    spy = get_data("SPY", period="1y", interval="1d")
    if spy is None: return "Unknown"
    spy['SMA_200'] = ta.sma(spy['Close'], length=200)
    current = spy['Close'].iloc[-1]
    sma = spy['SMA_200'].iloc[-1]
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

# --- 3. ML & TECHNICALS (FIXED FOR SGLN/ETFs) ---
def train_model(data):
    df = data.copy()
    
    # FIX: Fill missing volume with 0 to prevent dropna() from deleting everything
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    
    # Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # FIX: Safety check before dropping
    if len(df) < 50: return None, []
    
    df.dropna(inplace=True)
    
    # FIX: Final Empty Check
    if df.empty: return None, []
    
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'Volume']
    features = [f for f in features if f in df.columns] # Only use existing columns
    
    train_size = int(len(df) * 0.90)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    if len(train) < 10: return None, []
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(train[features], train['Target'])
    
    probs = model.predict_proba(test[features])[:, 1]
    test = test.copy()
    test['Confidence'] = probs
    
    return test, features

# --- 4. CALCULATORS ---
def calculate_position_size(account_size, risk_pct, current_price, atr):
    if atr == 0: return 0, 0
    risk_amount = account_size * (risk_pct / 100)
    stop_loss_dist = 1.5 * atr
    shares = int(risk_amount / stop_loss_dist)
    return shares, stop_loss_dist

# --- 5. AI NEWS ANALYSIS (FIXED) ---
def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing in Settings.", []
    
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news[:3]
        headlines = []
        for n in raw_news:
            if 'title' in n: headlines.append(n['title'])
            elif 'content' in n and 'title' in n['content']: headlines.append(n['content']['title'])
        
        if not headlines: return "No recent news found.", []

        client = OpenAI(api_key=api_key)
        prompt = f"Analyze these headlines for {ticker}: {headlines}. Return: Sentiment (Bullish/Bearish) & 1 sentence summary."
        
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
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        return []

def add_to_watchlist(ticker):
    current = get_watchlist()
    if ticker not in current:
        with open(WATCHLIST_FILE, "a") as f:
            f.write(f"{ticker}\n")
        return f"✅ {ticker} added."
    return f"⚠️ {ticker} exists."

def remove_from_watchlist(ticker):
    current = get_watchlist()
    if ticker in current:
        current.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            for t in current: f.write(f"{t}\n")
        return f"🗑️ {ticker} removed."
    return "⚠️ Not found."
