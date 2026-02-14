import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import requests

# --- 1. SMART TICKER SEARCH (New Feature) ---
def search_ticker(query):
    """
    Searches for a stock ticker by company name using Yahoo Finance API.
    Supports US (AAPL) and London (RR.L) stocks.
    """
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
                    # Create a readable label: "Apple Inc. (AAPL) - NASDAQ"
                    label = f"{name} ({symbol}) - {exch}"
                    results[label] = symbol
        return results
    except Exception as e:
        print(f"Search Error: {e}")
        return {}

# --- 2. DATA ACQUISITION ---
def get_data(ticker, period="2y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def get_market_status():
    spy = get_data("SPY", period="1y")
    if spy is None: return "Unknown"
    spy['SMA_200'] = ta.sma(spy['Close'], length=200)
    current_price = spy['Close'].iloc[-1]
    sma_200 = spy['SMA_200'].iloc[-1]
    return "Bullish 🐂" if current_price > sma_200 else "Bearish 🐻"

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Market Cap": info.get('marketCap', 'N/A'),
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "Profit Margin": info.get('profitMargins', 'N/A'),
            "52w High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "Sector": info.get('sector', 'Unknown')
        }
    except:
        return {}

# --- 3. ML & TECHNICALS ---
def train_model(data):
    df = data.copy()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    features = ['RSI', 'SMA_50', 'ATR', 'Volume']
    train_size = int(len(df) * 0.90)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(train[features], train['Target'])
    
    probs = model.predict_proba(test[features])[:, 1]
    test = test.copy()
    test['Confidence'] = probs
    
    return test, features

# --- 4. CALCULATORS & ALERTS ---
def calculate_position_size(account_size, risk_pct, current_price, atr):
    if atr == 0: return 0
    risk_amount = account_size * (risk_pct / 100)
    stop_loss_dist = 2 * atr
    return int(risk_amount / stop_loss_dist)

def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing in Settings.", []
    try:
        news = yf.Ticker(ticker).news[:3]
        headlines = [n['title'] for n in news]
        client = OpenAI(api_key=api_key)
        prompt = f"Analyze these headlines for {ticker}: {headlines}. Summary (1 sent) + Sentiment (Bullish/Bearish)."
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, headlines
    except Exception as e:
        return f"AI Error: {e}", []

def send_telegram_alert(token, chat_id, ticker, signal, price):
    if not token or not chat_id: return "⚠️ Telegram details missing in Settings."
    msg = f"🚀 *{ticker} Alert*\nSignal: {signal}\nPrice: ${price:.2f}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.get(url, params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
        return "✅ Alert Sent!"
    except:
        return "❌ Failed to send."