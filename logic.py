import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
import os
import pdfplumber
import io
import re
from scipy.optimize import minimize
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# --- 1. DATA & CACHING ---
def search_ticker(query, region="All"):
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
        allowed = filters.get(region, [])
        if 'quotes' in data:
            for q in data['quotes']:
                s, n, e = q.get('symbol'), q.get('shortname') or q.get('longname'), q.get('exchange')
                if s and n and e and (region == "All" or e in allowed):
                    results[f"{n} ({s}) - {e}"] = s
        return results
    except: return {}

@st.cache_data(ttl=300)
def get_data(ticker, period="2y", interval="1d"):
    try:
        if interval in ['15m', '30m', '60m', '1h']: period = "1mo"
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        return data
    except: return None

def add_technical_overlays(df):
    df = df.copy()
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
    return df

# --- 2. TRADING 212 PDF PARSER (ROBUST) ---
def process_t212_pdf(uploaded_file):
    """
    Robust parser for Trading 212 'Confirmation of Holdings' PDFs.
    """
    def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip()
    def _to_float(num_str):
        s = _norm(num_str).replace(":", ".").replace(",", "")
        s = re.sub(r"[^0-9\.\-]", "", s)
        try: return float(s)
        except: return None
    def _infer_ticker(instrument):
        name = _norm(instrument).upper()
        m = re.search(r"\(([A-Z0-9\.\-]{1,15})\)", name)
        return m.group(1) if m else name.split()[0]

    extracted = []
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2: continue
                    
                    # Find header row
                    header_idx = None
                    for i, row in enumerate(table[:10]):
                        joined = " | ".join(_norm(x).upper() for x in row)
                        if "INSTRUMENT" in joined and "QUANTITY" in joined:
                            header_idx = i
                            break
                    if header_idx is None: continue

                    header = [_norm(x).upper() for x in table[header_idx]]
                    try:
                        c_inst = [i for i, h in enumerate(header) if "INSTRUMENT" in h][0]
                        c_qty = [i for i, h in enumerate(header) if "QUANTITY" in h][0]
                        c_price = [i for i, h in enumerate(header) if "PRICE" in h][0]
                    except: continue

                    for row in table[header_idx + 1:]:
                        if not row or len(row) <= max(c_inst, c_qty, c_price): continue
                        
                        inst = _norm(row[c_inst])
                        if not inst or inst.upper() == "TOTAL": continue
                        
                        qty = _to_float(row[c_qty])
                        if not qty: continue
                        
                        price_txt = _norm(row[c_price]).upper()
                        price_val = _to_float(price_txt)
                        
                        currency = "USD"
                        if "GBX" in price_txt: 
                            currency = "GBP"
                            if price_val: price_val /= 100
                        elif "GBP" in price_txt: currency = "GBP"
                        elif "EUR" in price_txt: currency = "EUR"
                        
                        ticker = _infer_ticker(inst)
                        if ticker:
                            extracted.append({
                                "Ticker": ticker,
                                "Buy_Price_USD": price_val if price_val else 0.0,
                                "Shares": qty,
                                "Date": pd.Timestamp.now(),
                                "Status": "OPEN",
                                "Currency": currency
                            })
        return pd.DataFrame(extracted)
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

# --- 3. PORTFOLIO MANAGEMENT ---
PORTFOLIO_FILE = "portfolio.csv"

def get_portfolio():
    if os.path.exists(PORTFOLIO_FILE): return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame(columns=["Ticker", "Buy_Price_USD", "Shares", "Date", "Status", "Currency"])

def sync_portfolio_with_df(new_data_df):
    if new_data_df is not None and not new_data_df.empty:
        new_data_df.to_csv(PORTFOLIO_FILE, index=False)
        return True
    return False

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
    df.to_csv(PORTFOLIO_FILE, index=False)
    return "✅ Executed"

def get_exchange_rate(base_currency="GBP"):
    if base_currency == "USD": return 1.0
    try:
        d = yf.Ticker(f"{base_currency}USD=X").history(period="1d")
        return 1.0 / d['Close'].iloc[-1] if not d.empty else 0.78
    except: return 0.78

# --- 4. ROBUST AI (RESTORED) ---
def train_consensus_model(data):
    df = data.copy()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].fillna(0)
    
    # Indicators
    df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
    df['SMA_20'] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df['ATR'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    
    # Lagged Features
    df['Lag_1'] = df['Close'].pct_change().shift(1)
    df['Lag_2'] = df['Close'].pct_change().shift(2)
    df['Lag_5'] = df['Close'].pct_change().shift(5)

    # Noise Target (>0.2% move)
    df['Target'] = ((df['Close'].shift(-1) / df['Close'] - 1) > 0.002).astype(int)
    
    df.dropna(inplace=True)
    if len(df) < 50: return None, [], {}
    
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'Volume', 'Lag_1', 'Lag_2', 'Lag_5']
    features = [f for f in features if f in df.columns]
    
    train_size = int(len(df) * 0.90)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(train[features], train['Target'])
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(train[features], train['Target'])
    lr = LogisticRegression(solver='liblinear').fit(train[features], train['Target'])
    
    test = test.copy()
    test['Confidence'] = (rf.predict_proba(test[features])[:,1] + gb.predict_proba(test[features])[:,1] + lr.predict_proba(test[features])[:,1]) / 3
    
    last_row = test.iloc[[-1]]
    votes = {
        "Random Forest": rf.predict_proba(last_row[features])[:,1][0],
        "Gradient Boost": gb.predict_proba(last_row[features])[:,1][0],
        "Logistic Reg": lr.predict_proba(last_row[features])[:,1][0]
    }
    return test, features, votes

# --- 5. MISSING FUNCTIONS RESTORED ---
def predict_long_term_trends(data):
    """Restored Forecast Function"""
    df = data.copy()
    horizons = {"1 Week": 5, "1 Month": 20, "3 Months": 60, "6 Months": 120}
    preds = {}
    
    df['RSI'] = RSIIndicator(close=df["Close"]).rsi()
    df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    
    for k, v in horizons.items():
        # Simple trend logic if not enough data for ML
        if len(df) > v:
            change = (df['Close'].iloc[-1] / df['Close'].iloc[-v] - 1)
            preds[k] = "Bullish 🟢" if change > 0.02 else "Bearish 🔴" if change < -0.02 else "Neutral ⚪"
        else:
            preds[k] = "N/A"
    return preds

def optimize_portfolio(tickers):
    if len(tickers) < 2: return None
    data = pd.DataFrame()
    for t in tickers:
        df = get_data(t, period="1y")
        if df is not None: data[t] = df['Close']
    if data.empty: return None
    
    returns = data.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    
    def neg_sharpe(w):
        p_ret = np.sum(w * mean_ret)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        return -p_ret / p_vol
        
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init = [1/len(tickers)] * len(tickers)
    
    res = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
    return dict(zip(tickers, res.x))

def calculate_portfolio_var(portfolio_df, confidence=0.95):
    try:
        tickers = portfolio_df['Ticker'].unique().tolist()
        total_val = (portfolio_df['Shares'] * portfolio_df['Buy_Price_USD']).sum()
        
        data = pd.DataFrame()
        for t in tickers:
            d = get_data(t, period="1y")
            if d is not None: data[t] = d['Close'].pct_change()
        
        data.dropna(inplace=True)
        weights = (portfolio_df.groupby('Ticker')['Shares'].sum() * portfolio_df.groupby('Ticker')['Buy_Price_USD'].mean()) / total_val
        # Align weights with data columns
        weights = weights[data.columns]
        
        port_ret = data.dot(weights.values)
        var_pct = np.percentile(port_ret, (1-confidence)*100)
        return abs(var_percent * total_val)
    except: return 0.0

def get_correlation_matrix(portfolio_df):
    try:
        tickers = portfolio_df['Ticker'].unique().tolist()
        data = pd.DataFrame({t: get_data(t, period="6mo")['Close'] for t in tickers})
        return data.corr()
    except: return None

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Market Cap": info.get("marketCap", 0), "P/E Ratio": info.get("trailingPE", "N/A"),
            "Beta": info.get("beta", "N/A"), "Dividend Yield": info.get("dividendYield", 0),
            "Sector": info.get("sector", "Unknown"), "Industry": info.get("industry", "Unknown")
        }
    except: return None

def get_macro_data():
    tickers = {"🇺🇸 10Y Yield": "^TNX", "💵 DXY": "DX-Y.NYB", "🎢 VIX": "^VIX", "🛢️ Oil": "CL=F", "🏆 Gold": "GC=F"}
    data = {}
    for n, s in tickers.items():
        try:
            d = yf.Ticker(s).history(period="2d")
            chg = ((d['Close'].iloc[-1]/d['Close'].iloc[-2])-1)*100
            data[n] = {"Price": d['Close'].iloc[-1], "Change": chg}
        except: pass
    return data

def get_sector_heatmap():
    s = {"Tech": "XLK", "Finance": "XLF", "Energy": "XLE", "Health": "XLV", "Consumer": "XLY"}
    return {k: yf.Ticker(v).history(period="2d")['Close'].pct_change().iloc[-1]*100 for k, v in s.items()}

# --- 6. EXTRAS ---
def get_watchlist():
    if os.path.exists("watchlist.txt"):
        with open("watchlist.txt", "r") as f: return [l.strip() for l in f.readlines() if l.strip()]
    return []

def add_to_watchlist(t):
    if t not in get_watchlist():
        with open("watchlist.txt", "a") as f: f.write(f"{t}\n")

def remove_from_watchlist(t):
    wl = get_watchlist()
    if t in wl:
        wl.remove(t)
        with open("watchlist.txt", "w") as f: [f.write(f"{x}\n") for x in wl]

def get_ai_analysis(ticker, api_key):
    if not api_key: return "⚠️ API Key Missing.", []
    try:
        headlines = [n['title'] for n in yf.Ticker(ticker).news[:3]]
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":f"Analyze {ticker} based on {headlines}. Sentiment?"}])
        return res.choices[0].message.content, headlines
    except: return "AI Error", []

def send_telegram_alert(token, cid, msg):
    try: requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": cid, "text": msg})
    except: pass

def scan_market():
    wl = get_watchlist()
    res = []
    for t in wl:
        d = get_data(t)
        if d is not None:
            p, _, _ = train_consensus_model(d)
            if p is not None:
                last = p.iloc[-1]
                sig = "BUY 🟢" if last['Confidence'] > 0.6 else "SELL 🔴" if last['Confidence'] < 0.4 else "WAIT ⚪"
                res.append({"Ticker": t, "Price": last['Close'], "Signal": sig, "Confidence": last['Confidence']})
    return pd.DataFrame(res)

def run_backtest(ticker, capital):
    data = get_data(ticker, period="2y")
    if data is None: return None, None, 0
    processed, _, _ = train_consensus_model(data)
    if processed is None: return None, None, 0
    
    balance = capital
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
    final_ret = ((equity[-1] - capital) / capital) * 100
    return processed, pd.DataFrame(trades), final_ret
