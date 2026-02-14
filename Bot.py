import logic
import os

# --- LOAD SECRETS FROM ENVIRONMENT VARIABLES ---
# GitHub will inject these into the script securely
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_KEY = os.getenv("OPENAI_KEY")

# WATCHLIST (You can edit this file directly on GitHub to update it)
WATCHLIST_FILE = "watchlist.txt"

def get_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return ["AAPL", "TSLA", "BTC-USD"] # Default backup

def run_once():
    watchlist = get_watchlist()
    print(f"🤖 GitHub Bot Starting... Scanning: {watchlist}")
    
    for ticker in watchlist:
        try:
            # Check Daily or Hourly data
            data = logic.get_data(ticker, interval="1h")
            
            if data is not None:
                processed, _ = logic.train_model(data)
                last_row = processed.iloc[-1]
                conf = last_row['Confidence']
                price = last_row['Close']
                
                # Signal Logic
                signal = "HOLD"
                if conf > 0.60: signal = "BUY 🟢"
                elif conf < 0.40: signal = "SELL 🔴"
                
                if "BUY" in signal or "SELL" in signal:
                    print(f"🚀 SIGNAL: {ticker} -> {signal}")
                    logic.send_telegram_alert(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ticker, signal, price, "1h")
        except Exception as e:
            print(f"❌ Error on {ticker}: {e}")

if __name__ == "__main__":
    run_once()