import logic
import os
import requests

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

def send_telegram_alert(token, chat_id, ticker, signal, price, interval):
    """Send a trading signal alert via Telegram."""
    if not token or not chat_id:
        print(f"⚠️ Telegram not configured. Signal: {ticker} -> {signal} @ ${price:.2f}")
        return
    try:
        message = (
            f"📊 *StockMind-AI Alert*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Signal:* {signal}\n"
            f"*Price:* ${price:.2f}\n"
            f"*Interval:* {interval}"
        )
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def run_once():
    watchlist = get_watchlist()
    print(f"🤖 GitHub Bot Starting... Scanning: {watchlist}")

    for ticker in watchlist:
        try:
            # Get predictions using the multi-timeframe system
            result = logic.get_multi_timeframe_predictions(ticker)

            if result and 'predictions' in result and result['predictions']:
                # Use the daily prediction for bot alerts
                daily_pred = result['predictions'].get('daily', {})
                signal_str = daily_pred.get('signal', 'HOLD')
                confidence = daily_pred.get('confidence', 0)

                # Get current price
                data = logic.get_data(ticker, period="5d", interval="1d")
                price = data['Close'].iloc[-1] if data is not None else 0

                # Format signal
                if signal_str == 'BUY':
                    signal = f"BUY 🟢 ({confidence*100:.0f}%)"
                elif signal_str == 'SELL':
                    signal = f"SELL 🔴 ({confidence*100:.0f}%)"
                else:
                    signal = f"HOLD ⚪ ({confidence*100:.0f}%)"

                if signal_str in ('BUY', 'SELL'):
                    print(f"🚀 SIGNAL: {ticker} -> {signal}")
                    send_telegram_alert(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ticker, signal, price, "daily")
                else:
                    print(f"⏸️ {ticker} -> {signal}")
            else:
                print(f"⚠️ No predictions available for {ticker}")
        except Exception as e:
            print(f"❌ Error on {ticker}: {e}")

if __name__ == "__main__":
    run_once()