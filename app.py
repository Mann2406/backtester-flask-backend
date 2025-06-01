from flask import Flask, jsonify, send_file, request
from flask_cors import CORS  # Added for CORS support
import pandas as pd
import pandas_ta as ta
import requests
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=5000):
    """
    Fetch OHLC data from Binance API.
    Returns a pandas DataFrame with timestamp, open, high, low, close, volume.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    """
    Calculate MACD, RSI, and EMA indicators.
    Returns DataFrame with additional indicator columns.
    """
    df = df.copy()
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_ema10"] = ta.ema(df["macd_line"], length=10)
    df["macd_ema20"] = ta.ema(df["macd_line"], length=20)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema21"] = ta.ema(df["close"], length=21)
    return df

def macd_strategy(df):
    """
    MACD-based strategy: Buy when macd_ema10 crosses above macd_ema20, sell when it crosses below.
    Returns DataFrame with buy/sell signals.
    """
    df = df.copy()
    df["signal"] = 0
    df["macd_crossover"] = (df["macd_ema10"] > df["macd_ema20"]) & (df["macd_ema10"].shift(1) <= df["macd_ema20"].shift(1))
    df["macd_crossunder"] = (df["macd_ema10"] < df["macd_ema20"]) & (df["macd_ema10"].shift(1) >= df["macd_ema20"].shift(1))
    df.loc[df["macd_crossover"], "signal"] = 1
    df.loc[df["macd_crossunder"], "signal"] = -1
    df["strategy"] = "MACD"
    return df

def rsi_ema_strategy(df):
    """
    RSI-EMA strategy: Buy when RSI crosses above 30 and close > EMA21, exit when RSI crosses below 70 or close < EMA21.
    Returns DataFrame with buy/sell signals.
    """
    df = df.copy()
    df["signal"] = 0
    # Buy: RSI crosses above 30 and close > EMA21
    df["rsi_above_30"] = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    df["price_above_ema21"] = df["close"] > df["ema21"]
    df["buy_signal"] = df["rsi_above_30"] & df["price_above_ema21"]
    # Exit: RSI crosses below 70 or close < EMA21
    df["rsi_below_70"] = (df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)
    df["price_below_ema21"] = df["close"] < df["ema21"]
    df["exit_signal"] = df["rsi_below_70"] | df["price_below_ema21"]
    df.loc[df["buy_signal"], "signal"] = 1  # Buy
    df.loc[df["exit_signal"], "signal"] = -1  # Sell
    df["strategy"] = "RSI-EMA"
    # Debug: Log RSI and EMA values for the first few rows
    logging.info(f"RSI-EMA Debug: First 5 rows - RSI: {df['rsi'].head().to_dict()}, EMA21: {df['ema21'].head().to_dict()}, Close: {df['close'].head().to_dict()}")
    logging.info(f"RSI-EMA Signals: {df[['rsi_above_30', 'buy_signal', 'exit_signal']].sum().to_dict()}")
    return df

def simulate_trades(df, strategy_name):
    """
    Simulate trades based on strategy signals.
    Returns a trades DataFrame with entry/exit details and P&L.
    """
    trades = []
    position = None
    for i in range(1, len(df)):
        if df["signal"].iloc[i] == 1 and position is None:
            position = {
                "Entry Time": df["timestamp"].iloc[i],
                "Entry Price": df["close"].iloc[i],
                "Strategy": strategy_name
            }
        elif df["signal"].iloc[i] == -1 and position is not None:
            trade = {
                "Entry Time": position["Entry Time"],
                "Entry Price": position["Entry Price"],
                "Exit Time": df["timestamp"].iloc[i],
                "Exit Price": df["close"].iloc[i],
                "Strategy": strategy_name,
                "PnL": (df["close"].iloc[i] - position["Entry Price"]) * 1,
                "Status": "Closed"
            }
            trades.append(trade)
            position = None
    if position is not None:
        trade = {
            "Entry Time": position["Entry Time"],
            "Entry Price": position["Entry Price"],
            "Exit Time": df["timestamp"].iloc[-1],
            "Exit Price": df["close"].iloc[-1],
            "Strategy": strategy_name,
            "PnL": (df["close"].iloc[-1] - position["Entry Price"]) * 1,
            "Status": "Open"
        }
        trades.append(trade)
    return pd.DataFrame(trades)

def run_backtest(symbol="BTCUSDT", interval="1m", limit=5000):
    """
    Run backtest for MACD and RSI-EMA strategies.
    Returns trades DataFrame and stats.
    """
    df = fetch_binance_data(symbol, interval, limit)
    if df is None:
        return None, None
    
    df = calculate_indicators(df)
    macd_df = macd_strategy(df)
    rsi_ema_df = rsi_ema_strategy(df)
    
    macd_trades = simulate_trades(macd_df, "MACD")
    rsi_ema_trades = simulate_trades(rsi_ema_df, "RSI-EMA")
    
    trades_df = pd.concat([macd_trades, rsi_ema_trades], ignore_index=True)
    csv_path = "trades_output.csv"
    trades_df.to_csv(csv_path, index=False)
    logging.info(f"Trades saved to {csv_path}")
    
    stats = {
        "total_trades": len(trades_df),
        "total_pnl": trades_df["PnL"].sum() if not trades_df.empty else 0,
        "win_rate": len(trades_df[trades_df["PnL"] > 0]) / len(trades_df) if not trades_df.empty else 0
    }
    return trades_df, stats

@app.route("/backtest", methods=["GET"])
def backtest():
    """
    Run backtest and return trades and stats as JSON.
    Supports query params: symbol, interval, limit, strategy.
    """
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "1m")
    limit = int(request.args.get("limit", 5000))
    strategy = request.args.get("strategy", None)  # Optional: "MACD" or "RSI-EMA"
    
    trades_df, stats = run_backtest(symbol, interval, limit)
    if trades_df is None:
        return jsonify({"error": "Failed to fetch data"}), 500
    
    # Filter by strategy if specified
    if strategy in ["MACD", "RSI-EMA"]:
        trades_df = trades_df[trades_df["Strategy"] == strategy]
        stats = {
            "total_trades": len(trades_df),
            "total_pnl": trades_df["PnL"].sum() if not trades_df.empty else 0,
            "win_rate": len(trades_df[trades_df["PnL"] > 0]) / len(trades_df) if not trades_df.empty else 0
        }
    
    # Convert timestamps to strings for JSON
    trades_df["Entry Time"] = trades_df["Entry Time"].astype(str)
    trades_df["Exit Time"] = trades_df["Exit Time"].astype(str)
    
    return jsonify({
        "trades": trades_df.to_dict(orient="records"),
        "stats": stats
    })

@app.route("/download_csv", methods=["GET"])
def download_csv():
    """
    Download trades_output.csv.
    """
    csv_path = "trades_output.csv"
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404
    return send_file(csv_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)