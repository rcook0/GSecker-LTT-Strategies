import json
import csv
from datetime import datetime
from flask import Flask, request

# ===== Option A: Run as Webhook Server =====
# TradingView can POST alerts here using your local/public endpoint
app = Flask(__name__)
LEDGER_FILE = "trades.csv"

def log_trade(event: dict):
    """Append trade event dict to CSV ledger"""
    header = ["timestamp", "strategy", "dir", "event", "entry", "sl", "tp", "price"]
    file_exists = False
    try:
        file_exists = open(LEDGER_FILE).readline() != ""
    except:
        pass

    with open(LEDGER_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": event.get("strategy"),
            "dir": event.get("dir"),
            "event": event.get("event"),
            "entry": event.get("entry", ""),
            "sl": event.get("sl", ""),
            "tp": event.get("tp", ""),
            "price": event.get("price", ""),
        }
        writer.writerow(row)
    print("Logged:", row)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        log_trade(data)
        return {"status": "ok"}, 200
    except Exception as e:
        return {"error": str(e)}, 400

# ===== Option B: Manual File/StdIn Reader =====
def parse_alert_line(line: str):
    try:
        event = json.loads(line.strip())
        log_trade(event)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", line)

if __name__ == "__main__":
    # Uncomment one mode depending on usage:

    # ---- Webhook Mode ----
    app.run(host="0.0.0.0", port=5000, debug=True)

    # ---- File/StdIn Mode (comment out Flask above) ----
    # with open("alerts.txt") as f:
    #     for line in f:
    #         parse_alert_line(line)
