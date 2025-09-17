import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

def generate_dummy_trades(n_trades=100, start_date="2025-01-01"):
    strategies = ["180PC", "T-Wave", "Breakfast"]
    dirs = ["long", "short"]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for i in range(n_trades):
        strat = random.choice(strategies)
        direction = random.choice(dirs)

        # Synthetic price levels
        base_price = 100 + np.random.normal(0, 5)
        risk = np.random.uniform(1.0, 3.0)  # distance to SL
        reward = risk * np.random.choice([0.8, 1.0, 1.2, 2.0])  # RR between 0.8–2.0

        if direction == "long":
            entry = base_price
            sl = entry - risk
            tp = entry + reward
        else:
            entry = base_price
            sl = entry + risk
            tp = entry - reward

        # Entry timestamp
        entry_time = start_dt + timedelta(hours=i*6)
        # Exit timestamp within 1–12 hours later
        exit_time = entry_time + timedelta(hours=random.randint(1, 12))

        # Decide outcome
        outcome = random.choice(["sl", "tp"])
        exit_price = sl if outcome == "sl" else tp

        # Entry row
        rows.append({
            "timestamp": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strat,
            "event": "entry",
            "dir": direction,
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "price": ""
        })

        # Exit row
        rows.append({
            "timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strat,
            "event": outcome,
            "dir": direction,
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "price": round(exit_price, 2)
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy trades.csv for backtester")
    parser.add_argument("--trades", type=int, default=200, help="Number of trades to generate (default: 200)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="trades.csv", help="Output CSV file")
    args = parser.parse_args()

    df = generate_dummy_trades(n_trades=args.trades, start_date=args.start)
    df.to_csv(args.out, index=False)
    print(f"✅ Dummy {args.out} generated with {args.trades} trades ({len(df)} rows)")