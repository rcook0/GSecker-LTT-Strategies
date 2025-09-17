import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

STRATEGIES = [
    "180PC",
    "T-Wave",
    "Breakfast",
    "Bollinger Bounce",
    "Pip River",
    "Pip Runner",
    "Volatility Reversal",
    "Power Pivots"
]

def generate_dummy_trades(n_trades=100, start_date="2025-01-01", fixed_strategy=None, trail_prob=0.2):
    dirs = ["long", "short"]
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for i in range(n_trades):
        strat = fixed_strategy if fixed_strategy else random.choice(STRATEGIES)
        direction = random.choice(dirs)

        # Synthetic price levels
        base_price = 100 + np.random.normal(0, 10)
        risk = np.random.uniform(1.0, 3.0)
        reward = risk * np.random.choice([0.8, 1.0, 1.5, 2.0])

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
        # Exit timestamp
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

        # Trailing stop event with configurable probability
        if random.random() < trail_prob:
            trail_time = entry_time + timedelta(hours=random.randint(1, 6))
            trail_price = (entry + exit_price) / 2  # midpoint between entry & exit
            rows.append({
                "timestamp": trail_time.strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": strat,
                "event": "trail",
                "dir": direction,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "price": round(trail_price, 2)
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
    parser = argparse.ArgumentParser(description="Generate dummy trades.csv for Secker backtester")
    parser.add_argument("--trades", type=int, default=200, help="Number of trades to generate (default: 200)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="trades.csv", help="Output CSV file")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, help="Generate only for one strategy (optional)")
    parser.add_argument("--trail-prob", type=float, default=0.2, help="Probability of adding a trail event (0–1, default=0.2)")
    args = parser.parse_args()

    df = generate_dummy_trades(
        n_trades=args.trades,
        start_date=args.start,
        fixed_strategy=args.strategy,
        trail_prob=args.trail_prob
    )
    df.to_csv(args.out, index=False)
    print(f"✅ Dummy {args.out} generated with {args.trades} trades ({len(df)} rows)")
    if args.strategy:
        print(f"👉 Strategy fixed: {args.strategy}")
    else:
        print("👉 Strategy mix: uniform random across all")
    print(f"👉 Trailing stop probability: {args.trail_prob*100:.0f}%")