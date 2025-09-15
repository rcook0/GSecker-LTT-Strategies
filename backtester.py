'''
Reads your trades.csv (from alert logger).

Rebuilds each trade from entry → exit.

Computes R-multiple for each trade (normalized profit).

Prints win rate, expectancy.

Plots an equity curve in R units.
'''

import pandas as pd
import matplotlib.pyplot as plt

LEDGER_FILE = "trades.csv"

def load_trades(file=LEDGER_FILE):
    df = pd.read_csv(file, parse_dates=["timestamp"])
    # Keep only entry/exit rows
    return df

def build_trade_pairs(df: pd.DataFrame):
    """
    Reconstruct trades: match entry rows with next exit (tp/sl/trail)
    """
    trades = []
    open_trades = {}

    for _, row in df.iterrows():
        strat = row["strategy"]
        if row["event"] == "entry":
            open_trades[strat] = {
                "timestamp": row["timestamp"],
                "strategy": strat,
                "dir": row["dir"],
                "entry": row["entry"],
                "sl": row["sl"],
                "tp": row["tp"],
            }
        elif row["event"] in ["sl","tp","trail"]:
            if strat in open_trades:
                tr = open_trades.pop(strat)
                tr["exit_event"] = row["event"]
                tr["exit_price"] = row["price"]
                trades.append(tr)

    return pd.DataFrame(trades)

def analyze_trades(trades: pd.DataFrame):
    trades = trades.copy()
    # Calculate R-multiple = (PnL / Risk)
    trades["risk"] = (trades["entry"] - trades["sl"]).abs()
    trades["pnl"] = trades.apply(
        lambda r: (r["exit_price"]-r["entry"]) if r["dir"]=="long"
        else (r["entry"]-r["exit_price"]), axis=1)
    trades["R"] = trades["pnl"] / trades["risk"]

    # Basic stats
    winrate = (trades["R"]>0).mean()
    avgR    = trades["R"].mean()
    expectancy = trades["R"].mean()
    print("Win rate:", round(winrate*100,2),"%")
    print("Avg R:", round(avgR,2))
    print("Expectancy (R):", round(expectancy,2))

    # Equity curve
    trades = trades.sort_values("timestamp")
    trades["equity"] = trades["R"].cumsum()
    plt.figure(figsize=(10,5))
    plt.plot(trades["timestamp"], trades["equity"], marker="o")
    plt.title("Equity Curve (in R units)")
    plt.grid(True)
    plt.show()

    return trades

if __name__ == "__main__":
    df = load_trades()
    trades = build_trade_pairs(df)
    print(trades)
    analyzed = analyze_trades(trades)
