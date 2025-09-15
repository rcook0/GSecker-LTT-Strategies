'''
Risk of Ruin estimate

Probability of equity dropping below a user-defined threshold (e.g. -10R, or -20R).
Based on Monte Carlo resampled equity paths.
This gives a realistic sense of blowing up risk if trade order goes badly.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby

LEDGER_FILE = "trades.csv"

def load_trades(file=LEDGER_FILE):
    df = pd.read_csv(file, parse_dates=["timestamp"])
    return df

def build_trade_pairs(df: pd.DataFrame):
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

def analyze_trades(trades: pd.DataFrame, montecarlo_runs=1000, ruin_threshold=-10):
    trades = trades.copy()
    trades["risk"] = (trades["entry"] - trades["sl"]).abs()
    trades["pnl"] = trades.apply(
        lambda r: (r["exit_price"]-r["entry"]) if r["dir"]=="long"
        else (r["entry"]-r["exit_price"]), axis=1)
    trades["R"] = trades["pnl"] / trades["risk"]

    # === Overall stats ===
    winrate = (trades["R"] > 0).mean()
    expectancy = trades["R"].mean()
    print("=== Overall Results ===")
    print("Trades:", len(trades))
    print("Win rate:", round(winrate*100,2), "%")
    print("Expectancy (R):", round(expectancy,2))

    # === Strategy stats ===
    grouped = trades.groupby("strategy")
    print("\n=== Strategy Split ===")
    for strat, g in grouped:
        wr = (g["R"]>0).mean()*100
        exp = g["R"].mean()
        count = len(g)
        print(f"{strat}: {count} trades | Win rate {wr:.1f}% | Expectancy {exp:.2f} R")

    # === Portfolio equity ===
    trades = trades.sort_values("timestamp")
    trades["portfolio_equity"] = trades["R"].cumsum()

    # Drawdowns
    equity = trades["portfolio_equity"].values
    peaks = np.maximum.accumulate(equity)
    dd = equity - peaks
    max_dd = dd.min()
    dd_durations = []
    cur_dur = 0
    for d in dd:
        if d < 0:
            cur_dur += 1
        elif cur_dur > 0:
            dd_durations.append(cur_dur)
            cur_dur = 0
    avg_dd_dur = np.mean(dd_durations) if dd_durations else 0

    print("\n=== Drawdown Analysis ===")
    print("Max Drawdown (R):", round(max_dd,2))
    print("Avg Drawdown Duration (trades):", round(avg_dd_dur,1))

    # Equity curves
    plt.figure(figsize=(12,7))
    for strat, g in grouped:
        g = g.sort_values("timestamp")
        g["equity"] = g["R"].cumsum()
        plt.plot(g["timestamp"], g["equity"], marker="o", label=strat)
    plt.plot(trades["timestamp"], trades["portfolio_equity"],
             color="black", linewidth=2.5, label="Portfolio Total")
    plt.title("Equity Curves by Strategy + Portfolio (in R units)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Monte Carlo Simulation ===
    if montecarlo_runs > 0:
        print(f"\n=== Monte Carlo Simulation ({montecarlo_runs} runs) ===")
        results = []
        ruin_count = 0
        for _ in range(montecarlo_runs):
            shuffled = trades["R"].sample(frac=1, replace=False).reset_index(drop=True)
            eq = shuffled.cumsum()
            results.append(eq.iloc[-1])
            if (eq.min() <= ruin_threshold):
                ruin_count += 1
        results = pd.Series(results)
        print("Median final equity (R):", round(results.median(),2))
        print("5th percentile:", round(results.quantile(0.05),2))
        print("95th percentile:", round(results.quantile(0.95),2))
        ruin_prob = ruin_count / montecarlo_runs
        print(f"Risk of Ruin (equity ≤ {ruin_threshold}R): {ruin_prob*100:.2f}%")

        plt.figure(figsize=(10,5))
        plt.hist(results, bins=30, alpha=0.7, color="skyblue", edgecolor="k")
        plt.axvline(results.median(), color="red", linestyle="--", label="Median")
        plt.title("Monte Carlo Final Equity Distribution (R)")
        plt.legend()
        plt.show()

    return trades

if __name__ == "__main__":
    df = load_trades()
    trades = build_trade_pairs(df)
    print(trades.head())
    analyze_trades(trades, montecarlo_runs=1000, ruin_threshold=-10)
