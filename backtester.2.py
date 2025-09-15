#portfolio equity curve

import pandas as pd
import matplotlib.pyplot as plt

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

def analyze_trades(trades: pd.DataFrame):
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
    print()

    # === Strategy stats ===
    grouped = trades.groupby("strategy")
    print("=== Strategy Split ===")
    for strat, g in grouped:
        wr = (g["R"]>0).mean()*100
        exp = g["R"].mean()
        count = len(g)
        print(f"{strat}: {count} trades | Win rate {wr:.1f}% | Expectancy {exp:.2f} R")

    # === Equity curves ===
    plt.figure(figsize=(12,7))
    # per-strategy
    for strat, g in grouped:
        g = g.sort_values("timestamp")
        g["equity"] = g["R"].cumsum()
        plt.plot(g["timestamp"], g["equity"], marker="o", label=strat)
    # portfolio equity
    trades = trades.sort_values("timestamp")
    trades["portfolio_equity"] = trades["R"].cumsum()
    plt.plot(trades["timestamp"], trades["portfolio_equity"],
             color="black", linewidth=2.5, label="Portfolio Total")
    plt.title("Equity Curves by Strategy + Portfolio (in R units)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return trades

if __name__ == "__main__":
    df = load_trades()
    trades = build_trade_pairs(df)
    print(trades.head())
    analyze_trades(trades)
