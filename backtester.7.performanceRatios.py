'''
Portfolio-Level Performance Ratios

Profit Factor = gross profits ÷ gross losses

Sharpe Ratio = mean(R) ÷ std(R) (using R units, assume risk-free = 0)

Sortino Ratio = mean(R) ÷ std(negative R only)

Expectancy (R) = already included

Win Rate % = already included

Per Strategy Ratios:

Portfolio row (overall stats)

One row per strategy (individual stats)

Each row includes: Win Rate %, Expectancy (R), Profit Factor, Sharpe, Sortino, MaxDD, AvgDD_Dur, RuinProb%

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from datetime import datetime
import argparse
import io
import json, yaml
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

LEDGER_FILE = "trades.csv"
CONFIG_FILE = "config.yaml"

DEFAULT_CONFIG = {
    "montecarlo_runs": 1000,
    "ruin_threshold": -10,
    "embed_charts": True,
    "ruin_thresholds": {
        "180PC": -6,
        "T-Wave": -12,
        "Breakfast": -8
    }
}

def ensure_config():
    cfg_path = Path(CONFIG_FILE)
    if not cfg_path.exists():
        with open(cfg_path, "w") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
        print(f"⚙️  Default config created → {CONFIG_FILE}")
        return DEFAULT_CONFIG
    else:
        with open(cfg_path, "r") as f:
            if cfg_path.suffix == ".json":
                return json.load(f)
            else:
                return yaml.safe_load(f)

def load_trades(file=LEDGER_FILE):
    return pd.read_csv(file, parse_dates=["timestamp"])

def build_trade_pairs(df: pd.DataFrame):
    trades, open_trades = [], {}
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
        elif row["event"] in ["sl", "tp", "trail"]:
            if strat in open_trades:
                tr = open_trades.pop(strat)
                tr["exit_event"] = row["event"]
                tr["exit_price"] = row["price"]
                trades.append(tr)
    return pd.DataFrame(trades)

def compute_drawdown(equity):
    peaks = np.maximum.accumulate(equity)
    dd = equity - peaks
    max_dd = dd.min() if len(dd) else 0
    durations, cur_dur = [], 0
    for d in dd:
        if d < 0: cur_dur += 1
        elif cur_dur > 0:
            durations.append(cur_dur)
            cur_dur = 0
    avg_dd_dur = np.mean(durations) if durations else 0
    return max_dd, avg_dd_dur

def performance_ratios(R_series):
    """Compute Profit Factor, Sharpe, Sortino."""
    R = np.array(R_series.dropna())
    if len(R) == 0:
        return (None, None, None)

    gross_profit = R[R > 0].sum()
    gross_loss = -R[R < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    sharpe = np.mean(R) / np.std(R) if np.std(R) > 0 else None

    downside = R[R < 0]
    sortino = np.mean(R) / np.std(downside) if len(downside) > 0 and np.std(downside) > 0 else None

    return (profit_factor, sharpe, sortino)

def save_chart_as_image(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

def analyze_trades(trades: pd.DataFrame, montecarlo_runs=1000,
                   ruin_threshold=-10, ruin_thresholds=None,
                   embed_charts=True):
    trades = trades.copy()
    trades["risk"] = (trades["entry"] - trades["sl"]).abs()
    trades["pnl"] = trades.apply(
        lambda r: (r["exit_price"] - r["entry"]) if r["dir"] == "long"
        else (r["entry"] - r["exit_price"]), axis=1)
    trades["R"] = trades["pnl"] / trades["risk"]

    summary = []
    mc_summary = []
    mc_results = pd.DataFrame()

    # === Portfolio stats ===
    trades = trades.sort_values("timestamp")
    trades["portfolio_equity"] = trades["R"].cumsum()
    max_dd, avg_dd_dur = compute_drawdown(trades["portfolio_equity"].values)
    winrate = (trades["R"] > 0).mean()
    expectancy = trades["R"].mean()
    pf, sharpe, sortino = performance_ratios(trades["R"])

    summary.append({
        "Strategy": "Portfolio",
        "Trades": len(trades),
        "WinRate%": winrate*100,
        "ExpectancyR": expectancy,
        "ProfitFactor": pf,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD(R)": max_dd,
        "AvgDD_Dur": avg_dd_dur,
        "RuinProb%": None,
    })

    # === Strategy stats ===
    grouped = trades.groupby("strategy")
    equity_curves = {}
    for strat, g in grouped:
        g = g.sort_values("timestamp")
        g["equity"] = g["R"].cumsum()
        equity_curves[strat] = g[["timestamp", "equity"]].copy()
        max_dd, avg_dd_dur = compute_drawdown(g["equity"].values)
        winrate = (g["R"] > 0).mean()
        expectancy = g["R"].mean()
        pf, sharpe, sortino = performance_ratios(g["R"])
        summary.append({
            "Strategy": strat,
            "Trades": len(g),
            "WinRate%": winrate*100,
            "ExpectancyR": expectancy,
            "ProfitFactor": pf,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD(R)": max_dd,
            "AvgDD_Dur": avg_dd_dur,
            "RuinProb%": None,
        })

    # === Monte Carlo as before ===
    if montecarlo_runs > 0:
        results = []
        ruin_count = 0
        for _ in range(montecarlo_runs):
            shuffled = trades["R"].sample(frac=1, replace=False).reset_index(drop=True)
            eq = shuffled.cumsum()
            results.append(eq.iloc[-1])
            if eq.min() <= ruin_threshold:
                ruin_count += 1
        ruin_prob = ruin_count / montecarlo_runs
        for s in summary:
            if s["Strategy"] == "Portfolio":
                s["RuinProb%"] = ruin_prob*100
        mc_summary.append({
            "Strategy": "Portfolio",
            "RuinThreshold": ruin_threshold,
            "RuinProb%": ruin_prob*100,
            "MedianFinalR": np.median(results),
            "P05": np.percentile(results, 5),
            "P95": np.percentile(results, 95)
        })
        mc_results["Portfolio_FinalEquity"] = results

        if ruin_thresholds:
            for strat, g in grouped:
                if len(g) == 0: continue
                strat_thr = ruin_thresholds.get(strat, ruin_threshold)
                results = []
                ruin_count = 0
                for _ in range(montecarlo_runs):
                    shuffled = g["R"].sample(frac=1, replace=False).reset_index(drop=True)
                    eq = shuffled.cumsum()
                    results.append(eq.iloc[-1])
                    if eq.min() <= strat_thr:
                        ruin_count += 1
                ruin_prob = ruin_count / montecarlo_runs
                for s in summary:
                    if s["Strategy"] == strat:
                        s["RuinProb%"] = ruin_prob*100
                mc_summary.append({
                    "Strategy": strat,
                    "RuinThreshold": strat_thr,
                    "RuinProb%": ruin_prob*100,
                    "MedianFinalR": np.median(results),
                    "P05": np.percentile(results, 5),
                    "P95": np.percentile(results, 95)
                })
                mc_results[strat+"_FinalEquity"] = results

    # === Save Excel with Config & Charts same as before ===
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = f"strategy_report_{ts}.xlsx"
    curve_file  = f"equity_curve_{ts}.png"
    mc_file     = f"mc_histogram_{ts}.png"

    df_summary = pd.DataFrame(summary)
    df_mc_summary = pd.DataFrame(mc_summary)

    runtime_info = [
        {"Parameter": "Run Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Parameter": "Monte Carlo Runs", "Value": montecarlo_runs},
        {"Parameter": "Global Ruin Threshold", "Value": ruin_threshold},
        {"Parameter": "Embed Charts", "Value": embed_charts},
        {"Parameter": "Trades.csv Rows", "Value": len(trades)},
        {"Parameter": "Strategies Detected", "Value": ", ".join(sorted(trades["strategy"].unique()))},
    ]
    if ruin_thresholds:
        for k,v in ruin_thresholds.items():
            runtime_info.append({"Parameter": f"RuinThreshold_{k}", "Value": v})
    df_config = pd.DataFrame(runtime_info)

    with pd.ExcelWriter(report_file, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        trades.to_excel(writer, sheet_name="Trades", index=False)
        eq_df = trades[["timestamp", "portfolio_equity"]].rename(columns={"portfolio_equity":"Portfolio"})
        for strat, eq in equity_curves.items():
            eq_df = eq_df.merge(eq, on="timestamp", how="outer", suffixes=("","_"+strat))
        eq_df.to_excel(writer, sheet_name="Equity", index=False)
        if not df_mc_summary.empty:
            df_mc_summary.to_excel(writer, sheet_name="MonteCarlo", index=False)
            mc_results.to_excel(writer, sheet_name="MC_Distribution", index=False)
        df_config.to_excel(writer, sheet_name="Config", index=False)

    # Charts & embedding (unchanged from last version)...

    return trades, df_summary, df_mc_summary
