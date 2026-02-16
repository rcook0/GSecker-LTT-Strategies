# Changelog

## v1.2
- 180PC: switched to Secker-style **Ring Low / Ring High** 2-bar pattern.
- 180PC + Volatility Reversal: default entry is now a **stop order** 1 pip beyond the signal bar (configurable), with pending expiry (configurable).
- EA: added **OnTradeTransaction exit logging** so the CSV ledger captures SL/TP exits (and other broker reasons).
- Indicator: updated to match v1.2 signal definitions (Ring + false-break reversal) and draws:
  - Blue arrows (buy/sell)
  - Red SL dot + green TP1 dot on the signal bar
  - Thin blue dotted TP line extended to the right
  - Yellow live SL dot aligned with the current candle (reads from open position SL)

## v1.1
- Structure/trailing switched from fractals to an **ATR*N ZigZag** (non-repainting confirmation).
- 180PC: SL can be anchored to previous confirmed swing high/low (ZigZag).
- 180PC: optional “tiny regime gate” (slope/ATR or ADX) to reduce dead-flat chop.
- EA: fixed new-bar handling (single `IsNewBar()` evaluation per tick) to avoid missed entries.
- Indicator + EA kept behaviorally consistent.
