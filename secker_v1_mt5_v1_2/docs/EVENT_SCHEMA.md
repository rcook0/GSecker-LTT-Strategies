# Event schema (CSV / log contract)

The EA optionally writes a **CSV ledger** to `MQL5/Files/`.

Default filename: `secker_v1_ledger.csv`

## Header
The EA auto-creates a header row if the file is empty:

```
timestamp_utc,symbol,timeframe,strategy,position_id,magic,direction,event,qty,entry_price,sl_price,tp1_price,tpnext_price,event_price,comment
```

## Field definitions
- **timestamp_utc**: time in UTC (format produced by `TimeToString(TIME_DATE|TIME_SECONDS)`, e.g. `2026.02.13 14:20:00`)
- **symbol**: MT5 symbol
- **timeframe**: chart timeframe string (`M5`, `H1`, `H4`, etc.)
- **strategy**: `180PC` | `T-Wave` | `Volatility Reversal` | `Power Pivots`
- **position_id**:
  - for pending orders: the **order ticket**
  - for market positions/events: the **position ticket**
  - for broker exits: the **DEAL_POSITION_ID** (from history)
- **magic**: the EA magic number (`InpMagicBase + strategyIndex`)
- **direction**: `BUY` or `SELL` (direction of the *position*, not the closing deal)
- **event**: one of:
  - `order_place` (pending placed)
  - `order_cancel` (pending deleted: expiry or replacement)
  - `entry` (market position opened)
  - `be_set` (SL moved to break-even)
  - `trail_update` (SL updated based on structure)
  - `time_exit` (manual time-based close; VolRev expiry)
  - `exit_sl` | `exit_tp` | `exit_stopout` | `exit_manual` | `exit_other` (captured in `OnTradeTransaction`)
- **qty**: lots
- **entry_price**: entry price reference (best-effort)
- **sl_price**: SL at the time of the event (best-effort)
- **tp1_price**: TP milestone (1R) reference (best-effort)
- **tpnext_price**: TPNext reference used for RR tightening after BE (best-effort)
- **event_price**: price at which the event occurred (for exits: deal price)
- **comment**: free-form (expiry bars, deal id, deal reason, etc.)

## Notes / limitations
- The EA is designed to be MT5 Strategy Tester compatible. The most accurate intrabar BE behavior occurs with **Every tick based on real ticks**.
- The indicator does not write CSV; it only renders visuals.
