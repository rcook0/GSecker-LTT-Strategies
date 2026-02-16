# Secker v1.2 MT5 Bundle

This workspace contains MT5 (MQL5) implementations for a Secker-style **master framework** with **4 strategies**:

- 180PC
- T-Wave
- Volatility Reversal
- Power Pivots

## Key behavior
- Bar-close entries (deterministic)
- **TP milestone = 1R**
- **Break-even (BE) is set on intrabar touch of TP(1R)**
- Post-BE trailing is **structural** (no ATR trailing)

## v1.2 highlights
### Strategy accuracy upgrades
- **180PC**: implements Secker's **Ring Low / Ring High** 2-bar pattern with **stop entries** (1 pip offset), optional HTF EMA agreement (default D1) and optional previous D1 candle-color filter.
- **Volatility Reversal**: implements the false-break reversal (break prior extreme + close back inside) with **stop entries** and pending expiry.

### Structure + trade management
1) **Structure = ATR*N ZigZag (non-repainting confirmation)**
   - Used for post-BE **structural trailing** (no ATR trail).
   - Controls:
     - `InpZZ_ATRMult` (N)
     - `InpZZ_Lookback` (scan window)

2) **180PC tiny regime gate (optional chop filter)**
   - `REG_SLOPE_ATR` (default) or `REG_ADX`
   - Defaults are mild.

3) **Break-even (BE) is set intrabar on touch of TP(1R)**
   - BE trigger uses live Bid/Ask (or real ticks in Strategy Tester).

4) **Exit logging**
   - `OnTradeTransaction` logs exits that happen via SL/TP (and other broker reasons) so the CSV ledger is complete.

## Contents
- `mt5/SeckerMasterSignals_v1_2.mq5` : Indicator (arrows + SL/TP dots + TP line + yellow live SL dot)
- `mt5/SeckerMasterEA_v1_2.mq5`      : EA (auto trading, Strategy Tester compatible)
- `docs/MT5_SETUP.md`                : install/compile/tester steps
- `docs/EVENT_SCHEMA.md`             : CSV ledger schema
- `docs/VERIFY.md`                   : compile + tester verification checklist
- `examples/sample_ledger_lines.csv` : example ledger lines

## Quick start
1. Copy files into your MT5 Data Folder:
   - Indicator -> `MQL5/Indicators/`
   - EA -> `MQL5/Experts/`
2. Compile in MetaEditor.
3. Run Strategy Tester on `SeckerMasterEA_v1_2`:
   - Model: **Every tick based on real ticks**
   - Choose strategy via `InpStrategy`

## Notes
- If you want strict Secker basic mode (single TP at 1R), set `InpUseHardTP = true`.
- Structural trailing is pivot-event driven (bar-close), while BE is intrabar (tick driven).
