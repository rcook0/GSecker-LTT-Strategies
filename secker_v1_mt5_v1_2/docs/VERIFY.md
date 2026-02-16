# v1.2 Verification Checklist

This is a practical checklist to verify the v1.2 MT5 bundle compiles and behaves as designed in the MT5 Strategy Tester and on a live/demo chart.

## 1) Compile verification (MetaEditor)
- Open and compile:
  - `SeckerMasterSignals_v1_2.mq5`
  - `SeckerMasterEA_v1_2.mq5`
- Expected: **0 errors** (warnings are OK, but should be minimal).

## 2) Smoke test in Strategy Tester
Recommended settings:
- Model: **Every tick based on real ticks**
- Use a reasonably liquid symbol/timeframe (e.g., EURUSD H4 for 180PC)

Steps:
1. Select Expert: `SeckerMasterEA_v1_2`
2. Set `InpStrategy` to each strategy one at a time:
   - `180PC` (prefers H4; refers to D1 EMA agreement)
   - `T-Wave` (works on multiple TFs; pin bar signal on close)
   - `Volatility Reversal` (works on multiple TFs; pending expiry important on lower TFs)
   - `Power Pivots` (bias from D1 EMA(200); pivots from previous session)
3. Run for a sample period where trades occur.

Expected behavior checks:
- **Entries occur only on bar-close** (no mid-bar “new signals”).
- **180PC & VolRev use stop entries** when their pending toggles are enabled:
  - BUY stop = signal bar high + pip offset
  - SELL stop = signal bar low - pip offset
- **BE trigger is intrabar**:
  - when price touches TP(1R) (bid/ask), SL is moved to entry (unless `InpUseHardTP=true`).
- After BE, **structural trailing** updates only on **new confirmed swing pivots** (ATR*N ZigZag).

## 3) Indicator + EA parity
Attach the indicator to the same symbol/timeframe you are testing.

Expected:
- Blue arrows for buy/sell signals.
- On the same signal candle:
  - **Red dot** at the SL reference
  - **Green dot** at the TP(1R) reference
  - **Thin blue dotted horizontal line** at TP(1R) extending to the right
- If the EA has an open position for the selected strategy/magic:
  - **Yellow dot** on the current candle at the live SL.

## 4) Ledger (CSV) verification
If `InpWriteCSV=true`:
- The file should appear under: `MQL5/Files/secker_v1_ledger.csv`
- Expect to see events like:
  - `order_place`, `entry`, `be_set`, `trail_update`, and exit events `exit_sl` / `exit_tp`.

## Known limitations (intentional in v1.2)
- No news filter (manual avoidance recommended on lower TFs).
- No partial take-profit logic (Secker default tends to be single TP=1R + structural trailing).
- Exit “reason” is derived from broker deal reason; trailing-stop exits are typically reported as `exit_sl`.
