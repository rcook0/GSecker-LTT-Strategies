# MT5 v1.2 Setup (Secker Master)

## Files
Copy these files from this ZIP into your MT5 **Data Folder**:

- `mt5/SeckerMasterSignals_v1_2.mq5` → `MQL5/Indicators/`
- `mt5/SeckerMasterEA_v1_2.mq5`      → `MQL5/Experts/`

Open the Data Folder via:
- MT5 → **File → Open Data Folder**

## Compile
1. MT5 → **Tools → MetaQuotes Language Editor** (MetaEditor)
2. Open both `.mq5` files
3. Press **F7** to compile
4. Confirm **0 errors** in the bottom output pane

## Strategy Tester (recommended)
1. MT5 → **View → Strategy Tester**
2. Expert: `SeckerMasterEA_v1_2`
3. Symbol: choose your instrument (e.g., EURUSD, XAUUSD)
4. Period: the timeframe you want to test (e.g., H4 for 180PC)
5. Model: **Every tick based on real ticks** (best parity with intrabar BE trigger)
6. Inputs:
   - `InpStrategy`: select one of the 4 strategies
   - `InpUseHardTP`: OFF by default (TP(1R) is a reference milestone)
   - `InpZZ_ATRMult` and `InpZZ_Lookback`: structural trailing sensitivity
7. Run

## Live usage
- Attach `SeckerMasterSignals_v1_2` to a chart for visuals.
- Attach `SeckerMasterEA_v1_2` to the same symbol/timeframe if you want auto-trading.
- Ensure **Algo Trading / AutoTrading** is enabled.

## Notes
- BE is applied on **intrabar touch** of TP(1R) reference.
- Structural trailing updates are **bar-close pivot events** (ATR*N ZigZag), not tick-by-tick.
