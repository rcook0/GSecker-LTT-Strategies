//+------------------------------------------------------------------+
//| SeckerMasterEA_v1_1.mq5                                             |
//| v1.1: 180PC, T-Wave, Volatility Reversal, Power Pivots              |
//| Management: TP(1R) milestone -> BE intrabar touch -> structural   |
//| trailing (no ATR trail). MT5 Strategy Tester compatible.          |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>
CTrade trade;

//-------------------- Inputs
enum StrategySel
{
   STRAT_180PC = 0,
   STRAT_TWAVE = 1,
   STRAT_VOLREV = 2,
   STRAT_POWERPIVOTS = 3
};

input StrategySel      InpStrategy           = STRAT_180PC;
input long             InpMagicBase           = 180100;
input double           InpFixedLots           = 0.10;

// Core / shared
input bool             InpUseHardTP           = false;   // If true: place broker TP at TP1 (=1R). If false: TP1 is milestone only.
input double           InpTPNextR             = 2.0;     // Reference target for RR-consistent tightening after BE
input double           InpRRTarget            = 1.0;     // Maintain approx RR from spot to TPNext (>= 1 by default)
input int              InpATRLen              = 14;
input double           InpSLBufferATR         = 0.10;    // structural SL buffer = ATR * this (0.10 = small)
input bool             InpUseRRTighten        = true;    // after BE, tighten stop relative to remaining distance to TPNext

// ZigZag structure (ATR*N)
input double           InpZZ_ATRMult          = 1.50;    // N in ATR(14) * N (swing confirmation threshold)
input int              InpZZ_Lookback         = 800;     // bars scanned to find last confirmed swing

// 180PC
input ENUM_TIMEFRAMES  Inp180_HTF             = PERIOD_D1;
input int              Inp180_EMAFast         = 8;
input int              Inp180_EMASlow         = 20;

enum RegimeMethod { REG_SLOPE_ATR = 0, REG_ADX = 1 };
input bool             Inp180_RegimeEnable    = true;   // tiny chop filter (refinement)
input RegimeMethod     Inp180_RegimeMethod    = REG_SLOPE_ATR;
input int              Inp180_SlopeLookback   = 5;      // bars
input double           Inp180_MinSlopeATR     = 0.15;   // |EMA_slow(t)-EMA_slow(t-L)| / ATR >= threshold
input int              Inp180_ADXLen          = 14;
input double           Inp180_MinADX          = 18.0;

// T-Wave
input double           InpTW_WickBodyRatio    = 2.5;     // wick >= ratio * body
input double           InpTW_MinWickATR       = 0.50;    // wick >= ATR * this
input double           InpTW_MaxBodyATR       = 0.60;    // body <= ATR * this

// Volatility Reversal
input int              InpVR_ExpiryBars       = 1;       // hold for 1 bar if not exited
input double           InpVR_BufferATR        = 0.10;    // SL buffer = ATR * this

// Power Pivots
input ENUM_TIMEFRAMES  InpPP_PivotTF          = PERIOD_D1;
input int              InpPP_BiasEMA          = 200;
input bool             InpPP_TPToNextPivot    = false;   // if true: TP uses pivot ladder, else uses 1R milestone logic

// Logging
input bool             InpWriteCSV            = true;
input string           InpCSVFileName         = "secker_v1_ledger.csv";

//-------------------- Handles
int hATR = INVALID_HANDLE;
int hEmaFastCur = INVALID_HANDLE, hEmaSlowCur = INVALID_HANDLE;
int hEmaFastHTF = INVALID_HANDLE, hEmaSlowHTF = INVALID_HANDLE;
int hEmaBiasD1 = INVALID_HANDLE;
int hADX = INVALID_HANDLE; // 180PC regime filter (optional)

//-------------------- State
datetime g_lastBarTime = 0;

struct StratState
{
   bool     hasPos;
   ulong    ticket;
   bool     beSet;
   datetime entryTime;
   double   entry;
   double   sl0;
   double   R;
   double   tp1;
   double   tpNext;
   datetime lastTrailPivotTime;
};
StratState S;

//-------------------- Utils
string StrategyName(StrategySel s)
{
   if(s==STRAT_180PC) return "180PC";
   if(s==STRAT_TWAVE) return "T-Wave";
   if(s==STRAT_VOLREV) return "Volatility Reversal";
   if(s==STRAT_POWERPIVOTS) return "Power Pivots";
   return "Unknown";
}

int MagicFor(StrategySel s)
{
   return (int)(InpMagicBase + (int)s);
}

bool IsNewBar()
{
   datetime t = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(t != g_lastBarTime)
   {
      g_lastBarTime = t;
      return true;
   }
   return false;
}

double GetATR(int shift=1)
{
   if(hATR==INVALID_HANDLE) return 0.0;
   double buf[];
   if(CopyBuffer(hATR, 0, shift, 1, buf) != 1) return 0.0;
   return buf[0];
}

double GetMA(int handle, int shift=1)
{
   if(handle==INVALID_HANDLE) return 0.0;
   double buf[];
   if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return 0.0;
   return buf[0];
}

// ZigZag (ATR*N) — non-repainting swing confirmation on bar data
// We scan a lookback window and return the last confirmed swing (high or low).
// Confirmation rule:
//  - In up-leg: track extreme high; confirm swing HIGH when price drops by >= ATR*N from that extreme.
//  - In down-leg: track extreme low; confirm swing LOW when price rises by >= ATR*N from that extreme.
bool GetLastZigZagPivot(bool wantLow, int lookback, double atrMult, double &outPrice, datetime &outTime)
{
   outPrice = 0.0; outTime = 0;
   if(hATR==INVALID_HANDLE) return false;

   if(lookback < 80) lookback = 80;

   MqlRates rates[];
   int got = CopyRates(_Symbol, PERIOD_CURRENT, 1, lookback, rates);
   if(got < 50) return false;
   ArraySetAsSeries(rates, true);

   double atr[];
   int ca = CopyBuffer(hATR, 0, 1, got, atr);
   if(ca != got) return false;
   ArraySetAsSeries(atr, true);

   int oldest = got - 1;

   int dir = +1; // start by seeking a swing high
   double extremeH = rates[oldest].high;
   datetime extremeHTime = rates[oldest].time;
   double extremeL = rates[oldest].low;
   datetime extremeLTime = rates[oldest].time;

   double lastHigh = 0.0; datetime lastHighTime = 0;
   double lastLow  = 0.0; datetime lastLowTime  = 0;

   for(int idx = oldest - 1; idx >= 0; --idx)
   {
      double thr = atr[idx] * atrMult;
      if(thr <= 5*_Point) thr = 5*_Point;

      if(dir == +1)
      {
         if(rates[idx].high > extremeH)
         {
            extremeH = rates[idx].high;
            extremeHTime = rates[idx].time;
         }
         if((extremeH - rates[idx].low) >= thr)
         {
            // confirm swing HIGH at extremeHTime
            lastHigh = extremeH;
            lastHighTime = extremeHTime;

            dir = -1;
            extremeL = rates[idx].low;
            extremeLTime = rates[idx].time;
         }
      }
      else
      {
         if(rates[idx].low < extremeL)
         {
            extremeL = rates[idx].low;
            extremeLTime = rates[idx].time;
         }
         if((rates[idx].high - extremeL) >= thr)
         {
            // confirm swing LOW at extremeLTime
            lastLow = extremeL;
            lastLowTime = extremeLTime;

            dir = +1;
            extremeH = rates[idx].high;
            extremeHTime = rates[idx].time;
         }
      }
   }

   if(wantLow)
   {
      if(lastLowTime == 0) return false;
      outPrice = lastLow;
      outTime  = lastLowTime;
      return true;
   }
   else
   {
      if(lastHighTime == 0) return false;
      outPrice = lastHigh;
      outTime  = lastHighTime;
      return true;
   }
}

double GetADX(int shift=1)
{
   if(hADX==INVALID_HANDLE) return 0.0;
   double b[];
   // ADX buffer is index 2 in MT5 iADX
   if(CopyBuffer(hADX, 2, shift, 1, b) != 1) return 0.0;
   return b[0];
}

bool RegimeOK_180PC()
{
   if(!Inp180_RegimeEnable) return true;

   double atr = GetATR(1);
   if(atr <= 0) atr = _Point;

   if(Inp180_RegimeMethod == REG_ADX)
   {
      double adx = GetADX(1);
      return (adx >= Inp180_MinADX);
   }
   else // REG_SLOPE_ATR
   {
      int L = Inp180_SlopeLookback;
      if(L < 1) L = 1;

      double emaS_now  = GetMA(hEmaSlowCur, 1);
      double emaS_past = GetMA(hEmaSlowCur, 1 + L);

      double slopeATR = MathAbs(emaS_now - emaS_past) / atr;
      return (slopeATR >= Inp180_MinSlopeATR);
   }
}

int StopsLevelPoints()
{
   long v=0;
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL, v)) return 0;
   return (int)v;
}

double NormalizePrice(double p)
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(p, digits);
}

bool HasOpenPositionByMagic(int magic, ulong &ticket)
{
   for(int i=PositionsTotal()-1; i>=0; --i)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      ticket = t;
      return true;
   }
   return false;
}

bool WriteCSV(const string &line)
{
   if(!InpWriteCSV) return true;
   int fh = FileOpen(InpCSVFileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_WRITE, ',');
   if(fh == INVALID_HANDLE) return false;
   FileSeek(fh, 0, SEEK_END);
   FileWriteString(fh, line + "\n");
   FileClose(fh);
   return true;
}

string TFToString(ENUM_TIMEFRAMES tf)
{
   if(tf==PERIOD_M1) return "M1";
   if(tf==PERIOD_M5) return "M5";
   if(tf==PERIOD_M15) return "M15";
   if(tf==PERIOD_M30) return "M30";
   if(tf==PERIOD_H1) return "H1";
   if(tf==PERIOD_H4) return "H4";
   if(tf==PERIOD_D1) return "D1";
   return IntegerToString((int)tf);
}

void LogEvent(const string &event, double qty, double entry, double sl, double tp1, double tpNext, double eventPrice, const string &comment)
{
   datetime now = TimeGMT();
   string line = StringFormat("%s,%s,%s,%s,%I64u,%d,%s,%s,%.2f,%.5f,%.5f,%.5f,%.5f,%.5f,%s",
                              TimeToString(now, TIME_DATE|TIME_SECONDS),
                              _Symbol,
                              TFToString((ENUM_TIMEFRAMES)Period()),
                              StrategyName(InpStrategy),
                              (long)S.ticket,
                              MagicFor(InpStrategy),
                              (S.hasPos && (int)PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) ? "LONG" : "SHORT",
                              event,
                              qty, entry, sl, tp1, tpNext, eventPrice, comment);
   WriteCSV(line);
}

//-------------------- Strategy signals
bool Signal_180PC(bool &isBuy, double &slOut)
{
   // Pullback-cross of EMA fast in direction of EMA8/EMA20 trend, confirmed on HTF too.
   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;

   double emaF1 = GetMA(hEmaFastCur, 1);
   double emaS1 = GetMA(hEmaSlowCur, 1);
   double emaF2 = GetMA(hEmaFastCur, 2);

   double emaFh = GetMA(hEmaFastHTF, 1);
   double emaSh = GetMA(hEmaSlowHTF, 1);

   bool trendUp = (emaFh > emaSh) && (emaF1 > emaS1);
   bool trendDn = (emaFh < emaSh) && (emaF1 < emaS1);

   // Regime / chop filter (refinement)
   if(!RegimeOK_180PC()) return false;

   // Pullback-cross: previous close below emaF then close above emaF (for long), opposite for short.
   bool crossUp = (r[2].close <= emaF2) && (r[1].close > emaF1);
   bool crossDn = (r[2].close >= emaF2) && (r[1].close < emaF1);

   double atr = GetATR(1);
   double buf = atr * InpSLBufferATR;

   if(trendUp && crossUp)
   {
      isBuy = true;
      double swingLow=0.0;
      datetime tmpT=0;
      if(GetLastZigZagPivot(true, InpZZ_Lookback, InpZZ_ATRMult, swingLow, tmpT))
         slOut = NormalizePrice(swingLow - buf);
      else
         slOut = NormalizePrice(r[1].low - buf);
      return true;
   }
   if(trendDn && crossDn)
   {
      isBuy = false;
      double swingHigh=0.0;
      datetime tmpT=0;
      if(GetLastZigZagPivot(false, InpZZ_Lookback, InpZZ_ATRMult, swingHigh, tmpT))
         slOut = NormalizePrice(swingHigh + buf);
      else
         slOut = NormalizePrice(r[1].high + buf);
      return true;
   }
   return false;
}

bool IsBullPin(const MqlRates &bar, double atr)
{
   double body = MathAbs(bar.close - bar.open);
   double upper = bar.high - MathMax(bar.close, bar.open);
   double lower = MathMin(bar.close, bar.open) - bar.low;
   if(body <= 0) body = 0.0000001;
   // bullish: long lower wick, small body, close near high
   if(lower < InpTW_WickBodyRatio * body) return false;
   if(lower < atr * InpTW_MinWickATR) return false;
   if(body  > atr * InpTW_MaxBodyATR) return false;
   if(bar.close <= bar.open) return false;
   if(upper > lower*0.6) return false;
   return true;
}

bool IsBearPin(const MqlRates &bar, double atr)
{
   double body = MathAbs(bar.close - bar.open);
   double upper = bar.high - MathMax(bar.close, bar.open);
   double lower = MathMin(bar.close, bar.open) - bar.low;
   if(body <= 0) body = 0.0000001;
   // bearish: long upper wick, small body, close near low
   if(upper < InpTW_WickBodyRatio * body) return false;
   if(upper < atr * InpTW_MinWickATR) return false;
   if(body  > atr * InpTW_MaxBodyATR) return false;
   if(bar.close >= bar.open) return false;
   if(lower > upper*0.6) return false;
   return true;
}

bool Signal_TWave(bool &isBuy, double &slOut)
{
   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;
   double atr = GetATR(1);
   double buf = atr * InpSLBufferATR;

   if(IsBullPin(r[1], atr))
   {
      isBuy = true;
      slOut = NormalizePrice(r[1].low - buf);
      return true;
   }
   if(IsBearPin(r[1], atr))
   {
      isBuy = false;
      slOut = NormalizePrice(r[1].high + buf);
      return true;
   }
   return false;
}

bool Signal_VolRev(bool &isBuy, double &slOut)
{
   // Buy: break prior bar low and close back above that prior low (false breakdown)
   // Sell: break prior bar high and close back below that prior high (false breakout)
   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;
   double atr = GetATR(1);
   double buf = atr * InpVR_BufferATR;

   double prevLow  = r[2].low;
   double prevHigh = r[2].high;

   bool buySig  = (r[1].low < prevLow)  && (r[1].close > prevLow);
   bool sellSig = (r[1].high > prevHigh) && (r[1].close < prevHigh);

   if(buySig)
   {
      isBuy = true;
      slOut = NormalizePrice(r[1].low - buf);
      return true;
   }
   if(sellSig)
   {
      isBuy = false;
      slOut = NormalizePrice(r[1].high + buf);
      return true;
   }
   return false;
}

void ComputeDailyPivots(double &P, double &R1, double &S1)
{
   double H = iHigh(_Symbol, InpPP_PivotTF, 1);
   double L = iLow(_Symbol, InpPP_PivotTF, 1);
   double C = iClose(_Symbol, InpPP_PivotTF, 1);
   P = (H + L + C) / 3.0;
   R1 = 2.0*P - L;
   S1 = 2.0*P - H;
}

bool Signal_PowerPivots(bool &isBuy, double &slOut, double &tpOut)
{
   // Bias: D1 close vs EMA200(D1)
   double emaBias = GetMA(hEmaBiasD1, 1);
   double d1Close = iClose(_Symbol, PERIOD_D1, 1);
   bool biasLong = d1Close >= emaBias;
   bool biasShort = d1Close < emaBias;

   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;

   double P,R1,S1;
   ComputeDailyPivots(P,R1,S1);

   bool crossUp = (r[2].close <= P) && (r[1].close > P);
   bool crossDn = (r[2].close >= P) && (r[1].close < P);

   double atr = GetATR(1);
   double buf = atr * InpSLBufferATR;

   tpOut = 0.0;

   if(biasLong && crossUp)
   {
      isBuy = true;
      slOut = NormalizePrice(S1 - buf); // protection below S1
      if(InpPP_TPToNextPivot)
         tpOut = NormalizePrice(R1 - buf); // near R1
      return true;
   }
   if(biasShort && crossDn)
   {
      isBuy = false;
      slOut = NormalizePrice(R1 + buf); // protection above R1
      if(InpPP_TPToNextPivot)
         tpOut = NormalizePrice(S1 + buf); // near S1 (just north)
      return true;
   }
   return false;
}

//-------------------- Trading actions
bool OpenTrade(bool isBuy, double sl0, double hardTP)
{
   double entry = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // Stops-level clamp
   int stopsPts = StopsLevelPoints();
   double minDist = stopsPts * _Point;

   if(isBuy && (entry - sl0) < minDist) sl0 = entry - minDist;
   if(!isBuy && (sl0 - entry) < minDist) sl0 = entry + minDist;

   entry = NormalizePrice(entry);
   sl0   = NormalizePrice(sl0);

   double R = MathAbs(entry - sl0);
   if(R <= 0) return false;

   double tp1 = NormalizePrice(entry + (isBuy? +1 : -1) * R);
   double tpNext = NormalizePrice(entry + (isBuy? +1 : -1) * (InpTPNextR * R));

   double tpToUse = 0.0;
   if(InpUseHardTP)
   {
      // Secker basic mode: take profit at 1R
      tpToUse = tp1;
   }
   else if(hardTP > 0.0)
   {
      // Power Pivots optional mode: hard TP to pivot target
      tpToUse = hardTP;
   }

   trade.SetExpertMagicNumber(MagicFor(InpStrategy));
   trade.SetDeviationInPoints(20);

   bool ok = false;
   if(isBuy) ok = trade.Buy(InpFixedLots, _Symbol, 0.0, sl0, tpToUse, StrategyName(InpStrategy));
   else      ok = trade.Sell(InpFixedLots, _Symbol, 0.0, sl0, tpToUse, StrategyName(InpStrategy));

   if(!ok) return false;

   // Refresh position ticket
   ulong ticket=0;
   if(!HasOpenPositionByMagic(MagicFor(InpStrategy), ticket)) return false;

   // fill state
   S.hasPos = true;
   S.ticket = ticket;
   S.entryTime = TimeCurrent();
   S.entry = entry;
   S.sl0 = sl0;
   S.R = R;
   S.tp1 = tp1;
   S.tpNext = tpNext;
   // if hard tp at 1R, BE isn't meaningful; but we keep beSet false
   S.beSet = false;
   S.lastTrailPivotTime = 0;

   // Log
   LogEvent("entry", InpFixedLots, S.entry, S.sl0, S.tp1, S.tpNext, entry, "v1");

   return true;
}

bool ClosePosition(const string &reason)
{
   if(!S.hasPos) return false;
   if(!PositionSelectByTicket(S.ticket)) return false;
   double vol = PositionGetDouble(POSITION_VOLUME);
   double price = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                                                        : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool ok = trade.PositionClose(_Symbol);
   if(ok)
   {
      LogEvent(reason, vol, S.entry, PositionGetDouble(POSITION_SL), S.tp1, S.tpNext, price, "");
      S.hasPos=false;
      S.ticket=0;
      S.beSet=false;
   }
   return ok;
}

void UpdateManagement(bool newBar)
{
   // Refresh position state
   ulong ticket=0;
   if(!HasOpenPositionByMagic(MagicFor(InpStrategy), ticket))
   {
      S.hasPos = false;
      S.ticket = 0;
      S.beSet = false;
   S.lastTrailPivotTime = 0;
      return;
   }
   S.hasPos = true;
   S.ticket = ticket;

   if(!PositionSelectByTicket(ticket)) return;

   bool isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl = PositionGetDouble(POSITION_SL);
   double vol = PositionGetDouble(POSITION_VOLUME);

   // If we lost state (restart), reconstruct key values
   if(S.entry <= 0.0 || MathAbs(S.entry-entry) > 10*_Point)
   {
      S.entry = entry;
      S.sl0 = sl;
      S.R = MathAbs(entry - sl);
      if(S.R <= 0) S.R = MathMax(GetATR(1), _Point);
      S.tp1 = NormalizePrice(entry + (isBuy? +1 : -1) * S.R);
      S.tpNext = NormalizePrice(entry + (isBuy? +1 : -1) * (InpTPNextR * S.R));
   }

   // BE detection (intrabar touch of TP1 reference)
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool tpTouched = isBuy ? (bid >= S.tp1) : (ask <= S.tp1);

   // Determine if BE already effectively set
   if(!S.beSet)
   {
      if(MathAbs(sl - entry) <= 2*_Point) S.beSet = true;
   }

   if(!S.beSet && tpTouched)
   {
      // Set SL to BE (entry)
      double newSL = NormalizePrice(entry);
      // Ensure stop distance rules
      int stopsPts = StopsLevelPoints();
      double minDist = stopsPts * _Point;
      if(isBuy && (bid - newSL) < minDist) newSL = bid - minDist;
      if(!isBuy && (newSL - ask) < minDist) newSL = ask + minDist;

      if(trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP)))
      {
         S.beSet = true;
         LogEvent("be_set", vol, entry, newSL, S.tp1, S.tpNext, (isBuy? bid:ask), "");
      }
   }

      // Structural trailing after BE (bar-close pivot events, no ATR trail)
   if(S.beSet && !InpUseHardTP) // only meaningful if we are not closing at 1R
   {
      if(!newBar) return; // only update structure trail on bar boundaries

      double atr = GetATR(1);
      double buf = atr * InpSLBufferATR;

      double pivot=0.0;
      datetime pivT=0;
      bool okPivot = false;

      if(isBuy) okPivot = GetLastZigZagPivot(true, InpZZ_Lookback, InpZZ_ATRMult, pivot, pivT);
      else      okPivot = GetLastZigZagPivot(false, InpZZ_Lookback, InpZZ_ATRMult, pivot, pivT);

      bool pivotEvent = okPivot && (pivT != 0) && (pivT != S.lastTrailPivotTime);

      if(!pivotEvent) return; // no new confirmed swing -> no trail update

      double candidate = sl;

      // 1) Structural stop behind last confirmed swing
      double structural = isBuy ? (pivot - buf) : (pivot + buf);
      if(isBuy) candidate = MathMax(candidate, structural);
      else      candidate = MathMin(candidate, structural);

      // 2) RR-consistent tightening (optional) — applied on pivot events only
      if(InpUseRRTighten && InpRRTarget > 0.0)
      {
         double spot = isBuy ? bid : ask;
         double D = MathAbs(S.tpNext - spot);
         double allowedRisk = D / InpRRTarget;
         double rrStop = isBuy ? (spot - allowedRisk) : (spot + allowedRisk);
         if(isBuy) candidate = MathMax(candidate, rrStop);
         else      candidate = MathMin(candidate, rrStop);
      }

      // Clamp stop to correct side with stop-level constraints
      int stopsPts = StopsLevelPoints();
      double minDist = stopsPts * _Point;

      if(isBuy)
      {
         candidate = MathMin(candidate, bid - minDist);
         candidate = MathMin(candidate, bid - _Point);
         if(candidate > sl + _Point)
         {
            candidate = NormalizePrice(candidate);
            if(trade.PositionModify(_Symbol, candidate, PositionGetDouble(POSITION_TP)))
            {
               S.lastTrailPivotTime = pivT;
               LogEvent("trail_update", vol, entry, candidate, S.tp1, S.tpNext, bid, "");
            }
         }
      }
      else
      {
         candidate = MathMax(candidate, ask + minDist);
         candidate = MathMax(candidate, ask + _Point);
         if(candidate < sl - _Point)
         {
            candidate = NormalizePrice(candidate);
            if(trade.PositionModify(_Symbol, candidate, PositionGetDouble(POSITION_TP)))
            {
               S.lastTrailPivotTime = pivT;
               LogEvent("trail_update", vol, entry, candidate, S.tp1, S.tpNext, ask, "");
            }
         }
      }
   }
}

void EnforceVolRevExpiry()
{
   if(InpStrategy != STRAT_VOLREV) return;
   if(!S.hasPos) return;
   if(!PositionSelectByTicket(S.ticket)) return;

   // If position has been open for >= InpVR_ExpiryBars full bars, close at bar close.
   datetime entryT = S.entryTime;
   int shift = iBarShift(_Symbol, PERIOD_CURRENT, entryT, true);
   if(shift < 0) return;

   if(shift >= InpVR_ExpiryBars)
   {
      ClosePosition("time_exit");
   }
}

//-------------------- Lifecycle
int OnInit()
{
   hATR = iATR(_Symbol, PERIOD_CURRENT, InpATRLen);
      hEmaFastCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);

   hEmaFastHTF = iMA(_Symbol, Inp180_HTF, Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowHTF = iMA(_Symbol, Inp180_HTF, Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);

   hEmaBiasD1 = iMA(_Symbol, PERIOD_D1, InpPP_BiasEMA, 0, MODE_EMA, PRICE_CLOSE);

   // 180PC regime filter (optional)
   hADX = iADX(_Symbol, PERIOD_CURRENT, Inp180_ADXLen);

   g_lastBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

   // CSV header (write once if file empty)
   if(InpWriteCSV)
   {
      int fh = FileOpen(InpCSVFileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_WRITE, ',');
      if(fh != INVALID_HANDLE)
      {
         if(FileSize(fh) == 0)
            FileWrite(fh, "timestamp_utc","symbol","timeframe","strategy","position_id","magic","direction","event","qty","entry_price","sl_price","tp1_price","tpnext_price","event_price","comment");
         FileClose(fh);
      }
   }

   // Load existing position state if any
   ulong t=0;
   if(HasOpenPositionByMagic(MagicFor(InpStrategy), t))
   {
      S.hasPos=true; S.ticket=t;
      PositionSelectByTicket(t);
      S.entry = PositionGetDouble(POSITION_PRICE_OPEN);
      S.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
      S.sl0 = PositionGetDouble(POSITION_SL);
      S.R = MathAbs(S.entry - S.sl0);
      S.tp1 = NormalizePrice(S.entry + ((PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)? +1 : -1) * S.R);
      S.tpNext = NormalizePrice(S.entry + ((PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)? +1 : -1) * (InpTPNextR * S.R));
      S.beSet = (MathAbs(S.sl0 - S.entry) <= 2*_Point);
      S.lastTrailPivotTime = 0;
   }
   else
   {
      S.hasPos=false; S.ticket=0; S.beSet=false;
   }

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   // nothing
}

void OnTick()
{
   // Compute new-bar once (important for correctness)
   bool newBar = IsNewBar();

   // Always manage open positions (BE intrabar, structure trail on bar-close)
   UpdateManagement(newBar);

   // Expiry for VolRev (bar-close)
   if(newBar)
      EnforceVolRevExpiry();

   // Entry logic on bar close only (deterministic)
   if(!newBar) return;

   ulong t=0;
   if(HasOpenPositionByMagic(MagicFor(InpStrategy), t)) return; // one at a time

   bool isBuy=false;
   double slOut=0.0;
   double hardTP=0.0;

   bool sig=false;
   if(InpStrategy==STRAT_180PC)
      sig = Signal_180PC(isBuy, slOut);
   else if(InpStrategy==STRAT_TWAVE)
      sig = Signal_TWave(isBuy, slOut);
   else if(InpStrategy==STRAT_VOLREV)
      sig = Signal_VolRev(isBuy, slOut);
   else if(InpStrategy==STRAT_POWERPIVOTS)
      sig = Signal_PowerPivots(isBuy, slOut, hardTP);

   if(sig)
      OpenTrade(isBuy, slOut, hardTP);
}
