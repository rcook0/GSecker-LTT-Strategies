//+------------------------------------------------------------------+
//| SeckerMasterEA_v1_2.mq5                                          |
//| MT5 EA: 180PC, T-Wave, Volatility Reversal, Power Pivots          |
//| v1.2: Ring Low/High + VolRev stop entries, ATR*N ZigZag structure |
//|       BE trigger: intrabar touch of TP1 (1R)                      |
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

input StrategySel      InpStrategy            = STRAT_180PC;
input long             InpMagicBase           = 180100;
input double           InpFixedLots           = 0.10;

// Prices / offsets
input double           InpPipOffset           = 1.0;     // price offset in pips for stop entry + SL

// Core / shared management
input bool             InpUseHardTP           = false;   // if true: set broker TP at 1R (trade exits at 1R)
input double           InpTPNextR             = 2.0;     // reference for RR tightening after BE
input double           InpRRTarget            = 1.0;     // maintain approx RR from spot to TPNext
input int              InpATRLen              = 14;
input double           InpSLBufferATR         = 0.10;    // structural buffer = ATR * this
input bool             InpUseRRTighten        = true;    // after BE, tighten stop relative to remaining distance to TPNext

// ZigZag structure (ATR*N)
input double           InpZZ_ATRMult          = 1.50;    // N in ATR(14) * N
input int              InpZZ_Lookback         = 800;     // bars scanned to find last confirmed swing

// Pending orders
input bool             InpUsePending180       = true;
input int              Inp180_PendingExpiryBars = 2;
input bool             InpUsePendingVR        = true;
input int              InpVR_PendingExpiryBars  = 1;

// 180PC
input ENUM_TIMEFRAMES  Inp180_HTF             = PERIOD_D1;
input int              Inp180_EMAFast         = 8;
input int              Inp180_EMASlow         = 20;
input bool             Inp180_UsePrevDailyColor = false; // optional filter

enum SLMode180 { SL_RING_BAR = 0, SL_STRUCTURE = 1 };
input SLMode180        Inp180_SLMode          = SL_RING_BAR;

enum RegimeMethod { REG_SLOPE_ATR = 0, REG_ADX = 1 };
input bool             Inp180_RegimeEnable    = true;
input RegimeMethod     Inp180_RegimeMethod    = REG_SLOPE_ATR;
input int              Inp180_SlopeLookback   = 5;
input double           Inp180_MinSlopeATR     = 0.15;
input int              Inp180_ADXLen          = 14;
input double           Inp180_MinADX          = 18.0;

// T-Wave
input double           InpTW_WickBodyRatio    = 2.5;
input double           InpTW_MinWickATR       = 0.50;
input double           InpTW_MaxBodyATR       = 0.60;

// Volatility Reversal
input int              InpVR_ExpiryBars       = 1;       // hold for 1 bar if not exited

// Power Pivots
input ENUM_TIMEFRAMES  InpPP_PivotTF          = PERIOD_D1;
input int              InpPP_BiasEMA          = 200;
input bool             InpPP_TPToNextPivot    = false;

// Logging
input bool             InpWriteCSV            = true;
input string           InpCSVFileName         = "secker_v1_ledger.csv";

//-------------------- Handles
int hATR = INVALID_HANDLE;
int hEmaFastCur = INVALID_HANDLE, hEmaSlowCur = INVALID_HANDLE;
int hEmaFastHTF = INVALID_HANDLE, hEmaSlowHTF = INVALID_HANDLE;
int hEmaBiasD1  = INVALID_HANDLE;
int hADX        = INVALID_HANDLE;

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

//+------------------------------------------------------------------+
//| Utilities                                                        |
//+------------------------------------------------------------------+
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

double PipPoint()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits==3 || digits==5) return 10.0 * _Point;
   return _Point;
}

double PipsToPrice(double pips)
{
   return pips * PipPoint();
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

double GetADX(int shift=1)
{
   if(hADX==INVALID_HANDLE) return 0.0;
   double b[];
   if(CopyBuffer(hADX, 0, shift, 1, b) != 1) return 0.0;
   return b[0];
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

//+------------------------------------------------------------------+
//| ZigZag (ATR*N) structure (non-repainting swing confirmation)     |
//+------------------------------------------------------------------+
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

   int dir = +1; // seek swing high first
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

   int L = Inp180_SlopeLookback;
   if(L < 1) L = 1;
   double emaS_now  = GetMA(hEmaSlowCur, 1);
   double emaS_past = GetMA(hEmaSlowCur, 1 + L);
   double slopeATR = MathAbs(emaS_now - emaS_past) / atr;
   return (slopeATR >= Inp180_MinSlopeATR);
}

//+------------------------------------------------------------------+
//| Pending orders helpers                                           |
//+------------------------------------------------------------------+
bool HasPendingOrderByMagic(int magic, ulong &ticket, ENUM_ORDER_TYPE &type, double &price, datetime &expiration)
{
   ticket = 0; type = ORDER_TYPE_BUY_STOP; price = 0.0; expiration = 0;
   for(int i=OrdersTotal()-1; i>=0; --i)
   {
      ulong tk = OrderGetTicket(i);
      if(tk == 0) continue;
      if(!OrderSelect(tk)) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != magic) continue;

      ENUM_ORDER_TYPE t = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(t != ORDER_TYPE_BUY_STOP && t != ORDER_TYPE_SELL_STOP)
         continue;

      ticket = tk;
      type = t;
      price = OrderGetDouble(ORDER_PRICE_OPEN);
      expiration = (datetime)OrderGetInteger(ORDER_TIME_EXPIRATION);
      return true;
   }
   return false;
}

bool CancelPendingOrder(ulong ticket)
{
   if(ticket == 0) return false;
   return trade.OrderDelete(ticket);
}

bool PlacePendingStop(bool isBuy, double entryStop, double sl0, double tp, int expiryBars, const string &comment, ulong &outTicket)
{
   outTicket = 0;

   entryStop = NormalizePrice(entryStop);
   sl0       = NormalizePrice(sl0);
   tp        = NormalizePrice(tp);

   // Stops-level clamp for pending price and SL distance
   int stopsPts = StopsLevelPoints();
   double minDist = stopsPts * _Point;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(isBuy)
   {
      if(entryStop < ask + minDist) entryStop = NormalizePrice(ask + minDist);
      if(entryStop - sl0 < minDist) sl0 = NormalizePrice(entryStop - minDist);
   }
   else
   {
      if(entryStop > bid - minDist) entryStop = NormalizePrice(bid - minDist);
      if(sl0 - entryStop < minDist) sl0 = NormalizePrice(entryStop + minDist);
   }

   trade.SetExpertMagicNumber(MagicFor(InpStrategy));
   trade.SetDeviationInPoints(20);

   datetime exp = 0;
   if(expiryBars > 0)
      exp = TimeCurrent() + (expiryBars * PeriodSeconds());

   bool ok = false;
   if(isBuy)
      ok = trade.BuyStop(InpFixedLots, entryStop, _Symbol, sl0, tp, (exp>0?ORDER_TIME_SPECIFIED:ORDER_TIME_GTC), exp, comment);
   else
      ok = trade.SellStop(InpFixedLots, entryStop, _Symbol, sl0, tp, (exp>0?ORDER_TIME_SPECIFIED:ORDER_TIME_GTC), exp, comment);

   if(!ok) return false;

   // Best-effort: find the most recent pending order for our magic
   ENUM_ORDER_TYPE t; double p; datetime e;
   ulong tk;
   if(HasPendingOrderByMagic(MagicFor(InpStrategy), tk, t, p, e))
      outTicket = tk;

   return true;
}

//+------------------------------------------------------------------+
//| CSV logging                                                      |
//+------------------------------------------------------------------+
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

void LogEventEx(ulong id, const string &direction, const string &event, double qty, double entry, double sl, double tp1, double tpNext, double eventPrice, const string &comment)
{
   datetime now = TimeGMT();
   string line = StringFormat("%s,%s,%s,%s,%I64u,%d,%s,%s,%.2f,%.5f,%.5f,%.5f,%.5f,%.5f,%s",
                              TimeToString(now, TIME_DATE|TIME_SECONDS),
                              _Symbol,
                              TFToString((ENUM_TIMEFRAMES)Period()),
                              StrategyName(InpStrategy),
                              (long)id,
                              MagicFor(InpStrategy),
                              direction,
                              event,
                              qty, entry, sl, tp1, tpNext, eventPrice, comment);
   WriteCSV(line);
}

//+------------------------------------------------------------------+
//| Signals                                                         |
//+------------------------------------------------------------------+
bool Signal_180PC(bool &isBuy, double &entryStop, double &slOut)
{
   // Uses Ring Low/High 2-bar pattern in direction of EMA(8/20) agreement on HTF+current.
   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;

   double emaF_cur = GetMA(hEmaFastCur, 1);
   double emaS_cur = GetMA(hEmaSlowCur, 1);
   double emaF_h   = GetMA(hEmaFastHTF, 1);
   double emaS_h   = GetMA(hEmaSlowHTF, 1);

   bool trendUp = (emaF_h > emaS_h) && (emaF_cur > emaS_cur);
   bool trendDn = (emaF_h < emaS_h) && (emaF_cur < emaS_cur);

   if(!RegimeOK_180PC()) return false;

   if(Inp180_UsePrevDailyColor)
   {
      double d1O = iOpen(_Symbol, PERIOD_D1, 1);
      double d1C = iClose(_Symbol, PERIOD_D1, 1);
      if(trendUp && !(d1C > d1O)) trendUp = false;
      if(trendDn && !(d1C < d1O)) trendDn = false;
   }

   // Ring Low: second bar has lower low and lower high than the first bar.
   bool ringLow  = (r[1].low  < r[2].low)  && (r[1].high < r[2].high);
   bool ringHigh = (r[1].high > r[2].high) && (r[1].low  > r[2].low);

   double pip = PipsToPrice(InpPipOffset);
   double atr = GetATR(1);
   double buf = atr * InpSLBufferATR;

   if(trendUp && ringLow)
   {
      isBuy = true;
      entryStop = NormalizePrice(r[1].high + pip);
      if(Inp180_SLMode == SL_STRUCTURE)
      {
         double swingLow=0.0; datetime t=0;
         if(GetLastZigZagPivot(true, InpZZ_Lookback, InpZZ_ATRMult, swingLow, t))
            slOut = NormalizePrice(swingLow - buf);
         else
            slOut = NormalizePrice(r[1].low - pip);
      }
      else
      {
         slOut = NormalizePrice(r[1].low - pip);
      }
      return true;
   }

   if(trendDn && ringHigh)
   {
      isBuy = false;
      entryStop = NormalizePrice(r[1].low - pip);
      if(Inp180_SLMode == SL_STRUCTURE)
      {
         double swingHigh=0.0; datetime t=0;
         if(GetLastZigZagPivot(false, InpZZ_Lookback, InpZZ_ATRMult, swingHigh, t))
            slOut = NormalizePrice(swingHigh + buf);
         else
            slOut = NormalizePrice(r[1].high + pip);
      }
      else
      {
         slOut = NormalizePrice(r[1].high + pip);
      }
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
   double pip = PipsToPrice(InpPipOffset);
   double atr = GetATR(1);

   if(IsBullPin(r[1], atr))
   {
      isBuy = true;
      slOut = NormalizePrice(r[1].low - pip);
      return true;
   }
   if(IsBearPin(r[1], atr))
   {
      isBuy = false;
      slOut = NormalizePrice(r[1].high + pip);
      return true;
   }
   return false;
}

bool Signal_VolRev(bool &isBuy, double &entryStop, double &slOut)
{
   // Buy: break prior bar low and close back above that prior low (false breakdown)
   // Sell: break prior bar high and close back below that prior high (false breakout)
   MqlRates r[3];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, r) != 3) return false;

   double prevLow  = r[2].low;
   double prevHigh = r[2].high;

   bool buySig  = (r[1].low  < prevLow)  && (r[1].close > prevLow);
   bool sellSig = (r[1].high > prevHigh) && (r[1].close < prevHigh);

   double pip = PipsToPrice(InpPipOffset);

   if(buySig)
   {
      isBuy = true;
      entryStop = NormalizePrice(r[1].high + pip);
      slOut = NormalizePrice(r[1].low - pip);
      return true;
   }
   if(sellSig)
   {
      isBuy = false;
      entryStop = NormalizePrice(r[1].low - pip);
      slOut = NormalizePrice(r[1].high + pip);
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
   double emaBias = GetMA(hEmaBiasD1, 1);
   double d1Close = iClose(_Symbol, PERIOD_D1, 1);
   bool biasLong  = d1Close >= emaBias;
   bool biasShort = d1Close <  emaBias;

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
      slOut = NormalizePrice(S1 - buf);
      if(InpPP_TPToNextPivot)
         tpOut = NormalizePrice(R1 - buf);
      return true;
   }
   if(biasShort && crossDn)
   {
      isBuy = false;
      slOut = NormalizePrice(R1 + buf);
      if(InpPP_TPToNextPivot)
         tpOut = NormalizePrice(S1 + buf);
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Trading actions                                                  |
//+------------------------------------------------------------------+
bool OpenMarketTrade(bool isBuy, double sl0, double hardTP)
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
      tpToUse = tp1;
   else if(hardTP > 0.0)
      tpToUse = hardTP;

   trade.SetExpertMagicNumber(MagicFor(InpStrategy));
   trade.SetDeviationInPoints(20);

   bool ok = false;
   if(isBuy) ok = trade.Buy(InpFixedLots, _Symbol, 0.0, sl0, tpToUse, StrategyName(InpStrategy));
   else      ok = trade.Sell(InpFixedLots, _Symbol, 0.0, sl0, tpToUse, StrategyName(InpStrategy));

   if(!ok) return false;

   ulong ticket=0;
   if(!HasOpenPositionByMagic(MagicFor(InpStrategy), ticket)) return false;

   // Fill state
   S.hasPos = true;
   S.ticket = ticket;
   S.entryTime = TimeCurrent();
   S.entry = entry;
   S.sl0 = sl0;
   S.R = R;
   S.tp1 = tp1;
   S.tpNext = tpNext;
   S.beSet = false;
   S.lastTrailPivotTime = 0;

   LogEventEx(ticket, (isBuy?"BUY":"SELL"), "entry", InpFixedLots, S.entry, S.sl0, S.tp1, S.tpNext, entry, "market");
   return true;
}

bool PlaceStopEntry(bool isBuy, double entryStop, double sl0, double hardTP, int expiryBars)
{
   // Compute 1R from planned entryStop
   double entry = NormalizePrice(entryStop);
   double sl    = NormalizePrice(sl0);
   double R = MathAbs(entry - sl);
   if(R <= 0) return false;

   double tp1 = NormalizePrice(entry + (isBuy? +1 : -1) * R);
   double tpNext = NormalizePrice(entry + (isBuy? +1 : -1) * (InpTPNextR * R));

   double tpToUse = 0.0;
   if(InpUseHardTP)
      tpToUse = tp1;
   else if(hardTP > 0.0)
      tpToUse = hardTP;

   ulong orderTicket=0;
   bool ok = PlacePendingStop(isBuy, entry, sl, tpToUse, expiryBars, StrategyName(InpStrategy), orderTicket);
   if(!ok) return false;

   LogEventEx(orderTicket, (isBuy?"BUY":"SELL"), "order_place", InpFixedLots, entry, sl, tp1, tpNext, entry, (StringFormat("expiry_bars=%d", expiryBars)));
   return true;
}

bool ClosePosition(const string &reason)
{
   if(!S.hasPos) return false;
   if(!PositionSelectByTicket(S.ticket)) return false;

   bool isBuy = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY);
   double vol = PositionGetDouble(POSITION_VOLUME);
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl = PositionGetDouble(POSITION_SL);
   double price = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   bool ok = trade.PositionClose(_Symbol);
   if(ok)
   {
      LogEventEx(S.ticket, (isBuy?"BUY":"SELL"), reason, vol, entry, sl, S.tp1, S.tpNext, price, "");
      S.hasPos=false; S.ticket=0; S.beSet=false; S.lastTrailPivotTime=0;
      S.entry=0; S.sl0=0; S.R=0; S.tp1=0; S.tpNext=0; S.entryTime=0;
   }
   return ok;
}

//+------------------------------------------------------------------+
//| Management                                                       |
//+------------------------------------------------------------------+
void UpdateManagement(bool newBar)
{
   ulong ticket=0;
   if(!HasOpenPositionByMagic(MagicFor(InpStrategy), ticket))
   {
      S.hasPos=false; S.ticket=0; S.beSet=false; S.lastTrailPivotTime=0;
      return;
   }
   S.hasPos=true; S.ticket=ticket;
   if(!PositionSelectByTicket(ticket)) return;

   bool isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl = PositionGetDouble(POSITION_SL);
   double vol = PositionGetDouble(POSITION_VOLUME);

   // Recover state after restart
   if(S.entry <= 0.0 || MathAbs(S.entry-entry) > 50*_Point)
   {
      S.entry = entry;
      S.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
      S.sl0 = sl;
      S.R = MathAbs(entry - sl);
      if(S.R <= 0) S.R = MathMax(GetATR(1), _Point);
      S.tp1 = NormalizePrice(entry + (isBuy? +1 : -1) * S.R);
      S.tpNext = NormalizePrice(entry + (isBuy? +1 : -1) * (InpTPNextR * S.R));
      S.beSet = (MathAbs(sl - entry) <= 2*_Point);
      S.lastTrailPivotTime = 0;
   }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // BE detection (intrabar touch of TP1 reference)
   bool tpTouched = isBuy ? (bid >= S.tp1) : (ask <= S.tp1);
   if(!S.beSet)
   {
      if(MathAbs(sl - entry) <= 2*_Point) S.beSet=true;
   }

   if(!S.beSet && tpTouched && !InpUseHardTP)
   {
      double newSL = NormalizePrice(entry);
      int stopsPts = StopsLevelPoints();
      double minDist = stopsPts * _Point;
      if(isBuy && (bid - newSL) < minDist) newSL = NormalizePrice(bid - minDist);
      if(!isBuy && (newSL - ask) < minDist) newSL = NormalizePrice(ask + minDist);

      if(trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP)))
      {
         S.beSet=true;
         LogEventEx(ticket, (isBuy?"BUY":"SELL"), "be_set", vol, entry, newSL, S.tp1, S.tpNext, (isBuy?bid:ask), "");
      }
   }

   // Structural trailing after BE (bar-close pivot events)
   if(S.beSet && !InpUseHardTP)
   {
      if(!newBar) return;

      double atr = GetATR(1);
      double buf = atr * InpSLBufferATR;

      double pivot=0.0; datetime pivT=0;
      bool okPivot = isBuy ? GetLastZigZagPivot(true, InpZZ_Lookback, InpZZ_ATRMult, pivot, pivT)
                           : GetLastZigZagPivot(false, InpZZ_Lookback, InpZZ_ATRMult, pivot, pivT);

      bool pivotEvent = okPivot && (pivT != 0) && (pivT != S.lastTrailPivotTime);
      if(!pivotEvent) return;

      double candidate = sl;
      double structural = isBuy ? (pivot - buf) : (pivot + buf);
      if(isBuy) candidate = MathMax(candidate, structural);
      else      candidate = MathMin(candidate, structural);

      if(InpUseRRTighten && InpRRTarget > 0.0)
      {
         double spot = isBuy ? bid : ask;
         double D = MathAbs(S.tpNext - spot);
         double allowedRisk = D / InpRRTarget;
         double rrStop = isBuy ? (spot - allowedRisk) : (spot + allowedRisk);
         if(isBuy) candidate = MathMax(candidate, rrStop);
         else      candidate = MathMin(candidate, rrStop);
      }

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
               LogEventEx(ticket, "BUY", "trail_update", vol, entry, candidate, S.tp1, S.tpNext, bid, "");
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
               LogEventEx(ticket, "SELL", "trail_update", vol, entry, candidate, S.tp1, S.tpNext, ask, "");
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

   datetime entryT = (datetime)PositionGetInteger(POSITION_TIME);
   int shift = iBarShift(_Symbol, PERIOD_CURRENT, entryT, true);
   if(shift < 0) return;

   if(shift >= InpVR_ExpiryBars)
      ClosePosition("time_exit");
}

//+------------------------------------------------------------------+
//| Lifecycle                                                        |
//+------------------------------------------------------------------+
int OnInit()
{
   hATR = iATR(_Symbol, PERIOD_CURRENT, InpATRLen);

   hEmaFastCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);
   hEmaFastHTF = iMA(_Symbol, Inp180_HTF,      Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowHTF = iMA(_Symbol, Inp180_HTF,      Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);

   hEmaBiasD1  = iMA(_Symbol, PERIOD_D1, InpPP_BiasEMA, 0, MODE_EMA, PRICE_CLOSE);

   hADX = iADX(_Symbol, PERIOD_CURRENT, Inp180_ADXLen);

   g_lastBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

   // CSV header if empty
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
      bool isBuy = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY);
      S.entry = PositionGetDouble(POSITION_PRICE_OPEN);
      S.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
      S.sl0 = PositionGetDouble(POSITION_SL);
      S.R = MathAbs(S.entry - S.sl0);
      if(S.R <= 0) S.R = MathMax(GetATR(1), _Point);
      S.tp1 = NormalizePrice(S.entry + (isBuy? +1 : -1) * S.R);
      S.tpNext = NormalizePrice(S.entry + (isBuy? +1 : -1) * (InpTPNextR * S.R));
      S.beSet = (MathAbs(S.sl0 - S.entry) <= 2*_Point);
      S.lastTrailPivotTime = 0;
   }
   else
   {
      S.hasPos=false; S.ticket=0; S.beSet=false; S.lastTrailPivotTime=0;
      S.entry=0; S.sl0=0; S.R=0; S.tp1=0; S.tpNext=0; S.entryTime=0;
   }

   // Ensure history is available for exit logging in OnTradeTransaction.
   HistorySelect(0, TimeCurrent());

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
}

void OnTick()
{
   bool newBar = IsNewBar();

   // Manage any open position
   UpdateManagement(newBar);

   // Clean up expired pending orders (best-effort)
   if(newBar)
   {
      ulong ot=0; ENUM_ORDER_TYPE type; double price; datetime exp;
      if(HasPendingOrderByMagic(MagicFor(InpStrategy), ot, type, price, exp))
      {
         if(exp > 0 && exp <= TimeCurrent())
         {
            if(CancelPendingOrder(ot))
               LogEventEx(ot, (type==ORDER_TYPE_BUY_STOP?"BUY":"SELL"), "order_cancel", InpFixedLots, price, 0, 0, 0, price, "expired");
         }
      }
   }

   // VolRev position expiry at bar close
   if(newBar)
      EnforceVolRevExpiry();

   // Entries on bar close only
   if(!newBar) return;

   ulong posTicket=0;
   if(HasOpenPositionByMagic(MagicFor(InpStrategy), posTicket)) return;

   // Pending order state
   ulong pendingTicket=0; ENUM_ORDER_TYPE pendingType; double pendingPrice; datetime pendingExp;
   bool hasPending = HasPendingOrderByMagic(MagicFor(InpStrategy), pendingTicket, pendingType, pendingPrice, pendingExp);

   bool isBuy=false;
   double slOut=0.0;
   double entryStop=0.0;
   double hardTP=0.0;

   bool sig=false;
   bool usePending=false;
   int expiryBars=0;

   if(InpStrategy==STRAT_180PC)
   {
      sig = Signal_180PC(isBuy, entryStop, slOut);
      usePending = InpUsePending180;
      expiryBars = Inp180_PendingExpiryBars;
   }
   else if(InpStrategy==STRAT_TWAVE)
   {
      sig = Signal_TWave(isBuy, slOut);
      usePending = false;
   }
   else if(InpStrategy==STRAT_VOLREV)
   {
      sig = Signal_VolRev(isBuy, entryStop, slOut);
      usePending = InpUsePendingVR;
      expiryBars = InpVR_PendingExpiryBars;
   }
   else if(InpStrategy==STRAT_POWERPIVOTS)
   {
      sig = Signal_PowerPivots(isBuy, slOut, hardTP);
      usePending = false;
   }

   if(!sig) return;

   // Replace any existing pending order with the new setup
   if(hasPending)
   {
      if(CancelPendingOrder(pendingTicket))
         LogEventEx(pendingTicket, (pendingType==ORDER_TYPE_BUY_STOP?"BUY":"SELL"), "order_cancel", InpFixedLots, pendingPrice, 0, 0, 0, pendingPrice, "replaced");
   }

   if(usePending)
      PlaceStopEntry(isBuy, entryStop, slOut, hardTP, expiryBars);
   else
      OpenMarketTrade(isBuy, slOut, hardTP);
}

string ExitEventFromReason(long reason)
{
   // Map the most common tester/live reasons to stable event names.
   // See ENUM_DEAL_REASON.
   switch((int)reason)
   {
      case DEAL_REASON_SL:   return "exit_sl";
      case DEAL_REASON_TP:   return "exit_tp";
      case DEAL_REASON_SO:   return "exit_stopout";
      case DEAL_REASON_CLIENT:
      case DEAL_REASON_MOBILE:
      case DEAL_REASON_WEB:  return "exit_manual";
      default:               return "exit_other";
   }
}

// Capture exits that occur via broker SL/TP/trailing.
// Without this, the CSV ledger would miss the majority of exits.
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(!InpWriteCSV) return;
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(trans.deal == 0) return;

   if(!HistoryDealSelect(trans.deal)) return;
   string sym = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
   if(sym != _Symbol) return;

   long magic = (long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
   if((int)magic != MagicFor(InpStrategy)) return;

   long entryType = (long)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   // We only log OUT deals here; entries and order placement are logged elsewhere.
   if(entryType != DEAL_ENTRY_OUT && entryType != DEAL_ENTRY_OUT_BY) return;

   long dealType = (long)HistoryDealGetInteger(trans.deal, DEAL_TYPE);
   double price  = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
   double vol    = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   long reason   = (long)HistoryDealGetInteger(trans.deal, DEAL_REASON);
   long posId    = (long)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);

   // Infer original position direction from the OUT deal side:
   // - closing a BUY position is a SELL deal
   // - closing a SELL position is a BUY deal
   bool origBuy = (dealType == DEAL_TYPE_SELL);
   string dir = origBuy ? "BUY" : "SELL";

   // Best-effort: use current state if it matches, otherwise rebuild minimal R.
   double entryPrice = S.entry;
   double slPrice    = S.sl0;
   double tp1Price   = S.tp1;
   double tpNext     = S.tpNext;

   // If state not available (EA restarted or other), approximate from history.
   if(entryPrice <= 0.0 || tp1Price <= 0.0)
   {
      // Recover entry price from the earliest IN deal for this position.
      HistorySelect(0, TimeCurrent());
      int n = HistoryDealsTotal();
      for(int k=n-1; k>=0 && k>n-400; --k)
      {
         ulong dk = HistoryDealGetTicket(k);
         if(dk == 0) continue;
         if((long)HistoryDealGetInteger(dk, DEAL_POSITION_ID) != posId) continue;
         if((long)HistoryDealGetInteger(dk, DEAL_MAGIC) != magic) continue;
         if((long)HistoryDealGetInteger(dk, DEAL_ENTRY) != DEAL_ENTRY_IN) continue;
         entryPrice = HistoryDealGetDouble(dk, DEAL_PRICE);
         break;
      }
      // SL/TP references may not be recoverable from deal history alone.
      slPrice = 0.0;
      tp1Price = 0.0;
      tpNext = 0.0;
   }

   string ev = ExitEventFromReason(reason);
   string comment = StringFormat("deal=%I64u reason=%d", (long)trans.deal, (int)reason);
   LogEventEx((ulong)posId, dir, ev, vol, entryPrice, slPrice, tp1Price, tpNext, price, comment);
}
