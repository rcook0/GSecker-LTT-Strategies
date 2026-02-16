//+------------------------------------------------------------------+
//| SeckerMasterSignals_v1_1.mq5                                        |
//| v1.1: 180PC, T-Wave, Volatility Reversal, Power Pivots              |
//| Visual indicator: arrows + SL/TP dots + (simulated) BE/trail dot. |
//+------------------------------------------------------------------+
#property strict
#property indicator_chart_window
#property indicator_plots 5

// Plot 1: Buy arrow
#property indicator_label1  "Buy"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

// Plot 2: Sell arrow
#property indicator_label2  "Sell"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrDodgerBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

// Plot 3: SL dot
#property indicator_label3  "SL"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrRed
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

// Plot 4: TP dot
#property indicator_label4  "TP"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrLime
#property indicator_style4  STYLE_SOLID
#property indicator_width4  1

// Plot 5: BE/Trail dot
#property indicator_label5  "BE/Trail"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrYellow
#property indicator_style5  STYLE_SOLID
#property indicator_width5  1

#include <Trade/Trade.mqh>

//-------------------- Inputs
enum StrategySel
{
   STRAT_180PC = 0,
   STRAT_TWAVE = 1,
   STRAT_VOLREV = 2,
   STRAT_POWERPIVOTS = 3
};

input StrategySel      InpStrategy           = STRAT_180PC;

// Shared
input bool             InpUseHardTP           = false;   // if true: sim exits at TP1(=1R) (Secker basic)
input double           InpTPNextR             = 2.0;     // reference TP for RR tightening
input double           InpRRTarget            = 1.0;
input int              InpATRLen              = 14;
input double           InpSLBufferATR         = 0.10;
input bool             InpUseRRTighten        = true;

// ZigZag structure (ATR*N)
input double           InpZZ_ATRMult          = 1.50;

input bool             InpDrawTPLines         = true;
input int              InpMaxTPLines          = 150;

// 180PC
input ENUM_TIMEFRAMES  Inp180_HTF             = PERIOD_D1;
input int              Inp180_EMAFast         = 8;
input int              Inp180_EMASlow         = 20;

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
input int              InpVR_ExpiryBars       = 1;
input double           InpVR_BufferATR        = 0.10;

// Power Pivots
input ENUM_TIMEFRAMES  InpPP_PivotTF          = PERIOD_D1;
input int              InpPP_BiasEMA          = 200;
input bool             InpPP_TPToNextPivot    = false;

//-------------------- Buffers
double BufBuy[];
double BufSell[];
double BufSL[];
double BufTP[];
double BufTrail[];

//-------------------- Handles
int hATR = INVALID_HANDLE;
int hEmaFastCur = INVALID_HANDLE, hEmaSlowCur = INVALID_HANDLE;
int hEmaFastHTF = INVALID_HANDLE, hEmaSlowHTF = INVALID_HANDLE;
int hEmaBiasD1 = INVALID_HANDLE;
int hADX = INVALID_HANDLE;

string Prefix()
{
   return "Secker_v1_1_";
}

string StrategyName()
{
   if(InpStrategy==STRAT_180PC) return "180PC";
   if(InpStrategy==STRAT_TWAVE) return "T-Wave";
   if(InpStrategy==STRAT_VOLREV) return "Volatility Reversal";
   if(InpStrategy==STRAT_POWERPIVOTS) return "Power Pivots";
   return "Unknown";
}

double GetATR(int shift)
{
   double b[];
   if(hATR==INVALID_HANDLE) return 0.0;
   if(CopyBuffer(hATR, 0, shift, 1, b)!=1) return 0.0;
   return b[0];
}

double GetMA(int handle, int shift)
{
   double b[];
   if(handle==INVALID_HANDLE) return 0.0;
   if(CopyBuffer(handle, 0, shift, 1, b)!=1) return 0.0;
   return b[0];
}

double GetADX(int shift)
{
   if(hADX==INVALID_HANDLE) return 0.0;
   double b[];
   if(CopyBuffer(hADX, 2, shift, 1, b)!=1) return 0.0;
   return b[0];
}

bool RegimeOK_180PC(int i)
{
   if(!Inp180_RegimeEnable) return true;
   double atr = GetATR(i);
   if(atr <= 0) atr = _Point;

   if(Inp180_RegimeMethod == REG_ADX)
   {
      double adx = GetADX(i);
      return (adx >= Inp180_MinADX);
   }
   else
   {
      int L = Inp180_SlopeLookback; if(L<1) L=1;
      double emaS_now  = GetMA(hEmaSlowCur, i);
      double emaS_past = GetMA(hEmaSlowCur, i+L);
      double slopeATR = MathAbs(emaS_now - emaS_past) / atr;
      return (slopeATR >= Inp180_MinSlopeATR);
   }
}

// ZigZag sim state used during OnCalculate loop
double gZZ_LastLow = 0.0;
double gZZ_LastHigh = 0.0;
int    gZZ_Dir = +1;
double gZZ_ExtremeH = 0.0; int gZZ_ExtremeHi = -1;
double gZZ_ExtremeL = 0.0; int gZZ_ExtremeLi = -1;

void ZZ_Reset(int oldestIndex, const double &high[], const double &low[])
{
   gZZ_Dir = +1;
   gZZ_LastLow = 0.0;
   gZZ_LastHigh = 0.0;
   gZZ_ExtremeH = high[oldestIndex]; gZZ_ExtremeHi = oldestIndex;
   gZZ_ExtremeL = low[oldestIndex];  gZZ_ExtremeLi = oldestIndex;
}

bool ZZ_Update(int i, const double &high[], const double &low[], double atr, double atrMult,
               bool &pivotEvent, bool &pivotIsLow, double &pivotPrice)
{
   pivotEvent=false; pivotIsLow=false; pivotPrice=0.0;
   double thr = atr * atrMult;
   if(thr <= 5*_Point) thr = 5*_Point;

   if(gZZ_Dir == +1)
   {
      if(high[i] > gZZ_ExtremeH) { gZZ_ExtremeH = high[i]; gZZ_ExtremeHi = i; }
      if((gZZ_ExtremeH - low[i]) >= thr)
      {
         // confirm swing HIGH
         gZZ_LastHigh = gZZ_ExtremeH;
         pivotEvent=true; pivotIsLow=false; pivotPrice=gZZ_ExtremeH;

         gZZ_Dir = -1;
         gZZ_ExtremeL = low[i]; gZZ_ExtremeLi = i;
         return true;
      }
   }
   else
   {
      if(low[i] < gZZ_ExtremeL) { gZZ_ExtremeL = low[i]; gZZ_ExtremeLi = i; }
      if((high[i] - gZZ_ExtremeL) >= thr)
      {
         // confirm swing LOW
         gZZ_LastLow = gZZ_ExtremeL;
         pivotEvent=true; pivotIsLow=true; pivotPrice=gZZ_ExtremeL;

         gZZ_Dir = +1;
         gZZ_ExtremeH = high[i]; gZZ_ExtremeHi = i;
         return true;
      }
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

bool IsBullPin(double o,double h,double l,double c,double atr)
{
   double body = MathAbs(c-o); if(body<=0) body=0.0000001;
   double upper = h - MathMax(c,o);
   double lower = MathMin(c,o) - l;
   if(lower < InpTW_WickBodyRatio * body) return false;
   if(lower < atr * InpTW_MinWickATR) return false;
   if(body  > atr * InpTW_MaxBodyATR) return false;
   if(c <= o) return false;
   if(upper > lower*0.6) return false;
   return true;
}
bool IsBearPin(double o,double h,double l,double c,double atr)
{
   double body = MathAbs(c-o); if(body<=0) body=0.0000001;
   double upper = h - MathMax(c,o);
   double lower = MathMin(c,o) - l;
   if(upper < InpTW_WickBodyRatio * body) return false;
   if(upper < atr * InpTW_MinWickATR) return false;
   if(body  > atr * InpTW_MaxBodyATR) return false;
   if(c >= o) return false;
   if(lower > upper*0.6) return false;
   return true;
}

void ClearOldTPLines()
{
   int total = ObjectsTotal(0, 0, -1);
   // crude cleanup: delete beyond max by matching prefix
   int kept=0;
   for(int i=total-1; i>=0; --i)
   {
      string name = ObjectName(0,i,0,-1);
      if(StringFind(name, Prefix()+"TP_") != 0) continue;
      kept++;
      if(kept > InpMaxTPLines)
         ObjectDelete(0,name);
   }
}

void DrawTPLine(datetime t, double price)
{
   if(!InpDrawTPLines) return;
   string name = Prefix()+"TP_"+IntegerToString((int)t);
   if(ObjectFind(0,name)>=0) return;
   ObjectCreate(0,name,OBJ_TREND,0,t,price,TimeCurrent(),price);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clrDodgerBlue);
   ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
   ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,true);
   ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_DOT);
}

// Simulation state (single position for visuals)
struct SimPos
{
   bool inTrade;
   bool isBuy;
   int  entry_i;       // index at entry (series index)
   double entry;
   double sl;
   double sl0;
   double R;
   double tp1;
   double tpNext;
   bool beSet;
};
SimPos P;

// Entry signals based on bar i (closed bar is i)
bool SignalAt(int i, const double &open[],const double &high[],const double &low[],const double &close[])
{
   // We use bar i as the signal bar (i>=2)
   if(i < 2) return false;

   double atr = GetATR(i);
   double buf = atr * InpSLBufferATR;

   // 180PC
   if(InpStrategy==STRAT_180PC)
   {
      double emaF_i = GetMA(hEmaFastCur, i);
      double emaS_i = GetMA(hEmaSlowCur, i);
      double emaF_prev = GetMA(hEmaFastCur, i+1);

      double emaFh = GetMA(hEmaFastHTF, 1);
      double emaSh = GetMA(hEmaSlowHTF, 1);

      bool trendUp = (emaFh > emaSh) && (emaF_i > emaS_i);
      bool trendDn = (emaFh < emaSh) && (emaF_i < emaS_i);

      bool crossUp = (close[i+1] <= emaF_prev) && (close[i] > emaF_i);
      bool crossDn = (close[i+1] >= emaF_prev) && (close[i] < emaF_i);

      if(!RegimeOK_180PC(i)) return false;

      if(trendUp && crossUp)
      {
         P.isBuy = true;
         // SL: previous confirmed swing low (ZigZag ATR*N), fallback to bar low
         P.sl0 = (gZZ_LastLow>0.0 ? (gZZ_LastLow - buf) : (low[i] - buf));
         return true;
      }
      if(trendDn && crossDn)
      {
         P.isBuy = false;
         P.sl0 = (gZZ_LastHigh>0.0 ? (gZZ_LastHigh + buf) : (high[i] + buf));
         return true;
      }
      return false;
   }

   // T-Wave
   if(InpStrategy==STRAT_TWAVE)
   {
      if(IsBullPin(open[i],high[i],low[i],close[i],atr))
      {
         P.isBuy=true;
         P.sl0 = low[i] - buf;
         return true;
      }
      if(IsBearPin(open[i],high[i],low[i],close[i],atr))
      {
         P.isBuy=false;
         P.sl0 = (gZZ_LastHigh>0.0 ? (gZZ_LastHigh + buf) : (high[i] + buf));
         return true;
      }
      return false;
   }

   // Volatility Reversal
   if(InpStrategy==STRAT_VOLREV)
   {
      double prevLow = low[i+1];
      double prevHigh = high[i+1];
      bool buySig = (low[i] < prevLow) && (close[i] > prevLow);
      bool sellSig = (high[i] > prevHigh) && (close[i] < prevHigh);
      double b = atr * InpVR_BufferATR;
      if(buySig)
      {
         P.isBuy=true;
         P.sl0 = low[i] - b;
         return true;
      }
      if(sellSig)
      {
         P.isBuy=false;
         P.sl0 = high[i] + b;
         return true;
      }
      return false;
   }

   // Power Pivots
   if(InpStrategy==STRAT_POWERPIVOTS)
   {
      double emaBias = GetMA(hEmaBiasD1, 1);
      double d1Close = iClose(_Symbol, PERIOD_D1, 1);
      bool biasLong = d1Close >= emaBias;
      bool biasShort = d1Close < emaBias;

      double Piv,R1,S1;
      ComputeDailyPivots(Piv,R1,S1);

      bool crossUp = (close[i+1] <= Piv) && (close[i] > Piv);
      bool crossDn = (close[i+1] >= Piv) && (close[i] < Piv);

      if(biasLong && crossUp)
      {
         P.isBuy=true;
         P.sl0 = S1 - buf;
         return true;
      }
      if(biasShort && crossDn)
      {
         P.isBuy=false;
         P.sl0 = R1 + buf;
         return true;
      }
      return false;
   }

   return false;
}

int OnInit()
{
   SetIndexBuffer(0, BufBuy, INDICATOR_DATA);
   SetIndexBuffer(1, BufSell, INDICATOR_DATA);
   SetIndexBuffer(2, BufSL, INDICATOR_DATA);
   SetIndexBuffer(3, BufTP, INDICATOR_DATA);
   SetIndexBuffer(4, BufTrail, INDICATOR_DATA);

   PlotIndexSetInteger(0, PLOT_ARROW, 233);
   PlotIndexSetInteger(1, PLOT_ARROW, 234);
   PlotIndexSetInteger(2, PLOT_ARROW, 159);
   PlotIndexSetInteger(3, PLOT_ARROW, 159);
   PlotIndexSetInteger(4, PLOT_ARROW, 159);

   ArraySetAsSeries(BufBuy, true);
   ArraySetAsSeries(BufSell, true);
   ArraySetAsSeries(BufSL, true);
   ArraySetAsSeries(BufTP, true);
   ArraySetAsSeries(BufTrail, true);

   hATR = iATR(_Symbol, PERIOD_CURRENT, InpATRLen);
      hEmaFastCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowCur = iMA(_Symbol, PERIOD_CURRENT, Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);
   hEmaFastHTF = iMA(_Symbol, Inp180_HTF, Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowHTF = iMA(_Symbol, Inp180_HTF, Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);
   hEmaBiasD1 = iMA(_Symbol, PERIOD_D1, InpPP_BiasEMA, 0, MODE_EMA, PRICE_CLOSE);
   hADX = iADX(_Symbol, PERIOD_CURRENT, Inp180_ADXLen);

   P.inTrade = false;

   return INIT_SUCCEEDED;
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < 300) return 0;

   // init buffers
   for(int i=0;i<rates_total;i++)
   {
      BufBuy[i]=EMPTY_VALUE;
      BufSell[i]=EMPTY_VALUE;
      BufSL[i]=EMPTY_VALUE;
      BufTP[i]=EMPTY_VALUE;
      BufTrail[i]=EMPTY_VALUE;
   }

   // Simulate from oldest->newest (series: highest index oldest)
   P.inTrade=false;
   P.beSet=false;

   // ZigZag init for this calculation window
int oldestIdx = maxBars-3;
ZZ_Reset(oldestIdx, high, low);

for(int i=maxBars-3; i>=2; --i)
   {
      // Update ZigZag state on this bar-close
      bool pivotEvent=false, pivotIsLow=false;
      double pivotPrice=0.0;
      double atr_i = GetATR(i);
      ZZ_Update(i, high, low, atr_i, InpZZ_ATRMult, pivotEvent, pivotIsLow, pivotPrice);

      // Expiry for VolRev
      if(P.inTrade && InpStrategy==STRAT_VOLREV)
      {
         int barsElapsed = P.entry_i - i;
         if(barsElapsed >= InpVR_ExpiryBars)
         {
            // exit at close of this bar
            P.inTrade=false;
         }
      }

      // Management update if in trade (simulate intrabar touch on bar i)
      if(P.inTrade)
      {
         // tp touch
         bool tpTouched = P.isBuy ? (high[i] >= P.tp1) : (low[i] <= P.tp1);

         if(!P.beSet && tpTouched)
         {
            P.sl = P.entry;
            P.beSet = true;
         }

         // structural trailing after BE: update only on favorable ZigZag pivot confirmations
if(P.beSet && !InpUseHardTP)
{
   double atr = GetATR(i);
   double buf = atr * InpSLBufferATR;

   // pivotEvent / pivotIsLow are produced by ZZ_Update() for this bar
   // Only trail on favorable pivot type (long -> low pivots, short -> high pivots)
   if(pivotEvent && ((P.isBuy && pivotIsLow) || (!P.isBuy && !pivotIsLow)))
   {
      double candidate = P.sl;

      double structural = P.isBuy ? (pivotPrice - buf) : (pivotPrice + buf);
      if(P.isBuy) candidate = MathMax(candidate, structural);
      else        candidate = MathMin(candidate, structural);

      if(InpUseRRTighten && InpRRTarget > 0.0)
      {
         double spot = close[i];
         double D = MathAbs(P.tpNext - spot);
         double allowedRisk = D / InpRRTarget;
         double rrStop = P.isBuy ? (spot - allowedRisk) : (spot + allowedRisk);
         if(P.isBuy) candidate = MathMax(candidate, rrStop);
         else        candidate = MathMin(candidate, rrStop);
      }

      P.sl = candidate;
   }
}

// Plot trail dot (yellow) on current candle if BE set (or trailing active)
 (yellow) on current candle if BE set (or trailing active)
         if(P.beSet)
            BufTrail[i] = P.sl;

         // Exit by stop hit (intrabar)
         bool slHit = P.isBuy ? (low[i] <= P.sl) : (high[i] >= P.sl);
         if(slHit)
         {
            P.inTrade=false;
         }
         else if(InpUseHardTP)
         {
            // basic mode: exit at TP1
            if(P.isBuy ? (high[i] >= P.tp1) : (low[i] <= P.tp1))
               P.inTrade=false;
         }
      }

      // Entry if flat
      if(!P.inTrade)
      {
         // signal at bar i (closed)
         if(SignalAt(i, open, high, low, close))
         {
            // entry assumed at next bar open (i-1) approximate
            int entryBar = i-1;
            if(entryBar < 1) continue;

            P.inTrade=true;
            P.beSet=false;
            P.entry_i=entryBar;
            P.entry = open[entryBar];
            P.sl = P.sl0;
            P.R = MathAbs(P.entry - P.sl0);
            if(P.R <= 0) { P.inTrade=false; continue; }
            P.tp1 = P.entry + (P.isBuy? +1 : -1) * P.R;
            P.tpNext = P.entry + (P.isBuy? +1 : -1) * (InpTPNextR * P.R);

            // Plot arrow + SL/TP dots on signal candle (i)
            if(P.isBuy) BufBuy[i] = low[i] - (GetATR(i)*0.05);
            else        BufSell[i] = high[i] + (GetATR(i)*0.05);

            BufSL[i] = P.sl0;
            BufTP[i] = P.tp1;

            // TP projected line
            if(InpDrawTPLines)
            {
               DrawTPLine(time[i], P.tp1);
               ClearOldTPLines();
            }
         }
      }
   }

   return rates_total;
}
