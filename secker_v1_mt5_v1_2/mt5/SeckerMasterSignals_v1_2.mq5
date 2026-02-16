//+------------------------------------------------------------------+
//| SeckerMasterSignals_v1_2.mq5                                     |
//| MT5 Indicator: visual signals + projected SL/TP(1R) levels        |
//| v1.2: 180PC Ring Low/High + VolRev false-break reversal,          |
//|      SL/TP dots on signal bar, thin blue TP line, yellow live SL  |
//+------------------------------------------------------------------+
#property strict
#property indicator_chart_window
#property indicator_plots 5

//--- plot 0: buy arrow
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2
#property indicator_label1  "Buy"
//--- plot 1: sell arrow
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrDodgerBlue
#property indicator_width2  2
#property indicator_label2  "Sell"
//--- plot 2: SL dot
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrRed
#property indicator_width3  1
#property indicator_label3  "SL"
//--- plot 3: TP dot
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrLime
#property indicator_width4  1
#property indicator_label4  "TP1"
//--- plot 4: Trailing/Live SL dot
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrYellow
#property indicator_width5  2
#property indicator_label5  "LiveSL"

#include <Trade/Trade.mqh>

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

// Visual + price offsets
input double           InpPipOffset           = 1.0;
input bool             InpDrawTPLines         = true;
input int              InpMaxTPLines          = 250;

// Common params
input int              InpATRLen              = 14;
input double           InpTPNextR             = 2.0;  // used for TP-line extension only

// 180PC
input ENUM_TIMEFRAMES  Inp180_HTF             = PERIOD_D1;
input int              Inp180_EMAFast         = 8;
input int              Inp180_EMASlow         = 20;
input bool             Inp180_UsePrevDailyColor = false;

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
input int              InpVR_ExpiryBars       = 1; // informational

// Power Pivots
input ENUM_TIMEFRAMES  InpPP_PivotTF          = PERIOD_D1;
input int              InpPP_BiasEMA          = 200;

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
int hEmaBiasD1  = INVALID_HANDLE;
int hADX        = INVALID_HANDLE;

//-------------------- Helpers
int MagicFor(StrategySel s) { return (int)(InpMagicBase + (int)s); }

double PipPoint()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits==3 || digits==5) return 10.0 * _Point;
   return _Point;
}

double PipsToPrice(double pips) { return pips * PipPoint(); }

double NormalizePrice(double p)
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(p, digits);
}

double GetATR(int shift)
{
   if(hATR==INVALID_HANDLE) return 0.0;
   double b[];
   if(CopyBuffer(hATR, 0, shift, 1, b) != 1) return 0.0;
   return b[0];
}

double GetMA(int handle, int shift)
{
   if(handle==INVALID_HANDLE) return 0.0;
   double b[];
   if(CopyBuffer(handle, 0, shift, 1, b) != 1) return 0.0;
   return b[0];
}

double GetADX(int shift)
{
   if(hADX==INVALID_HANDLE) return 0.0;
   double b[];
   if(CopyBuffer(hADX, 0, shift, 1, b) != 1) return 0.0;
   return b[0];
}

string Prefix()
{
   return "SeckerV1_" + IntegerToString(MagicFor(InpStrategy)) + "_";
}

void ClearOldTPLines()
{
   int total = ObjectsTotal(0, 0, -1);
   int kept = 0;
   for(int i=total-1; i>=0; --i)
   {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, Prefix()+"TP_") != 0) continue;
      kept++;
      if(kept > InpMaxTPLines)
         ObjectDelete(0, name);
   }
}

void DrawTPLine(datetime t, double price)
{
   if(!InpDrawTPLines) return;
   string name = Prefix()+"TP_"+IntegerToString((int)t);
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_TREND, 0, t, price, TimeCurrent(), price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrDodgerBlue);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
}

bool HasOpenPositionSL(int magic, double &slOut)
{
   for(int i=PositionsTotal()-1; i>=0; --i)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      slOut = PositionGetDouble(POSITION_SL);
      return (slOut > 0.0);
   }
   return false;
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

bool RegimeOK_180PC(int i)
{
   if(!Inp180_RegimeEnable) return true;
   double atr = GetATR(i);
   if(atr <= 0) atr = _Point;

   if(Inp180_RegimeMethod == REG_ADX)
      return (GetADX(i) >= Inp180_MinADX);

   int L = Inp180_SlopeLookback;
   if(L < 1) L = 1;
   double emaS_now  = GetMA(hEmaSlowCur, i);
   double emaS_past = GetMA(hEmaSlowCur, i + L);
   double slopeATR = MathAbs(emaS_now - emaS_past) / atr;
   return (slopeATR >= Inp180_MinSlopeATR);
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

//+------------------------------------------------------------------+
//| Lifecycle                                                        |
//+------------------------------------------------------------------+
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
   hEmaFastHTF = iMA(_Symbol, Inp180_HTF,      Inp180_EMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlowHTF = iMA(_Symbol, Inp180_HTF,      Inp180_EMASlow, 0, MODE_EMA, PRICE_CLOSE);
   hEmaBiasD1  = iMA(_Symbol, PERIOD_D1, InpPP_BiasEMA, 0, MODE_EMA, PRICE_CLOSE);
   hADX        = iADX(_Symbol, PERIOD_CURRENT, Inp180_ADXLen);

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
   if(rates_total < 50) return 0;

   // reset buffers
   for(int i=0;i<rates_total;i++)
   {
      BufBuy[i]=EMPTY_VALUE;
      BufSell[i]=EMPTY_VALUE;
      BufSL[i]=EMPTY_VALUE;
      BufTP[i]=EMPTY_VALUE;
      BufTrail[i]=EMPTY_VALUE;
   }

   double pip = PipsToPrice(InpPipOffset);

   // HTF agreement values (latest closed HTF bar)
   double emaF_h = GetMA(hEmaFastHTF, 1);
   double emaS_h = GetMA(hEmaSlowHTF, 1);

   // Optional prev daily candle color
   double d1O = iOpen(_Symbol, PERIOD_D1, 1);
   double d1C = iClose(_Symbol, PERIOD_D1, 1);

   // Iterate bars (series arrays: 0 is current, increasing is older)
   for(int i=rates_total-2; i>=2; --i)
   {
      bool buy=false, sell=false;
      double entry=0, sl=0, tp1=0;

      if(InpStrategy==STRAT_180PC)
      {
         double emaF = GetMA(hEmaFastCur, i);
         double emaS = GetMA(hEmaSlowCur, i);
         bool trendUp = (emaF_h > emaS_h) && (emaF > emaS);
         bool trendDn = (emaF_h < emaS_h) && (emaF < emaS);

         if(Inp180_UsePrevDailyColor)
         {
            if(trendUp && !(d1C > d1O)) trendUp=false;
            if(trendDn && !(d1C < d1O)) trendDn=false;
         }

         if(!RegimeOK_180PC(i)) { /*no-op*/ }
         else
         {
            bool ringLow  = (low[i] < low[i+1]) && (high[i] < high[i+1]);
            bool ringHigh = (high[i] > high[i+1]) && (low[i] > low[i+1]);

            if(trendUp && ringLow)
            {
               buy=true;
               entry = high[i] + pip;
               sl = low[i] - pip;
               tp1 = entry + (entry - sl);
            }
            if(trendDn && ringHigh)
            {
               sell=true;
               entry = low[i] - pip;
               sl = high[i] + pip;
               tp1 = entry - (sl - entry);
            }
         }
      }
      else if(InpStrategy==STRAT_TWAVE)
      {
         double atr = GetATR(i);
         if(IsBullPin(open[i],high[i],low[i],close[i],atr))
         {
            buy=true;
            entry = close[i];
            sl = low[i] - pip;
            tp1 = entry + (entry - sl);
         }
         if(IsBearPin(open[i],high[i],low[i],close[i],atr))
         {
            sell=true;
            entry = close[i];
            sl = high[i] + pip;
            tp1 = entry - (sl - entry);
         }
      }
      else if(InpStrategy==STRAT_VOLREV)
      {
         bool buySig  = (low[i] < low[i+1]) && (close[i] > low[i+1]);
         bool sellSig = (high[i] > high[i+1]) && (close[i] < high[i+1]);
         if(buySig)
         {
            buy=true;
            entry = high[i] + pip;
            sl = low[i] - pip;
            tp1 = entry + (entry - sl);
         }
         if(sellSig)
         {
            sell=true;
            entry = low[i] - pip;
            sl = high[i] + pip;
            tp1 = entry - (sl - entry);
         }
      }
      else if(InpStrategy==STRAT_POWERPIVOTS)
      {
         double emaBias = GetMA(hEmaBiasD1, 1);
         double dClose = iClose(_Symbol, PERIOD_D1, 1);
         bool biasLong  = (dClose >= emaBias);
         bool biasShort = !biasLong;

         double P,R1,S1;
         ComputeDailyPivots(P,R1,S1);

         bool crossUp = (close[i+1] <= P) && (close[i] > P);
         bool crossDn = (close[i+1] >= P) && (close[i] < P);

         double atr = GetATR(i);
         double buf = atr * 0.10;

         if(biasLong && crossUp)
         {
            buy=true;
            entry = close[i];
            sl = S1 - buf;
            tp1 = entry + (entry - sl);
         }
         if(biasShort && crossDn)
         {
            sell=true;
            entry = close[i];
            sl = R1 + buf;
            tp1 = entry - (sl - entry);
         }
      }

      if(buy)
      {
         BufBuy[i] = low[i] - (2.0*_Point);
         BufSL[i]  = NormalizePrice(sl);
         BufTP[i]  = NormalizePrice(tp1);
         DrawTPLine(time[i], NormalizePrice(tp1));
      }
      if(sell)
      {
         BufSell[i] = high[i] + (2.0*_Point);
         BufSL[i]   = NormalizePrice(sl);
         BufTP[i]   = NormalizePrice(tp1);
         DrawTPLine(time[i], NormalizePrice(tp1));
      }
   }

   // Live SL dot (yellow) aligned to current candle
   double liveSL=0.0;
   if(HasOpenPositionSL(MagicFor(InpStrategy), liveSL))
      BufTrail[0] = NormalizePrice(liveSL);

   ClearOldTPLines();
   return rates_total;
}
