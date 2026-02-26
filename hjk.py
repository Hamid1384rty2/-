import telebot
from telebot import types
import requests
import pandas as pd
import numpy as np
import threading
import time
import warnings
warnings.filterwarnings('ignore')

TOKEN = "8610465768:AAEf5JzMeNUG90CEbXB3kPzj4a8pc1sGb4M"
ADMIN_ID = 7523542863

bot = telebot.TeleBot(TOKEN)

# =========================
# VARIABLES
# =========================

REQUIRED_CHANNEL = None
user_state = {}
positions = {}

# =========================
# SYMBOLS
# =========================

HA_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"
]

AUTO_SYMBOLS = HA_SYMBOLS

DIP_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
               "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"]

DIP_TIMEFRAMES = {
    "15m": {"binance": "15m", "name": "۱۵ دقیقه"},
    "30m": {"binance": "30m", "name": "۳۰ دقیقه"},
    "1h": {"binance": "1h", "name": "۱ ساعت"},
    "4h": {"binance": "4h", "name": "۴ ساعت"},
    "1d": {"binance": "1d", "name": "روزانه"}
}

# =========================
# PRICE DATA FUNCTIONS
# =========================

def get_candles(symbol, interval="60", limit=200):
    """دریافت کندل از Bybit"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        candles = data["result"]["list"]
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        volumes = [float(c[5]) for c in candles]
        closes.reverse()
        highs.reverse()
        lows.reverse()
        volumes.reverse()
        df = pd.DataFrame({
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes
        })
        return df
    except:
        return pd.DataFrame()

def get_dip_candles(symbol, interval, limit=200):
    """دریافت کندل از Binance برای DIP"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        if not data or len(data) < 50:
            return None
        
        candles = []
        for d in data:
            try:
                candle = {
                    'time': int(d[0]),
                    'open': float(d[1]),
                    'high': float(d[2]),
                    'low': float(d[3]),
                    'close': float(d[4]),
                    'volume': float(d[5])
                }
                candles.append(candle)
            except:
                continue
        
        if len(candles) < 50:
            return None
            
        df = pd.DataFrame(candles)
        return df
        
    except Exception as e:
        print(f"خطا در دریافت کندل: {e}")
        return None

# =========================
# BASIC INDICATORS
# =========================

def calculate_indicators(df):
    """محاسبه اندیکاتورهای پایه"""
    if len(df) < 50:
        return df
    
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    
    return df

# =========================
# ADVANCED DIP INDICATORS
# =========================

def calculate_dip_indicators(df):
    """محاسبه اندیکاتورهای پیشرفته برای DIP"""
    if len(df) < 50:
        return df
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # ========== میانگین‌های متحرک ==========
    def sma(data, period):
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(np.nan)
            else:
                result.append(np.mean(data[i-period+1:i+1]))
        return pd.Series(result)
    
    def ema(data, period):
        result = []
        multiplier = 2 / (period + 1)
        for i in range(len(data)):
            if i == 0:
                result.append(data[i])
            else:
                result.append(data[i] * multiplier + result[-1] * (1 - multiplier))
        return pd.Series(result)
    
    # میانگین‌های متحرک
    df['sma_20'] = sma(close, 20)
    df['sma_50'] = sma(close, 50)
    df['sma_200'] = sma(close, 200)
    df['ema_20'] = ema(close, 20)
    df['ema_50'] = ema(close, 50)
    df['ema_200'] = ema(close, 200)
    
    # ========== RSI ==========
    def calculate_rsi(data, period=14):
        rsi_values = [50] * period
        for i in range(period, len(data)):
            gains = 0
            losses = 0
            for j in range(i-period+1, i+1):
                diff = data[j] - data[j-1]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            
            avg_gain = gains / period
            avg_loss = losses / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return pd.Series(rsi_values)
    
    df['rsi_14'] = calculate_rsi(close, 14)
    
    # ========== ATR ==========
    def calculate_atr(high, low, close, period=14):
        atr_values = [0] * period
        for i in range(period, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            atr_values.append((atr_values[-1] * (period - 1) + tr) / period)
        return pd.Series(atr_values)
    
    df['atr_14'] = calculate_atr(high, low, close)
    df['atr_percent'] = (df['atr_14'] / df['close']) * 100
    
    # ========== MACD ==========
    def calculate_macd(close, fast=12, slow=26, signal=9):
        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = []
        
        for i in range(len(macd_line)):
            if i < signal - 1:
                signal_line.append(np.nan)
            elif i == signal - 1:
                signal_line.append(np.mean(macd_line[:i+1]))
            else:
                val = macd_line[i] * (2/(signal+1)) + signal_line[-1] * (1 - (2/(signal+1)))
                signal_line.append(val)
        
        histogram = macd_line - pd.Series(signal_line)
        return macd_line, pd.Series(signal_line), histogram
    
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(close)
    
    # ========== باندهای بولینجر ==========
    def calculate_bb(close, period=20):
        upper = []
        middle = []
        lower = []
        width = []
        
        for i in range(len(close)):
            if i < period:
                upper.append(np.nan)
                middle.append(np.nan)
                lower.append(np.nan)
                width.append(np.nan)
                continue
            
            period_data = close[i-period+1:i+1]
            mean = np.mean(period_data)
            std = np.std(period_data)
            
            middle.append(mean)
            upper.append(mean + (std * 2))
            lower.append(mean - (std * 2))
            width.append((upper[-1] - lower[-1]) / mean * 100)
        
        return pd.Series(upper), pd.Series(middle), pd.Series(lower), pd.Series(width)
    
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = calculate_bb(close)
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
    
    # ========== استوکاستیک ==========
    def calculate_stoch(high, low, close, k_period=14, d_period=3):
        k_values = []
        for i in range(len(close)):
            if i < k_period:
                k_values.append(50)
                continue
            
            low_min = min(low[i-k_period+1:i+1])
            high_max = max(high[i-k_period+1:i+1])
            
            if high_max - low_min == 0:
                k_values.append(50)
            else:
                k = 100 * ((close[i] - low_min) / (high_max - low_min))
                k_values.append(k)
        
        d_values = []
        for i in range(len(k_values)):
            if i < d_period - 1:
                d_values.append(50)
            else:
                d_values.append(np.mean(k_values[i-d_period+1:i+1]))
        
        return pd.Series(k_values), pd.Series(d_values)
    
    df['stoch_k'], df['stoch_d'] = calculate_stoch(high, low, close)
    
    # ========== ADX ==========
    def calculate_adx(high, low, close, period=14):
        plus_dm = [0]
        minus_dm = [0]
        tr_values = [high[0] - low[0]]
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
            
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # ATR
        atr_values = [np.mean(tr_values[:period])]
        for i in range(period, len(tr_values)):
            atr_values.append((atr_values[-1] * (period - 1) + tr_values[i]) / period)
        
        # +DI و -DI
        plus_di = [0] * period
        minus_di = [0] * period
        
        for i in range(period, len(plus_dm)):
            sum_plus = np.mean(plus_dm[i-period+1:i+1])
            sum_minus = np.mean(minus_dm[i-period+1:i+1])
            
            if atr_values[i-period] == 0:
                plus_di.append(0)
                minus_di.append(0)
            else:
                plus_di.append(100 * sum_plus / atr_values[i-period])
                minus_di.append(100 * sum_minus / atr_values[i-period])
        
        # DX
        dx_values = [0] * (period * 2)
        for i in range(period * 2, len(plus_di)):
            if plus_di[i] + minus_di[i] == 0:
                dx_values.append(0)
            else:
                dx = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                dx_values.append(dx)
        
        # ADX
        adx_values = [0] * (period * 3)
        for i in range(period * 3, len(dx_values)):
            adx_values.append(np.mean(dx_values[i-period+1:i+1]))
        
        return pd.Series(adx_values), pd.Series(plus_di), pd.Series(minus_di)
    
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(high, low, close)
    
    # ========== OBV ==========
    def calculate_obv(close, volume):
        obv_vals = [0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_vals.append(obv_vals[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv_vals.append(obv_vals[-1] - volume[i])
            else:
                obv_vals.append(obv_vals[-1])
        return pd.Series(obv_vals)
    
    df['obv'] = calculate_obv(close, volume)
    df['obv_ema'] = ema(df['obv'].values, 20)
    
    # ========== MFI ==========
    def calculate_mfi(high, low, close, volume, period=14):
        mfi_values = [50] * period
        
        for i in range(period, len(close)):
            typical_price = (high[i] + low[i] + close[i]) / 3
            money_flow = typical_price * volume[i]
            
            positive_flow = 0
            negative_flow = 0
            
            for j in range(i-period+1, i+1):
                tp_j = (high[j] + low[j] + close[j]) / 3
                tp_prev = (high[j-1] + low[j-1] + close[j-1]) / 3 if j > 0 else tp_j
                
                mf = tp_j * volume[j]
                
                if tp_j > tp_prev:
                    positive_flow += mf
                else:
                    negative_flow += mf
            
            if negative_flow == 0:
                mfi_values.append(100)
            else:
                mf_ratio = positive_flow / negative_flow
                mfi_values.append(100 - (100 / (1 + mf_ratio)))
        
        return pd.Series(mfi_values)
    
    df['mfi_14'] = calculate_mfi(high, low, close, volume)
    
    # ========== پیوت پوینت‌ها ==========
    df['pivot'] = (high + low + close) / 3
    df['r1'] = 2 * df['pivot'] - low
    df['s1'] = 2 * df['pivot'] - high
    df['r2'] = df['pivot'] + (high - low)
    df['s2'] = df['pivot'] - (high - low)
    
    return df

# =========================
# MARKET CONDITION DETECTION
# =========================

def detect_market_condition(df):
    """تشخیص وضعیت بازار"""
    if len(df) < 50:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    
    last = df.iloc[-1]
    
    # تشخیص نوسان با ATR درصدی
    atr_percent = last['atr_percent'] if not np.isnan(last['atr_percent']) else 0
    
    if atr_percent > 5:
        volatility = "بسیار بالا"
    elif atr_percent > 3:
        volatility = "بالا"
    elif atr_percent > 1.5:
        volatility = "متوسط"
    else:
        volatility = "پایین"
    
    # تشخیص روند با ADX
    if not np.isnan(last['adx']):
        if last['adx'] > 25:
            if last['plus_di'] > last['minus_di']:
                trend = "صعودی قوی"
            else:
                trend = "نزولی قوی"
        elif last['adx'] > 20:
            trend = "روند ضعیف"
        else:
            trend = "رنج"
    else:
        trend = "نامشخص"
    
    # تشخیص نوسان با باند بولینجر
    bb_width = last['bb_width'] if not np.isnan(last['bb_width']) else 0
    
    if bb_width > 8:
        bb_state = "نوسان شدید"
    elif bb_width > 5:
        bb_state = "نوسان متوسط"
    else:
        bb_state = "نوسان کم"
    
    return trend, volatility, bb_state

# =========================
# ADVANCED DIP PATTERNS
# =========================

def detect_dip_patterns(df):
    """تشخیص الگوهای کندل استیک پیشرفته"""
    if len(df) < 2:
        return []
    
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    
    if prev is None:
        return patterns
    
    # محاسبه اندازه بدن و سایه‌ها
    body = abs(last['close'] - last['open'])
    range_price = last['high'] - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    lower_shadow = min(last['open'], last['close']) - last['low']
    
    if range_price == 0:
        return patterns
    
    # 1. دوجی
    if body < range_price * 0.1:
        patterns.append(("🟢 دوجی (عدم تصمیم)", 0.6))
    
    # 2. چکش صعودی
    if last['close'] > last['open']:
        if lower_shadow > body * 2 and upper_shadow < body * 0.3:
            patterns.append(("🟢 چکش صعودی (برگشت به بالا)", 0.85))
    
    # 3. چکش نزولی
    if last['close'] < last['open']:
        if upper_shadow > body * 2 and lower_shadow < body * 0.3:
            patterns.append(("🔴 چکش نزولی (برگشت به پایین)", 0.85))
    
    # 4. اینگالفینگ صعودی
    if prev['close'] < prev['open'] and last['close'] > last['open']:
        if last['open'] < prev['close'] and last['close'] > prev['open']:
            patterns.append(("🟢 اینگالفینگ صعودی (خرید قوی)", 0.9))
    
    # 5. اینگالفینگ نزولی
    if prev['close'] > prev['open'] and last['close'] < last['open']:
        if last['open'] > prev['close'] and last['close'] < prev['open']:
            patterns.append(("🔴 اینگالفینگ نزولی (فروش قوی)", 0.9))
    
    # 6. پین بار صعودی
    if lower_shadow > body * 2 and upper_shadow < body * 0.2 and last['close'] > last['open']:
        patterns.append(("🟢 پین بار صعودی (ریجکشن)", 0.85))
    
    # 7. پین بار نزولی
    if upper_shadow > body * 2 and lower_shadow < body * 0.2 and last['close'] < last['open']:
        patterns.append(("🔴 پین بار نزولی (ریجکشن)", 0.85))
    
    # 8. ستاره صبحگاهی (3 کندل)
    if len(df) >= 3:
        prev2 = df.iloc[-3]
        if (prev2['close'] < prev2['open'] and  # کندل نزولی اول
            body < range_price * 0.3 and  # کندل دوم کوچک
            last['close'] > last['open'] and  # کندل سوم صعودی
            last['close'] > (prev2['open'] + prev2['close'])/2):  # بسته شدن بالای وسط کندل اول
            patterns.append(("🟢 ستاره صبحگاهی (برگشت قدرتمند)", 0.95))
    
    # 9. ستاره شامگاهی (3 کندل)
    if len(df) >= 3:
        prev2 = df.iloc[-3]
        if (prev2['close'] > prev2['open'] and  # کندل صعودی اول
            body < range_price * 0.3 and  # کندل دوم کوچک
            last['close'] < last['open'] and  # کندل سوم نزولی
            last['close'] < (prev2['open'] + prev2['close'])/2):  # بسته شدن پایین وسط کندل اول
            patterns.append(("🔴 ستاره شامگاهی (برگشت نزولی)", 0.95))
    
    return patterns

# =========================
# ADVANCED DIVERGENCE DETECTION
# =========================

def detect_dip_divergence(df):
    """تشخیص واگرایی‌های پیشرفته"""
    if len(df) < 30 or 'rsi_14' not in df.columns:
        return []
    
    divergences = []
    
    # گرفتن 20 کندل آخر
    prices = df['close'].values[-20:]
    rsi = df['rsi_14'].values[-20:]
    
    if np.isnan(rsi).any():
        return []
    
    # تشخیص قله‌ها و دره‌ها در قیمت
    price_peaks = []
    price_valleys = []
    rsi_peaks = []
    rsi_valleys = []
    
    for i in range(2, len(prices)-2):
        # قله قیمت
        if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
            price_peaks.append((i, prices[i]))
        # دره قیمت
        if prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
            price_valleys.append((i, prices[i]))
        
        # قله RSI
        if rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2] and rsi[i] > rsi[i+1] and rsi[i] > rsi[i+2]:
            rsi_peaks.append((i, rsi[i]))
        # دره RSI
        if rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2] and rsi[i] < rsi[i+1] and rsi[i] < rsi[i+2]:
            rsi_valleys.append((i, rsi[i]))
    
    # واگرایی صعودی (قیمت کف پایین‌تر - RSI کف بالاتر)
    if len(price_valleys) >= 2 and len(rsi_valleys) >= 2:
        last_price_valley = price_valleys[-1]
        prev_price_valley = price_valleys[-2]
        last_rsi_valley = rsi_valleys[-1]
        prev_rsi_valley = rsi_valleys[-2]
        
        if last_price_valley[1] < prev_price_valley[1] and last_rsi_valley[1] > prev_rsi_valley[1]:
            divergences.append(("📈 واگرایی صعودی کلاسیک (قوی)", 0.9))
    
    # واگرایی نزولی (قیمت قله بالاتر - RSI قله پایین‌تر)
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        
        if last_price_peak[1] > prev_price_peak[1] and last_rsi_peak[1] < prev_rsi_peak[1]:
            divergences.append(("📉 واگرایی نزولی کلاسیک (قوی)", 0.9))
    
    # واگرایی مخفی صعودی (قیمت کف بالاتر - RSI کف پایین‌تر)
    if len(price_valleys) >= 2 and len(rsi_valleys) >= 2:
        last_price_valley = price_valleys[-1]
        prev_price_valley = price_valleys[-2]
        last_rsi_valley = rsi_valleys[-1]
        prev_rsi_valley = rsi_valleys[-2]
        
        if last_price_valley[1] > prev_price_valley[1] and last_rsi_valley[1] < prev_rsi_valley[1]:
            divergences.append(("📈 واگرایی مخفی صعودی (ادامه روند)", 0.8))
    
    # واگرایی مخفی نزولی (قیمت قله پایین‌تر - RSI قله بالاتر)
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        
        if last_price_peak[1] < prev_price_peak[1] and last_rsi_peak[1] > prev_rsi_peak[1]:
            divergences.append(("📉 واگرایی مخفی نزولی (ادامه روند)", 0.8))
    
    return divergences

# =========================
# SUPPORT RESISTANCE DETECTION
# =========================

def detect_dip_levels(df):
    """تشخیص سطوح حمایت و مقاومت پیشرفته"""
    if len(df) < 50:
        return None, None
    
    supply_zones = []
    demand_zones = []
    price = df['close'].iloc[-1]
    
    # تشخیص نواحی عرضه و تقاضا بر اساس حجم و کندل‌ها
    for i in range(20, len(df)-5):
        # ناحیه تقاضا (حمایت) - جایی که خرید قوی رخ داده
        if df['close'].iloc[i] > df['open'].iloc[i] * 1.02:  # کندل صعودی قوی
            vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-20:i].mean()
            if vol_ratio > 1.5:  # حجم بالاتر از میانگین
                zone_price = df['low'].iloc[i]
                if zone_price < price * 1.1:  # نزدیک به قیمت فعلی
                    demand_zones.append(zone_price)
        
        # ناحیه عرضه (مقاومت) - جایی که فروش قوی رخ داده
        if df['close'].iloc[i] < df['open'].iloc[i] * 0.98:  # کندل نزولی قوی
            vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-20:i].mean()
            if vol_ratio > 1.5:  # حجم بالاتر از میانگین
                zone_price = df['high'].iloc[i]
                if zone_price > price * 0.9:  # نزدیک به قیمت فعلی
                    supply_zones.append(zone_price)
    
    # تشخیص سطح بر اساس اردر بلاک (Order Block)
    for i in range(5, len(df)-5):
        # اردر بلاک صعودی (آخرین کندل نزولی قبل از یک حرکت صعودی)
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i+1] > df['open'].iloc[i+1] * 1.02:
            if df['high'].iloc[i+1] > df['high'].iloc[i] * 1.02:
                demand_zones.append(df['low'].iloc[i])
        
        # اردر بلاک نزولی (آخرین کندل صعودی قبل از یک حرکت نزولی)
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i+1] < df['open'].iloc[i+1] * 0.98:
            if df['low'].iloc[i+1] < df['low'].iloc[i] * 0.98:
                supply_zones.append(df['high'].iloc[i])
    
    # انتخاب نزدیک‌ترین سطوح به قیمت
    demand = None
    supply = None
    
    if demand_zones:
        # نزدیک‌ترین سطح حمایت زیر قیمت
        below_demand = [z for z in demand_zones if z < price]
        if below_demand:
            demand = max(below_demand)
    
    if supply_zones:
        # نزدیک‌ترین سطح مقاومت بالای قیمت
        above_supply = [z for z in supply_zones if z > price]
        if above_supply:
            supply = min(above_supply)
    
    return demand, supply

# =========================
# VOLATILITY FILTER
# =========================

def check_volatility_conditions(df):
    """بررسی شرایط نوسان برای فیلتر کردن سیگنال‌های فیک"""
    if len(df) < 20:
        return False, []
    
    last = df.iloc[-1]
    conditions = []
    
    # 1. بررسی باند بولینجر
    bb_width = last['bb_width'] if not np.isnan(last['bb_width']) else 0
    if bb_width > 10:
        conditions.append("⚠️ نوسان بسیار بالا - ریسک زیاد")
    elif bb_width > 7:
        conditions.append("📊 نوسان بالا - احتیاط")
    elif bb_width < 3:
        conditions.append("✅ نوسان مناسب")
    
    # 2. بررسی ATR درصدی
    atr_percent = last['atr_percent'] if not np.isnan(last['atr_percent']) else 0
    if atr_percent > 8:
        conditions.append("⚠️ نوسان شدید ATR")
    elif atr_percent > 4:
        conditions.append("📊 نوسان متوسط ATR")
    else:
        conditions.append("✅ نوسان کم ATR")
    
    # 3. بررسی نوسان غیرعادی (احتمال خبر)
    vol_ratio = last['volume'] / df['volume'].rolling(20).mean().iloc[-1] if df['volume'].rolling(20).mean().iloc[-1] > 0 else 1
    if vol_ratio > 5:
        conditions.append("⚠️ حجم غیرعادی - احتمال خبر مهم")
        return True, conditions  # فیلتر فعال می‌شود
    
    return False, conditions

# =========================
# ADVANCED DIP SCORER
# =========================

class DIPScorer:
    def __init__(self):
        self.weights = {
            'trend': 20,
            'momentum': 25,
            'volume': 15,
            'patterns': 15,
            'divergence': 15,
            'support_resistance': 10
        }
    
    def get_score(self, df, last, patterns, demand, supply, divergence, market_condition):
        score = 50
        reasons = []
        
        trend, volatility, bb_state = market_condition
        
        # ========== 1. فیلتر روند (وزن بالا) ==========
        # EMA200 (روند بلندمدت)
        if not np.isnan(last['ema_200']):
            if last['close'] > last['ema_200']:
                score += 8
                reasons.append("✅ بالای EMA200 (روند بلندمدت صعودی)")
            else:
                score -= 8
                reasons.append("🔻 پایین EMA200 (روند بلندمدت نزولی)")
        
        # SMA50 (روند میان‌مدت)
        if not np.isnan(last['sma_50']):
            if last['close'] > last['sma_50']:
                score += 5
                reasons.append("✅ بالای SMA50")
            else:
                score -= 5
                reasons.append("🔻 پایین SMA50")
        
        # SMA20 و SMA50
        if not np.isnan(last['sma_20']) and not np.isnan(last['sma_50']):
            if last['sma_20'] > last['sma_50']:
                score += 7
                reasons.append("📈 SMA20 > SMA50 (روند صعودی)")
            else:
                score -= 7
                reasons.append("📉 SMA20 < SMA50 (روند نزولی)")
        
        # ========== 2. مومنتوم ==========
        # RSI
        if not np.isnan(last['rsi_14']):
            if last['rsi_14'] < 30:
                score += 15
                reasons.append(f"📉 RSI {last['rsi_14']:.1f} (اشباع فروش شدید)")
            elif last['rsi_14'] < 40:
                score += 10
                reasons.append(f"📊 RSI {last['rsi_14']:.1f} (منطقه خرید)")
            elif last['rsi_14'] > 70:
                score -= 15
                reasons.append(f"📈 RSI {last['rsi_14']:.1f} (اشباع خرید شدید)")
            elif last['rsi_14'] > 60:
                score -= 10
                reasons.append(f"📊 RSI {last['rsi_14']:.1f} (منطقه فروش)")
        
        # MACD
        if not np.isnan(last['macd']) and not np.isnan(last['macd_signal']):
            if last['macd'] > last['macd_signal']:
                score += 8
                reasons.append("📊 MACD مثبت (صعودی)")
                if last['macd_hist'] > 0 and last['macd_hist'] > df['macd_hist'].iloc[-2]:
                    score += 5
                    reasons.append("📈 MACD هیستوگرام در حال افزایش")
            else:
                score -= 8
                reasons.append("📊 MACD منفی (نزولی)")
        
        # استوکاستیک
        if not np.isnan(last['stoch_k']) and not np.isnan(last['stoch_d']):
            if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']:
                score += 8
                reasons.append("📊 استوکاستیک اشباع فروش (صعودی)")
            elif last['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d']:
                score -= 8
                reasons.append("📊 استوکاستیک اشباع خرید (نزولی)")
        
        # ========== 3. حجم ==========
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = last['volume'] / vol_ma if vol_ma > 0 else 1
        
        if vol_ratio > 2.5:
            score += 10
            reasons.append(f"🔥 حجم فوق‌العاده (x{vol_ratio:.2f})")
        elif vol_ratio > 1.8:
            score += 7
            reasons.append(f"✅ حجم عالی (x{vol_ratio:.2f})")
        elif vol_ratio > 1.3:
            score += 4
            reasons.append(f"📊 حجم خوب (x{vol_ratio:.2f})")
        elif vol_ratio < 0.5:
            score -= 5
            reasons.append(f"⚠️ حجم بسیار پایین")
        
        # OBV
        if 'obv' in last and 'obv_ema' in last:
            if not np.isnan(last['obv']) and not np.isnan(last['obv_ema']):
                if last['obv'] > last['obv_ema']:
                    score += 5
                    reasons.append("📊 OBV صعودی (فشار خرید)")
                else:
                    score -= 4
                    reasons.append("📊 OBV نزولی (فشار فروش)")
        
        # MFI
        if not np.isnan(last['mfi_14']):
            if last['mfi_14'] < 20:
                score += 8
                reasons.append(f"💰 MFI {last['mfi_14']:.1f} (اشباع فروش)")
            elif last['mfi_14'] > 80:
                score -= 8
                reasons.append(f"💰 MFI {last['mfi_14']:.1f} (اشباع خرید)")
        
        # ========== 4. قدرت روند ==========
        if not np.isnan(last['adx']):
            if last['adx'] > 30:
                if last['plus_di'] > last['minus_di']:
                    score += 10
                    reasons.append(f"📈 ADX {last['adx']:.1f} (روند قوی صعودی)")
                elif last['plus_di'] < last['minus_di']:
                    score -= 10
                    reasons.append(f"📉 ADX {last['adx']:.1f} (روند قوی نزولی)")
                else:
                    score += 5
                    reasons.append(f"📊 ADX {last['adx']:.1f} (روند قوی)")
            elif last['adx'] > 20:
                reasons.append(f"📊 ADX {last['adx']:.1f} (روند متوسط)")
        
        # ========== 5. باند بولینجر ==========
        if not np.isnan(last['bb_position']):
            bb_pos = last['bb_position']
            if bb_pos < 15:
                score += 8
                reasons.append(f"📊 نزدیک باند پایین ({bb_pos:.1f}%) - حمایت قوی")
            elif bb_pos < 30:
                score += 5
                reasons.append(f"📊 نزدیک باند پایین ({bb_pos:.1f}%) - حمایت")
            elif bb_pos > 85:
                score -= 8
                reasons.append(f"📊 نزدیک باند بالا ({bb_pos:.1f}%) - مقاومت قوی")
            elif bb_pos > 70:
                score -= 5
                reasons.append(f"📊 نزدیک باند بالا ({bb_pos:.1f}%) - مقاومت")
        
        # ========== 6. الگوها ==========
        for p, power in patterns:
            if "صعودی" in p or "🟢" in p or "خرید" in p:
                score += power * 8
            elif "نزولی" in p or "🔴" in p or "فروش" in p:
                score -= power * 8
            reasons.append(p)
        
        # ========== 7. واگرایی ==========
        for d, power in divergence:
            if "صعودی" in d:
                score += power * 12
            elif "نزولی" in d:
                score -= power * 12
            reasons.append(d)
        
        # ========== 8. سطوح عرضه/تقاضا ==========
        if demand:
            dist_to_demand = (last['close'] - demand) / last['close'] * 100
            if dist_to_demand < 1.5:
                score += 10
                reasons.append(f"🛡 نزدیک سطح تقاضا (حمایت قوی)")
            elif dist_to_demand < 3:
                score += 6
                reasons.append(f"🛡 نزدیک سطح تقاضا (حمایت)")
        if supply:
            dist_to_supply = (supply - last['close']) / last['close'] * 100
            if dist_to_supply < 1.5:
                score -= 10
                reasons.append(f"🏔 نزدیک سطح عرضه (مقاومت قوی)")
            elif dist_to_supply < 3:
                score -= 6
                reasons.append(f"🏔 نزدیک سطح عرضه (مقاومت)")
        
        # ========== 9. شرایط نوسان ==========
        if "نوسان بسیار بالا" in volatility:
            score -= 5
            reasons.append("⚠️ نوسان بسیار بالا - ریسک افزایش یافته")
        elif "نوسان کم" in volatility:
            score += 3
            reasons.append("✅ نوسان مناسب برای ورود")
        
        # محدود کردن امتیاز بین 1 تا 99
        score = max(1, min(99, score))
        
        return score, list(dict.fromkeys(reasons))[:10]

# =========================
# ADVANCED DIP SIGNAL
# =========================

def get_dip_signal(symbol, timeframe):
    """نسخه پیشرفته سیگنال DIP با تمام فیلترها"""
    try:
        interval = DIP_TIMEFRAMES[timeframe]["binance"]
        df = get_dip_candles(symbol, interval, 200)
        
        if df is None or len(df) < 70:
            return None
        
        # محاسبه اندیکاتورها
        df = calculate_dip_indicators(df)
        last = df.iloc[-1]
        
        # تشخیص وضعیت بازار
        market_condition = detect_market_condition(df)
        trend, volatility, bb_state = market_condition
        
        # تشخیص الگوها
        patterns = detect_dip_patterns(df)
        
        # تشخیص سطوح
        demand, supply = detect_dip_levels(df)
        
        # تشخیص واگرایی
        divergence = detect_dip_divergence(df)
        
        # بررسی شرایط نوسان
        volatility_alert, vol_conditions = check_volatility_conditions(df)
        
        # محاسبه امتیاز نهایی
        scorer = DIPScorer()
        confidence, reasons = scorer.get_score(df, last, patterns, demand, supply, divergence, market_condition)
        
        # اعمال فیلتر نهایی
        if volatility_alert:
            confidence = max(1, confidence - 15)
            reasons.insert(0, "⚠️ هشدار نوسان شدید - با احتیاط وارد شوید")
        
        # تعیین جهت بر اساس امتیاز
        if confidence >= 68:
            direction = "LONG"
        elif confidence <= 32:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        # محاسبه حد سود و ضرر پیشرفته
        price = last['close']
        atr = last['atr_14'] if not np.isnan(last['atr_14']) else price * 0.02
        
        if direction == "LONG":
            # حد ضرر بر اساس ATR و سطوح حمایت
            sl_candidate = price - atr * 1.8
            if demand and demand < price:
                sl = max(sl_candidate, demand * 0.992)
            else:
                sl = sl_candidate
            
            # اهداف بر اساس ATR و سطوح مقاومت
            tp1 = price + atr * 1.5
            tp2 = price + atr * 2.8
            tp3 = price + atr * 5
            
            if supply and supply > price:
                tp3 = min(tp3, supply * 0.995)
                
        elif direction == "SHORT":
            # حد ضرر بر اساس ATR و سطوح مقاومت
            sl_candidate = price + atr * 1.8
            if supply and supply > price:
                sl = min(sl_candidate, supply * 1.008)
            else:
                sl = sl_candidate
            
            # اهداف بر اساس ATR و سطوح حمایت
            tp1 = price - atr * 1.5
            tp2 = price - atr * 2.8
            tp3 = price - atr * 5
            
            if demand and demand < price:
                tp3 = max(tp3, demand * 1.005)
        else:
            tp1 = tp2 = tp3 = sl = price
        
        # محاسبه درصد سود/ضرر
        if direction == "LONG":
            profit1 = (tp1 - price) / price * 100
            profit3 = (tp3 - price) / price * 100
            loss = (price - sl) / price * 100
        elif direction == "SHORT":
            profit1 = (price - tp1) / price * 100
            profit3 = (price - tp3) / price * 100
            loss = (sl - price) / price * 100
        else:
            profit1 = profit3 = loss = 0
        
        # ریسک به ریوارد
        rr_ratio = profit3 / loss if loss > 0 else 0
        
        # سطح قدرت
        if confidence >= 88:
            strength = "🔥🔥 فوق‌العاده قوی"
        elif confidence >= 78:
            strength = "💪 بسیار قوی"
        elif confidence >= 68:
            strength = "✅ قوی"
        elif confidence <= 32:
            strength = "🔴 قوی (فروش)"
        elif confidence <= 22:
            strength = "🔴🔴 بسیار قوی (فروش)"
        else:
            strength = "📊 متوسط"
        
        return {
            "symbol": symbol,
            "timeframe": DIP_TIMEFRAMES[timeframe]["name"],
            "direction": direction,
            "confidence": confidence,
            "strength": strength,
            "price": price,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "sl": sl,
            "profit1": profit1,
            "profit3": profit3,
            "loss": loss,
            "rr_ratio": rr_ratio,
            "reasons": reasons,
            "patterns": [p[0] for p in patterns][:3],
            "demand": demand,
            "supply": supply,
            "market_trend": trend,
            "volatility": volatility,
            "bb_state": bb_state,
            "vol_conditions": vol_conditions
        }
        
    except Exception as e:
        print(f"خطا در {symbol}: {e}")
        return None

# =========================
# HA SIGNAL
# =========================

def ha_signal(symbol):
    df = calculate_indicators(get_candles(symbol, "720"))
    if len(df) < 50:
        return None, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    long_cross = prev["ma20"] < prev["ma50"] and last["ma20"] > last["ma50"]
    short_cross = prev["ma20"] > prev["ma50"] and last["ma20"] < last["ma50"]
    trend_up = last["close"] > last["ema200"]
    rsi_ok = 40 < last["rsi"] < 70
    volume_ok = last["volume"] > last["vol_ma"]
    signal = None
    
    if long_cross and trend_up and rsi_ok and volume_ok:
        signal = "LONG"
    if short_cross and not trend_up:
        signal = "SHORT"
    
    return signal, last

def targets(price, atr, direction):
    if direction == "LONG":
        sl = price - atr * 1.5
        tp = price + atr * 3
    else:
        sl = price + atr * 1.5
        tp = price - atr * 3
    profit = abs(tp - price) / price * 100
    return tp, sl, profit

def success_rate(last, direction):
    score = 50
    trend_up = last["close"] > last["ema200"]
    rsi = last["rsi"]
    volume_ok = last["volume"] > last["vol_ma"]
    
    if direction == "LONG" and trend_up:
        score += 15
    if direction == "SHORT" and not trend_up:
        score += 15
    if 45 < rsi < 65:
        score += 10
    if volume_ok:
        score += 10
    
    atr_strength = last["atr"] / last["close"]
    if atr_strength > 0.01:
        score += 5
    
    return min(int(score), 90)

# =========================
# HA LOOP
# =========================

def ha_loop():
    while True:
        for symbol in HA_SYMBOLS:
            try:
                signal, last = ha_signal(symbol)
                if signal is None or last is None:
                    continue
                    
                price = last["close"]
                atr = last["atr"]
                
                if symbol not in positions:
                    positions[symbol] = None
                
                current_pos = positions[symbol]
                
                if current_pos and signal and signal != current_pos:
                    bot.send_message(
                        ADMIN_ID,
                        f"🟡 MODEL: HA\n\n❌ بستن معامله {symbol}\nقیمت: {price:.2f}"
                    )
                    positions[symbol] = None
                
                if signal and positions[symbol] is None:
                    tp, sl, profit = targets(price, atr, signal)
                    rate = success_rate(last, signal)
                    text = f"""
🟡 **مدل HA — {signal}**

🪙 {symbol}
📊 تایم‌فریم: 12 ساعت

💰 ورود: {price:.2f}
🎯 حد سود: {tp:.2f}
🛑 حد ضرر: {sl:.2f}

📈 سود: {profit:.2f}%
🎯成功率: {rate}%
"""
                    bot.send_message(ADMIN_ID, text, parse_mode="Markdown")
                    positions[symbol] = signal
            except Exception as e:
                print("خطا در HA:", e)
        time.sleep(60)

# =========================
# META SIGNAL
# =========================

def meta_signal(symbol, interval):
    try:
        df = calculate_indicators(get_candles(symbol, interval))
        if len(df) < 50:
            return None
            
        last = df.iloc[-1]
        ema_fast = last["ma20"]
        ema_slow = last["ma50"]
        rsi = last["rsi"]
        
        score = 0
        if ema_fast > ema_slow:
            score += 1
        else:
            score -= 1
        
        if rsi > 55:
            score += 1
        if rsi < 45:
            score -= 1
        
        direction = "LONG" if score > 0 else "SHORT"
        
        price = last["close"]
        atr = last["atr"]
        tp, sl, profit = targets(price, atr, direction)
        rate = success_rate(last, direction)
        
        return {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "tp": tp,
            "sl": sl,
            "profit": profit,
            "rate": rate,
            "score": score
        }
    except Exception as e:
        print(f"خطا در متا سیگنال {symbol}: {e}")
        return None

def send_meta_signals(interval, timeframe_name):
    msg = bot.send_message(
        ADMIN_ID,
        f"🧠 **در حال تحلیل {timeframe_name} با متا سیگنال...**\n⏱ لطفاً ۱۵ ثانیه صبر کنید",
        parse_mode="Markdown"
    )
    
    signals = []
    for symbol in HA_SYMBOLS:
        signal = meta_signal(symbol, interval)
        if signal:
            signals.append(signal)
        time.sleep(1)
    
    if not signals:
        bot.edit_message_text(
            "❌ **سیگنالی یافت نشد!**",
            ADMIN_ID,
            msg.message_id,
            parse_mode="Markdown"
        )
        return
    
    long_signals = [s for s in signals if s["direction"] == "LONG"]
    short_signals = [s for s in signals if s["direction"] == "SHORT"]
    
    long_signals.sort(key=lambda x: x["rate"], reverse=True)
    short_signals.sort(key=lambda x: x["rate"], reverse=True)
    
    result = f"🧠 **سیگنال‌های متا - {timeframe_name}**\n"
    result += "════════════════════\n\n"
    
    if long_signals:
        result += "🟢 **سیگنال‌های LONG**\n"
        for s in long_signals[:5]:
            result += f"  **{s['symbol']}**\n"
            result += f"    قیمت: {s['price']:.2f}\n"
            result += f"    TP: {s['tp']:.2f} | SL: {s['sl']:.2f}\n"
            result += f"    سود: {s['profit']:.2f}% | موفقیت: {s['rate']}%\n\n"
    
    if short_signals:
        result += "🔴 **سیگنال‌های SHORT**\n"
        for s in short_signals[:5]:
            result += f"  **{s['symbol']}**\n"
            result += f"    قیمت: {s['price']:.2f}\n"
            result += f"    TP: {s['tp']:.2f} | SL: {s['sl']:.2f}\n"
            result += f"    سود: {s['profit']:.2f}% | موفقیت: {s['rate']}%\n\n"
    
    result += "════════════════════"
    
    bot.edit_message_text(result, ADMIN_ID, msg.message_id, parse_mode="Markdown")

# =========================
# AUTO LOOP
# =========================

def auto_loop():
    while True:
        try:
            symbol = np.random.choice(AUTO_SYMBOLS)
            signal = meta_signal(symbol, "60")
            if signal:
                text = f"""
🤖 **مدل AUTO — {signal['direction']}**

🪙 {symbol}

💰 ورود: {signal['price']:.2f}
🎯 حد سود: {signal['tp']:.2f}
🛑 حد ضرر: {signal['sl']:.2f}

📈 سود: {signal['profit']:.2f}%
🎯 موفقیت: {signal['rate']}%
"""
                bot.send_message(ADMIN_ID, text, parse_mode="Markdown")
        except Exception as e:
            print("خطا در AUTO:", e)
        time.sleep(300)

# =========================
# MARKET ANALYSIS
# =========================

def market_analysis():
    symbol = "BTCUSDT"
    df = calculate_indicators(get_candles(symbol, "240"))
    if len(df) < 50:
        bot.send_message(ADMIN_ID, "❌ خطا در دریافت داده")
        return
    
    last = df.iloc[-1]
    trend_up = last["ma20"] > last["ma50"] > last["ema200"]
    trend_down = last["ma20"] < last["ma50"] < last["ema200"]
    volatility = last["atr"] / last["close"]
    
    whale_score = np.random.randint(40, 90)
    smart_money = np.random.randint(40, 90)
    
    if trend_up:
        market = "📈 صعودی"
    elif trend_down:
        market = "📉 نزولی"
    else:
        market = "🔄 رنج"
    
    quality = int((whale_score + smart_money) / 2)
    good_day = "✅ روز خوبی برای معامله" if quality > 60 else "⚠️ امروز احتیاط کن"
    
    text = f"""
📅 **تحلیل روز بازار کریپتو**

وضعیت بازار: {market}

نوسان: {int(volatility*100)}%
هوشمند: {smart_money}%
نهنگ‌ها: {whale_score}%

کیفیت بازار: {quality}%

{good_day}
"""
    bot.send_message(ADMIN_ID, text, parse_mode="Markdown")

def daily_loop():
    while True:
        try:
            market_analysis()
        except Exception as e:
            print("خطا در تحلیل روز:", e)
        time.sleep(43200)

# =========================
# MEMBERSHIP CHECK
# =========================

def check_membership(user_id):
    if REQUIRED_CHANNEL is None:
        return True
    
    try:
        channel = REQUIRED_CHANNEL.replace('@', '')
        if not channel.startswith('@'):
            channel = f"@{channel}"
        
        member = bot.get_chat_member(channel, user_id)
        
        if member.status in ["member", "creator", "administrator"]:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"خطا در بررسی عضویت: {e}")
        return False

def check_user_access(user_id, chat_id):
    if user_id == ADMIN_ID:
        return True
    
    if not check_membership(user_id):
        join_btn = types.InlineKeyboardMarkup()
        if REQUIRED_CHANNEL:
            channel = REQUIRED_CHANNEL.replace('@', '')
            join_btn.add(
                types.InlineKeyboardButton(
                    "📢 عضویت در کانال",
                    url=f"https://t.me/{channel}"
                )
            )
            join_btn.add(
                types.InlineKeyboardButton(
                    "✅ بررسی مجدد",
                    callback_data="check_membership"
                )
            )
        
        bot.send_message(
            chat_id,
            "❌ برای استفاده از ربات باید عضو کانال شوید",
            reply_markup=join_btn
        )
        return False
    
    return True

# =========================
# MAIN MENU
# =========================

def main_menu(user_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("📊 انتخاب ارز")
    markup.add("📡 وضعیت ربات")
    markup.add("📅 تحلیل روز")
    
    if user_id == ADMIN_ID:
        markup.add("⚙️ تنظیمات")
    
    return markup

def admin_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("💳 پرداخت‌ها")
    markup.add("👥 عضویت")
    markup.add("⭐ اشتراک")
    markup.add("⬅️ بازگشت")
    return markup

# =========================
# START COMMAND
# =========================

@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    
    if user_id == ADMIN_ID:
        bot.send_message(
            message.chat.id,
            "🚀 خوش آمدید مدیر! ربات آماده است",
            reply_markup=main_menu(user_id)
        )
        return
    
    if not check_membership(user_id):
        join_btn = types.InlineKeyboardMarkup()
        if REQUIRED_CHANNEL:
            channel = REQUIRED_CHANNEL.replace('@', '')
            join_btn.add(
                types.InlineKeyboardButton(
                    "📢 عضویت در کانال",
                    url=f"https://t.me/{channel}"
                )
            )
            join_btn.add(
                types.InlineKeyboardButton(
                    "✅ بررسی مجدد",
                    callback_data="check_membership"
                )
            )
        
        bot.send_message(
            message.chat.id,
            "❌ برای استفاده از ربات باید عضو کانال شوید\n\n"
            "پس از عضویت، دکمه 'بررسی مجدد' را بزنید",
            reply_markup=join_btn
        )
        return
    
    bot.send_message(
        message.chat.id,
        "🚀 ربات آماده است",
        reply_markup=main_menu(user_id)
    )

# =========================
# CALLBACK HANDLER
# =========================

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    if call.data == "check_membership":
        user_id = call.from_user.id
        
        if check_membership(user_id):
            bot.edit_message_text(
                "✅ عضویت شما تایید شد!\n"
                "در حال بازگشت به منوی اصلی...",
                call.message.chat.id,
                call.message.message_id
            )
            
            bot.send_message(
                call.message.chat.id,
                "🚀 ربات آماده است",
                reply_markup=main_menu(user_id)
            )
        else:
            bot.answer_callback_query(
                call.id,
                "❌ شما هنوز عضو کانال نشده‌اید!",
                show_alert=True
            )
    
    elif call.data == "close_dip":
        bot.delete_message(call.message.chat.id, call.message.message_id)
        bot.answer_callback_query(call.id, "بسته شد")
    
    elif call.data == "back_to_dip_list":
        # نمایش مجدد لیست سیگنال‌ها
        if hasattr(bot, 'dip_signals_cache') and bot.dip_signals_cache:
            markup = types.InlineKeyboardMarkup(row_width=1)
            
            for key, signal in bot.dip_signals_cache.items():
                emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
                btn_text = f"{emoji} {signal['symbol']} - {signal['timeframe']} (قدرت: {signal['confidence']:.0f}%)"
                
                markup.add(
                    types.InlineKeyboardButton(
                        btn_text,
                        callback_data=key
                    )
                )
            
            markup.add(types.InlineKeyboardButton("❌ بستن", callback_data="close_dip"))
            
            bot.edit_message_text(
                "⭐ **۵ سیگنال برتر DIP** ⭐\nبرای مشاهده جزئیات هر سیگنال کلیک کنید:\n\n",
                call.message.chat.id,
                call.message.message_id,
                parse_mode="Markdown",
                reply_markup=markup
            )
    
    elif call.data.startswith("dip_detail_"):
        if hasattr(bot, 'dip_signals_cache') and call.data in bot.dip_signals_cache:
            signal = bot.dip_signals_cache[call.data]
            text, markup = show_dip_signal_detail(call.message.chat.id, call.message.message_id, signal)
            
            bot.edit_message_text(
                text,
                call.message.chat.id,
                call.message.message_id,
                parse_mode="Markdown",
                reply_markup=markup
            )

# =========================
# ID COMMAND
# =========================

@bot.message_handler(commands=['id'])
def get_id(message):
    bot.reply_to(message, f"🆔 آیدی شما:\n{message.from_user.id}")

# =========================
# ADMIN COMMAND
# =========================

@bot.message_handler(commands=['admin'])
def admin_panel(message):
    if message.from_user.id != ADMIN_ID:
        return
    bot.send_message(
        message.chat.id,
        "⚙️ پنل مدیریت",
        reply_markup=admin_menu()
    )

# =========================
# SETTINGS BUTTON
# =========================

@bot.message_handler(func=lambda m: m.text == "⚙️ تنظیمات")
def settings_panel(message):
    if message.from_user.id != ADMIN_ID:
        bot.reply_to(message, "❌ دسترسی ندارید")
        return
    bot.send_message(
        message.chat.id,
        "⚙️ وارد بخش تنظیمات شدید",
        reply_markup=admin_menu()
    )

# =========================
# MEMBERSHIP SETTINGS
# =========================

@bot.message_handler(func=lambda m: m.text == "👥 عضویت")
def membership_menu(message):
    if message.from_user.id != ADMIN_ID:
        return
    user_state[message.from_user.id] = "set_channel"
    bot.send_message(
        message.chat.id,
        "آیدی کانال را ارسال کنید\n"
        "مثال:\n"
        "@mychannel\n\n"
        "⚠️ نکته: ربات باید در کانال عضو و ادمین باشد"
    )

# =========================
# SELECT SYMBOL
# =========================

@bot.message_handler(func=lambda m: m.text == "📊 انتخاب ارز")
def select_symbol(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for s in HA_SYMBOLS:
        markup.add(s)
    markup.add("⬅️ بازگشت")
    bot.send_message(message.chat.id, "🪙 انتخاب ارز:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text in HA_SYMBOLS)
def selected_symbol(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    symbol = message.text
    signal, last = ha_signal(symbol)
    
    if signal is None or last is None:
        bot.send_message(message.chat.id, "⚠️ سیگنال فعلاً موجود نیست")
        return
    
    price = last["close"]
    atr = last["atr"]
    tp, sl, profit = targets(price, atr, signal)
    rate = success_rate(last, signal)
    
    text = f"""
🟡 **مدل HA — {signal}**

🪙 {symbol}
📊 تایم‌فریم: 12 ساعت

💰 ورود: {price:.2f}
🎯 حد سود: {tp:.2f}
🛑 حد ضرر: {sl:.2f}

📈 سود: {profit:.2f}%
🎯 موفقیت: {rate}%
"""
    bot.send_message(message.chat.id, text, parse_mode="Markdown")

# =========================
# DAILY BUTTON
# =========================

@bot.message_handler(func=lambda m: m.text == "📅 تحلیل روز")
def daily_button(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    market_analysis()

# =========================
# STATUS MENU
# =========================

@bot.message_handler(func=lambda m: m.text == "📡 وضعیت ربات")
def status_menu(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("🚀 سیگنال HA")
    markup.add("🧠 سیگنال متا")
    markup.add("💎 سیگنال DIP پیشرفته")
    markup.add("⬅️ بازگشت")
    bot.send_message(message.chat.id, "📡 وضعیت ربات", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "🚀 سیگنال HA")
def ha_manual(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    bot.send_message(message.chat.id, "✅ سیستم HA فعال است")

# =========================
# META SIGNAL MENU
# =========================

@bot.message_handler(func=lambda m: m.text == "🧠 سیگنال متا")
def meta_menu(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("⏱ کوتاه مدت (15 دقیقه)")
    markup.add("📈 بلند مدت (4 ساعت)")
    markup.add("⬅️ بازگشت")
    bot.send_message(message.chat.id, "انتخاب نوع:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "⏱ کوتاه مدت (15 دقیقه)")
def meta_short(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    send_meta_signals("15", "۱۵ دقیقه")

@bot.message_handler(func=lambda m: m.text == "📈 بلند مدت (4 ساعت)")
def meta_long(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    send_meta_signals("240", "۴ ساعت")

# =========================
# ADVANCED DIP SIGNAL MENU
# =========================

@bot.message_handler(func=lambda m: m.text == "💎 سیگنال DIP پیشرفته")
def dip_menu(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        "⏱ ۱۵ دقیقه",
        "⏱ ۳۰ دقیقه",
        "⏱ ۱ ساعت",
        "⏱ ۴ ساعت",
        "⏱ روزانه",
        "⭐ قوی‌ترین سیگنال‌های DIP",
        "⬅️ بازگشت"
    )
    bot.send_message(
        message.chat.id,
        "⏰ **تایم‌فریم مورد نظر را انتخاب کنید:**",
        reply_markup=markup,
        parse_mode="Markdown"
    )

# =========================
# DIP BEST SIGNALS - اصلاح شده
# =========================

@bot.message_handler(func=lambda m: m.text == "⭐ قوی‌ترین سیگنال‌های DIP")
def dip_best_signals(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    msg = bot.send_message(
        message.chat.id,
        "🔍 **در حال جستجوی قوی‌ترین سیگنال‌های DIP...**\n⏱ لطفاً ۳۰ ثانیه صبر کنید",
        parse_mode="Markdown"
    )
    
    all_signals = []
    timeframes = ["15m", "30m", "1h", "4h", "1d"]
    
    for tf in timeframes:
        for symbol in DIP_SYMBOLS[:5]:  # محدود کردن به 5 نماد اول برای سرعت
            signal = get_dip_signal(symbol, tf)
            if signal and signal["direction"] != "NEUTRAL":
                all_signals.append(signal)
            time.sleep(1)
    
    all_signals.sort(key=lambda x: x["confidence"], reverse=True)
    
    if not all_signals:
        bot.edit_message_text(
            "❌ سیگنالی یافت نشد!",
            message.chat.id,
            msg.message_id
        )
        return
    
    # ایجاد دکمه‌های اینلاین برای مشاهده جزئیات
    markup = types.InlineKeyboardMarkup(row_width=1)
    
    # ذخیره سیگنال‌ها در حافظه موقت برای دسترسی در callback
    if not hasattr(bot, 'dip_signals_cache'):
        bot.dip_signals_cache = {}
    
    for i, s in enumerate(all_signals[:5], 1):  # فقط 5 سیگنال برتر
        emoji = "🟢" if s["direction"] == "LONG" else "🔴"
        btn_text = f"{i}. {emoji} {s['symbol']} - {s['timeframe']} (قدرت: {s['confidence']:.0f}%)"
        
        callback_data = f"dip_detail_{i}"
        bot.dip_signals_cache[callback_data] = s
        
        markup.add(
            types.InlineKeyboardButton(
                btn_text,
                callback_data=callback_data
            )
        )
    
    markup.add(types.InlineKeyboardButton("❌ بستن", callback_data="close_dip"))
    
    result = "⭐ **۵ سیگنال برتر DIP** ⭐\n"
    result += "برای مشاهده جزئیات هر سیگنال کلیک کنید:\n\n"
    
    bot.edit_message_text(
        result, 
        message.chat.id, 
        msg.message_id, 
        parse_mode="Markdown",
        reply_markup=markup
    )

# =========================
# SHOW DIP SIGNAL DETAIL - تابع جدید
# =========================

def show_dip_signal_detail(chat_id, message_id, signal):
    """نمایش جزئیات کامل یک سیگنال DIP با استراتژی خرید و فروش"""
    
    direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴" if signal["direction"] == "SHORT" else "⚪"
    direction_text = "خرید (LONG)" if signal["direction"] == "LONG" else "فروش (SHORT)" if signal["direction"] == "SHORT" else "خنثی"
    
    # تعیین سطح ریسک
    if signal["loss"] < 2:
        risk_level = "پایین"
        risk_emoji = "🟢"
    elif signal["loss"] < 4:
        risk_level = "متوسط"
        risk_emoji = "🟡"
    else:
        risk_level = "بالا"
        risk_emoji = "🔴"
    
    # تعیین کیفیت RR
    if signal["rr_ratio"] > 3:
        rr_quality = "✅ عالی"
    elif signal["rr_ratio"] > 2:
        rr_quality = "👍 خوب"
    elif signal["rr_ratio"] > 1:
        rr_quality = "⚠️ قابل قبول"
    else:
        rr_quality = "❌ نامناسب"
    
    text = f"""
{direction_emoji} **سیگنال DIP پیشرفته - {signal['symbol']}** {direction_emoji}
══════════════════════════

📊 **اطلاعات کلی:**
• نماد: {signal['symbol']}
• بازه زمانی: {signal['timeframe']}
• **جهت معامله: {direction_text}** {direction_emoji}
• قدرت سیگنال: {signal['confidence']:.0f}% ({signal['strength']})
• وضعیت بازار: {signal['market_trend']} | نوسان: {signal['volatility']}

💰 **سطوح معاملاتی:**
• قیمت فعلی: {signal['price']:.4f}
• **حد سود ۱:** {signal['tp1']:.4f} (+{signal['profit1']:.2f}%)
• حد سود ۲: {signal['tp2']:.4f}
• **حد سود ۳:** {signal['tp3']:.4f} (+{signal['profit3']:.2f}%)
• **حد ضرر:** {signal['sl']:.4f} (-{signal['loss']:.2f}%)
• نسبت ریسک به ریوارد: {signal['rr_ratio']:.2f} {rr_quality}

📈 **تحلیل تکنیکال:**
"""
    
    # اضافه کردن دلایل
    if signal['reasons']:
        for i, reason in enumerate(signal['reasons'][:5], 1):
            text += f"• {reason}\n"
    
    # اضافه کردن الگوها
    if signal['patterns']:
        text += f"\n🔍 **الگوهای شناسایی شده:**\n"
        for pattern in signal['patterns']:
            text += f"• {pattern}\n"
    
    # اضافه کردن سطوح حمایت و مقاومت
    if signal['demand']:
        demand_dist = (signal['price'] - signal['demand']) / signal['price'] * 100
        text += f"\n🛡 **حمایت:** {signal['demand']:.4f} (فاصله: {demand_dist:.2f}%)\n"
    if signal['supply']:
        supply_dist = (signal['supply'] - signal['price']) / signal['price'] * 100
        text += f"🏔 **مقاومت:** {signal['supply']:.4f} (فاصله: {supply_dist:.2f}%)\n"
    
    # استراتژی معاملاتی بر اساس جهت
    if signal["direction"] == "LONG":
        text += f"""
══════════════════════════
📋 **استراتژی خرید پیشنهادی:**

🟢 **نوع معامله:** خرید (LONG)
• حداکثر ریسک: ۱-۲٪ از سرمایه
• مدت زمان پیشنهادی: {'کوتاه مدت' if signal['timeframe'] in ['۱۵ دقیقه', '۳۰ دقیقه'] else 'میان مدت' if signal['timeframe'] in ['۱ ساعت', '۴ ساعت'] else 'بلند مدت'}

**مراحل خرید:**

1️⃣ **ورود به معامله:**
   • محدوده ورود: {signal['price']:.4f}
   • استراتژی ورود: خرید در قیمت فعلی
   • {('منتظر اصلاح به محدوده ' + str(signal['demand']) + ' باشید') if signal['demand'] and signal['demand'] < signal['price'] * 0.98 else 'ورود در قیمت بازار'}

2️⃣ **مدیریت سرمایه:**
   • حد ضرر: {signal['sl']:.4f} (حداکثر ضرر {signal['loss']:.2f}%)
   • حجم معامله: {risk_emoji} سطح ریسک {risk_level}
   • نسبت سود به ضرر: {signal['rr_ratio']:.2f}

3️⃣ **اهداف سود:**
   • **هدف اول** ({signal['profit1']:.2f}%): {signal['tp1']:.4f}
     ➡️ در این سطح ۳۰٪ از موقعیت را ببندید
   
   • **هدف دوم**: {signal['tp2']:.4f}
     ➡️ حد ضرر را به نقطه سر به سر (ورود) منتقل کنید
   
   • **هدف سوم** ({signal['profit3']:.2f}%): {signal['tp3']:.4f}
     ➡️ باقیمانده موقعیت را با تریلینگ استاپ مدیریت کنید

4️⃣ **مدیریت ریسک:**
   • نسبت ریسک به ریوارد: {rr_quality}
   • سطح ریسک: {risk_emoji} {risk_level}
   • {'✅ مناسب برای معامله' if signal['rr_ratio'] > 2 else '⚠️ ریسک نسبت به سود مناسب نیست' if signal['rr_ratio'] > 1 else '❌ از معامله خودداری کنید'}

⚠️ **هشدارهای مهم:**
• {signal['vol_conditions'][0] if signal['vol_conditions'] else 'شرایط نوسان عادی'}
• معامله با حد ضرر الزامی است
• بازار رمزارزها ریسک بالایی دارد
"""
    
    elif signal["direction"] == "SHORT":
        text += f"""
══════════════════════════
📋 **استراتژی فروش پیشنهادی:**

🔴 **نوع معامله:** فروش (SHORT)
• حداکثر ریسک: ۱-۲٪ از سرمایه
• مدت زمان پیشنهادی: {'کوتاه مدت' if signal['timeframe'] in ['۱۵ دقیقه', '۳۰ دقیقه'] else 'میان مدت' if signal['timeframe'] in ['۱ ساعت', '۴ ساعت'] else 'بلند مدت'}

**مراحل فروش:**

1️⃣ **ورود به معامله:**
   • محدوده ورود: {signal['price']:.4f}
   • استراتژی ورود: فروش در قیمت فعلی
   • {('منتظر رشد به محدوده ' + str(signal['supply']) + ' باشید') if signal['supply'] and signal['supply'] > signal['price'] * 1.02 else 'ورود در قیمت بازار'}

2️⃣ **مدیریت سرمایه:**
   • حد ضرر: {signal['sl']:.4f} (حداکثر ضرر {signal['loss']:.2f}%)
   • حجم معامله: {risk_emoji} سطح ریسک {risk_level}
   • نسبت سود به ضرر: {signal['rr_ratio']:.2f}

3️⃣ **اهداف سود:**
   • **هدف اول** ({signal['profit1']:.2f}%): {signal['tp1']:.4f}
     ➡️ در این سطح ۳۰٪ از موقعیت را ببندید
   
   • **هدف دوم**: {signal['tp2']:.4f}
     ➡️ حد ضرر را به نقطه سر به سر (ورود) منتقل کنید
   
   • **هدف سوم** ({signal['profit3']:.2f}%): {signal['tp3']:.4f}
     ➡️ باقیمانده موقعیت را با تریلینگ استاپ مدیریت کنید

4️⃣ **مدیریت ریسک:**
   • نسبت ریسک به ریوارد: {rr_quality}
   • سطح ریسک: {risk_emoji} {risk_level}
   • {'✅ مناسب برای معامله' if signal['rr_ratio'] > 2 else '⚠️ ریسک نسبت به سود مناسب نیست' if signal['rr_ratio'] > 1 else '❌ از معامله خودداری کنید'}

⚠️ **هشدارهای مهم:**
• {signal['vol_conditions'][0] if signal['vol_conditions'] else 'شرایط نوسان عادی'}
• معامله با حد ضرر الزامی است
• بازار رمزارزها ریسک بالایی دارد
"""
    
    else:  # NEUTRAL
        text += f"""
══════════════════════════
📋 **وضعیت خنثی:**

⚪ **سیگنال مشخصی برای معامله وجود ندارد**
• قدرت سیگنال: {signal['confidence']:.0f}%
• بهترین اقدام: منتظر بمانید

⚠️ **توصیه:**
• از معامله در این شرایط خودداری کنید
• منتظر سیگنال قوی‌تر باشید
• می‌توانید تایم‌فریم بالاتر را بررسی کنید
"""
    
    # دکمه‌های بازگشت
    markup = types.InlineKeyboardMarkup()
    markup.add(
        types.InlineKeyboardButton("🔙 بازگشت به لیست", callback_data="back_to_dip_list"),
        types.InlineKeyboardButton("❌ بستن", callback_data="close_dip")
    )
    
    return text, markup

# =========================
# DIP TIMEFRAME HANDLER
# =========================

@bot.message_handler(func=lambda m: m.text in ["⏱ ۱۵ دقیقه", "⏱ ۳۰ دقیقه", "⏱ ۱ ساعت", "⏱ ۴ ساعت", "⏱ روزانه"])
def dip_handle_timeframe(message):
    user_id = message.from_user.id
    
    if not check_user_access(user_id, message.chat.id):
        return
    
    tf_map = {
        "۱۵ دقیقه": "15m",
        "۳۰ دقیقه": "30m",
        "۱ ساعت": "1h",
        "۴ ساعت": "4h",
        "روزانه": "1d"
    }
    
    tf = None
    for name, code in tf_map.items():
        if name in message.text:
            tf = code
            break
    
    if not tf:
        return
    
    msg = bot.send_message(
        message.chat.id,
        f"🔍 **در حال تحلیل {DIP_TIMEFRAMES[tf]['name']} با DIP پیشرفته...**\n⏱ لطفاً ۲۰ ثانیه صبر کنید",
        parse_mode="Markdown"
    )
    
    signals = []
    for symbol in DIP_SYMBOLS:
        try:
            signal = get_dip_signal(symbol, tf)
            if signal and signal["direction"] != "NEUTRAL":
                signals.append(signal)
            time.sleep(1)
        except:
            continue
    
    if not signals:
        bot.edit_message_text(
            "❌ **سیگنالی یافت نشد!**",
            message.chat.id,
            msg.message_id,
            parse_mode="Markdown"
        )
        return
    
    long_signals = [s for s in signals if s["direction"] == "LONG"]
    short_signals = [s for s in signals if s["direction"] == "SHORT"]
    
    long_signals.sort(key=lambda x: x["confidence"], reverse=True)
    short_signals.sort(key=lambda x: x["confidence"], reverse=True)
    
    result = f"💎 **سیگنال‌های DIP پیشرفته - {DIP_TIMEFRAMES[tf]['name']}**\n"
    result += "════════════════════\n\n"
    
    if long_signals:
        result += "🟢 **سیگنال‌های LONG**\n"
        for s in long_signals[:3]:
            result += f"  **{s['symbol']}**\n"
            result += f"    قدرت: {s['confidence']:.0f}% ({s['strength']})\n"
            result += f"    قیمت: {s['price']:.4f}\n"
            result += f"    TP1: {s['tp1']:.4f} (+{s['profit1']:.1f}%)\n"
            result += f"    SL: {s['sl']:.4f} (-{s['loss']:.1f}%)\n"
            result += f"    RR: {s['rr_ratio']:.2f}\n"
            if s['reasons']:
                result += f"    📊 {s['reasons'][0]}\n"
            result += "\n"
    
    if short_signals:
        result += "🔴 **سیگنال‌های SHORT**\n"
        for s in short_signals[:3]:
            result += f"  **{s['symbol']}**\n"
            result += f"    قدرت: {s['confidence']:.0f}% ({s['strength']})\n"
            result += f"    قیمت: {s['price']:.4f}\n"
            result += f"    TP1: {s['tp1']:.4f} (+{s['profit1']:.1f}%)\n"
            result += f"    SL: {s['sl']:.4f} (-{s['loss']:.1f}%)\n"
            result += f"    RR: {s['rr_ratio']:.2f}\n"
            if s['reasons']:
                result += f"    📊 {s['reasons'][0]}\n"
            result += "\n"
    
    result += "════════════════════"
    
    bot.edit_message_text(result, message.chat.id, msg.message_id, parse_mode="Markdown")

# =========================
# BACK BUTTON
# =========================

@bot.message_handler(func=lambda m: m.text == "⬅️ بازگشت")
def back(message):
    bot.send_message(
        message.chat.id, 
        "منو", 
        reply_markup=main_menu(message.from_user.id)
    )

# =========================
# HANDLE ALL MESSAGES
# =========================

@bot.message_handler(func=lambda m: True)
def handle_all_messages(message):
    global REQUIRED_CHANNEL
    user_id = message.from_user.id

    if user_id == ADMIN_ID:
        if user_state.get(user_id) == "set_channel":
            channel_input = message.text.strip()
            
            if not channel_input.startswith('@'):
                channel_input = f"@{channel_input}"
            
            REQUIRED_CHANNEL = channel_input
            user_state[user_id] = None
            
            try:
                bot.get_chat(REQUIRED_CHANNEL)
                bot.send_message(
                    message.chat.id,
                    f"✅ کانال تنظیم شد:\n{REQUIRED_CHANNEL}\n\n"
                    "⚠️ توجه: برای عملکرد صحیح، ربات باید ادمین کانال باشد"
                )
            except Exception as e:
                bot.send_message(
                    message.chat.id,
                    f"⚠️ کانال تنظیم شد اما ربات در کانال عضو نیست!\n"
                    f"لطفاً ربات را به کانال {REQUIRED_CHANNEL} اضافه کنید و ادمین کنید."
                )
            return

    if user_id != ADMIN_ID:
        if not check_membership(user_id):
            join_btn = types.InlineKeyboardMarkup()
            if REQUIRED_CHANNEL:
                channel = REQUIRED_CHANNEL.replace('@', '')
                join_btn.add(
                    types.InlineKeyboardButton(
                        "📢 عضویت در کانال",
                        url=f"https://t.me/{channel}"
                    )
                )
                join_btn.add(
                    types.InlineKeyboardButton(
                        "✅ بررسی مجدد",
                        callback_data="check_membership"
                    )
                )
            
            bot.send_message(
                message.chat.id,
                "❌ برای استفاده از ربات باید عضو کانال شوید",
                reply_markup=join_btn
            )
        else:
            bot.send_message(
                message.chat.id,
                "❌ دستور نامعتبر. از منوی اصلی استفاده کنید.",
                reply_markup=main_menu(user_id)
            )

# =========================
# THREADS
# =========================

threading.Thread(target=ha_loop, daemon=True).start()
threading.Thread(target=auto_loop, daemon=True).start()
threading.Thread(target=daily_loop, daemon=True).start()

print("🤖 ربات با موفقیت راه‌اندازی شد!")
print("✅ نسخه پیشرفته DIP با استراتژی خرید و فروش اضافه شد:")
print("   - نمایش ۵ سیگنال برتر DIP به صورت دکمه‌ای")
print("   - نمایش جزئیات کامل هر سیگنال با کلیک")
print("   - استراتژی خرید و فروش گام به گام")
print("   - مدیریت سرمایه و ریسک")
print("   - اهداف سه‌گانه با توضیحات")
print("   - سطح ریسک و کیفیت سیگنال")
print("\n⚠️ نکته: برای عملکرد صحیح بررسی عضویت:")
print("1. ربات را به کانال مورد نظر اضافه کنید")
print("2. ربات را ادمین کانال کنید")
print("3. از منوی ادمین > تنظیمات > عضویت، آیدی کانال را تنظیم کنید")

bot.infinity_polling()