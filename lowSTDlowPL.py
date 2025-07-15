import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
prev_rsi = np.full(nInst, 50.0) 

# --- PARAMETERS ---
VOLATILITY_PERIOD = 20
VOLATILITY_THRESHOLD = 0.15

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50
TREND_CONFIRMATION_THRESHOLD = 0.02

# --- INDICATOR THRESHOLDS ---
RSI_WINDOW = 20
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
PULLBACK_ENTRY_THRESHOLD = 55
DIVERGENCE_LOOKBACK = 20
STOCHASTIC_WINDOW = 20
CCI_WINDOW = 20


def getEMA(prices):
    nInst, nt = prices.shape
    if nt < LONG_TERM_EMA_DAYS:
        return None
    prices_df = pd.DataFrame(prices.T)
    ema_short = prices_df.ewm(span=SHORT_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    ema_long = prices_df.ewm(span=LONG_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    return np.array([ema_short.to_numpy(), ema_long.to_numpy()])

def getRSI(prices, window=14):
    nInst, nt = prices.shape
    if nt < window + 1:
        return None
    prices_df = pd.DataFrame(prices.T)
    delta = prices_df.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def getStochastic(prices, window=14):
    nInst, nt = prices.shape
    if nt < window:
        return np.full(nInst, 50.0)
    prices_df = pd.DataFrame(prices.T)
    highest_high = prices_df.rolling(window=window, min_periods=window).max()
    lowest_low = prices_df.rolling(window=window, min_periods=window).min()
    stochastic_k = 100 * ((prices_df - lowest_low) / (highest_high - lowest_low + 1e-6))
    return stochastic_k.iloc[-1].fillna(50.0).to_numpy()

def getCCI(prices, window=20):
    nInst, nt = prices.shape
    if nt < window:
        return np.full(nInst, 0.0)
    prices_df = pd.DataFrame(prices.T)
    typical_price = prices_df
    sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
    mean_dev = (typical_price - sma_tp).abs().rolling(window=window, min_periods=window).mean()
    cci = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-6)
    return cci.iloc[-1].fillna(0.0).to_numpy()

def calculatePerChange(new_val, old_val):
    epsilon = 1e-8
    return ((new_val - old_val) / (old_val + epsilon)) * 100

def getVolatility(prices):
    nInst, nt = prices.shape
    if nt < VOLATILITY_PERIOD + 1:
        return np.zeros(nInst)
    price_history = prices[:, -(VOLATILITY_PERIOD + 1):]
    volatility = np.zeros(nInst)
    for n, stock in enumerate(price_history):
        per_change_arr = np.zeros(VOLATILITY_PERIOD)
        for i in range(1, VOLATILITY_PERIOD + 1):
            per_change_arr[i - 1] = calculatePerChange(stock[i], stock[i - 1])
        volatility[n] = np.std(per_change_arr)
    return volatility

def find_bearish_divergence(price_window, rsi_window):
    if price_window[-1] < max(price_window[:-1]): return False
    if rsi_window[-1] > max(rsi_window[:-1]): return False
    return True

def find_bullish_divergence(price_window, rsi_window):
    if price_window[-1] > min(price_window[:-1]): return False
    if rsi_window[-1] < min(rsi_window[:-1]): return False
    return True

def getMyPosition(prcSoFar):
    global currentPos, prev_rsi
    nInst, nt = prcSoFar.shape

    if nt < LONG_TERM_EMA_DAYS:
        return currentPos
    
    # --- Indicator Calculation ---
    volatility = getVolatility(prcSoFar)
    EMAs = getEMA(prcSoFar)
    if EMAs is None: return currentPos
    short_EMA, long_EMA = EMAs[0], EMAs[1]

    rsi_series_df = getRSI(prcSoFar)
    if rsi_series_df is None: return currentPos
    rsi = rsi_series_df.iloc[-1].to_numpy()
    
    # Integrate new indicators but don't use them in logic yet
    stochastic = getStochastic(prcSoFar)
    cci = getCCI(prcSoFar)
    
    # --- Main Logic Loop (For each instrument) ---
    for i in range(nInst):
        if volatility[i] < VOLATILITY_THRESHOLD:
            continue
        
        ema_diff_normalized = (short_EMA[i] - long_EMA[i]) / long_EMA[i]

        if abs(ema_diff_normalized) > TREND_CONFIRMATION_THRESHOLD:
            # TRENDING MARKET
            if ema_diff_normalized > 0:
                # UPTREND LOGIC
                price_window = prcSoFar[i, -DIVERGENCE_LOOKBACK:]
                rsi_window = rsi_series_df[i].iloc[-DIVERGENCE_LOOKBACK:].to_numpy()

                if find_bearish_divergence(price_window, rsi_window) and currentPos[i] > 0:
                    currentPos[i] = 0
                    continue

                if rsi[i] < RSI_OVERBOUGHT and currentPos[i] > 0:
                    currentPos[i] = 0
                    continue
                
                if rsi[i] > prev_rsi[i] and prev_rsi[i] < PULLBACK_ENTRY_THRESHOLD:
                    target_dollars = 2000
                    currentPos[i] = round(target_dollars / prcSoFar[i, -1])

            else: # DOWNTREND LOGIC
                price_window = prcSoFar[i, -DIVERGENCE_LOOKBACK:]
                rsi_window = rsi_series_df[i].iloc[-DIVERGENCE_LOOKBACK:].to_numpy()

                if find_bullish_divergence(price_window, rsi_window) and currentPos[i] < 0:
                    currentPos[i] = 0
                    continue
                
                if rsi[i] > RSI_OVERSOLD and currentPos[i] < 0:
                    currentPos[i] = 0
                    continue
                
                if rsi[i] < prev_rsi[i] and prev_rsi[i] > (100 - PULLBACK_ENTRY_THRESHOLD):
                    target_dollars = -2000 
                    currentPos[i] = round(target_dollars / prcSoFar[i, -1])
        else:
            # RANGING MARKET - Close any open positions
            currentPos[i] = 0
            pass

    prev_rsi = rsi
    return currentPos