import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)


VOLATILITY_PERIOD = 20
VOLATILITY_THRESHOLD = 0.15

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50


# TOOLS THRESHOLDS
RSI_SELL = 70
RSI_BUY = 30
STOCHASTIC_WINDOW = 14
STOCHASTIC_SELL = 80
STOCHASTIC_BUY = 20
CCI_WINDOW = 20
CCI_THRESHOLD = 100

def getEMA(prices):
    nInst, nt = prices.shape
    
    if nt < LONG_TERM_EMA_DAYS:
        return np.array([[], []])

    prices_df = pd.DataFrame(prices.T)
    
    ema_short = prices_df.ewm(span=SHORT_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    ema_long = prices_df.ewm(span=LONG_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    
    return np.array([ema_short.to_numpy(), ema_long.to_numpy()])

def getRSI(prices, window=14):
    nInst, nt = prices.shape
    
    if nt < window + 1:
        return np.full(nInst, 50.0) # 50 = neutral RSI value

    prices_df = pd.DataFrame(prices.T)
    delta = prices_df.diff()

    # DF that hold pos changes from 'delta' & set all neg changes to 0
    gain = (delta.where(delta > 0, 0)).fillna(0)    
    # DF that hold negative changes (but made into positive value for calculation) & set pos changes to 0.
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1].fillna(50.0).to_numpy() # .iloc[-1] -> take last day (current day) | .fillna(50.0) -> safeguard empty val w neutral val

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
        return np.full(nInst, 0.0) # Return a neutral CCI of 0.

    prices_df = pd.DataFrame(prices.T)
    
    typical_price = prices_df

    sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
    mean_dev = (typical_price - sma_tp).abs().rolling(window=window, min_periods=window).mean()

    cci = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-6)
    
    return cci.iloc[-1].fillna(0.0).to_numpy()


def calculatePerChange(new_val, old_val):
    return ((new_val - old_val)/old_val)*100

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

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    # --- Guard Clause ---
    # If there isn't enough data for the long-term EMA, return the current positions without change.
    if nt < LONG_TERM_EMA_DAYS:
        return currentPos
    
    # All the tools
    vol = getVolatility(prcSoFar)
    rsi = getRSI(prcSoFar)
    cci = getCCI(prcSoFar)
    stoc = getStochastic(prcSoFar)
    EMA = getEMA(prcSoFar)
    short_EMA = EMA[0]
    long_EMA = EMA[1]


    for i in range(nInst):
        if vol[i] < VOLATILITY_THRESHOLD:
            currentPos[i] = 0
            continue

        currPrice = prcSoFar[i][-1]
        vol_boost = vol[i] / VOLATILITY_THRESHOLD  # >1 if volatility is high

        if short_EMA[i] < long_EMA[i]:  # Downtrend
            sell = 0
            if rsi[i] >= RSI_SELL:
                sell += 1
            if cci[i] >= CCI_THRESHOLD:
                sell += 1
            if stoc[i] >= STOCHASTIC_SELL:
                sell += 1

            if sell >= 2:
                base_position = 1500
                currentPos[i] = (base_position * (sell / 3) )/currPrice

        else:  # Uptrend
            buy = 0
            if rsi[i] <= RSI_BUY:
                buy += 1
            if cci[i] <= -CCI_THRESHOLD:
                buy += 1
            if stoc[i] <= STOCHASTIC_BUY:
                buy += 1

            if buy >= 2:
                base_position = 1500
                currentPos[i] = -(base_position * (buy / 3))/currPrice

    return currentPos
            