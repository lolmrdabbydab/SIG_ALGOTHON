import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)


VOLATILITY_PERIOD = 20
VOLATILITY_THRESHOLD = 0.015

SHORT_TERM_EMA_DAYS = 50
LONG_TERM_EMA_DAYS = 180

# TOOLS THRESHOLDS
RSI_SELL = 70
RSI_BUY = 30
STOCHASTIC_SELL = 80
STOCHASTIC_BUY = 20
CCI_THRESHOLD = 100

def getSMA(prices):
    # Get dimensions of price data's array
    nInst, nt = prices.shape
    
    # --- Guard Clause ---
    # If not enough data -> return empty array
    if nt < LONG_TERM_EMA_DAYS:
        return np.array([[], []])

    # --- Calculate MAs ---    
    price_history_for_ma = prices[:, -LONG_TERM_EMA_DAYS:]
    
    sma_short = np.mean(price_history_for_ma[:, -SHORT_TERM_EMA_DAYS:], axis=1)
    sma_long = np.mean(price_history_for_ma, axis=1)
    
    # --- Return Calculated MAs ---
    return np.array([sma_short, sma_long])

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

def getStochastic(prices):
    return
def getCCI(prices):
    return

def calculatePerChange(new_val, old_val):
    return ((new_val - old_val)/old_val)*100

def getVolatility(prices):

    price_history = prices[:, -VOLATILITY_PERIOD - 1]
    volatility = np.zeros(nInst)

    for n, stock in enumerate(price_history):
        per_change_arr = np.zeros(VOLATILITY_PERIOD)
        for i in range(1, VOLATILITY_PERIOD + 1):
            per_change_arr = calculatePerChange(stock[i], stock[i - 1])
        
        volatility[n] = np.std(per_change_arr)

    return volatility


def getMyPosition(prcSoFar):
    global currentPos

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
            continue

        else:
            if (short_EMA < long_EMA):
                # DOWNTREND, ONLY LOOK FOR SELLING OPPORTUNITIES
                sell = 0
                if (rsi[i] >= RSI_SELL):
                    sell += 1
                
                if (cci[i] >= CCI_THRESHOLD):
                    sell += 1
                
                if (stoc[i] >= STOCHASTIC_SELL):
                    sell += 1
                
                if sell >= 2:
                    # Conservative but profitable approach
                    base_position = 5000  # 50% of max position as base
                    currentPos[i] = -base_position * (sell / 3)  # Scale by signal strength
                    
                elif buy >= 2:
                    currentPos[i] = base_position * (buy / 3)   # Scale by signal strength
                    # calculate selling position
                
            else:
                # UPTREND, ONLY LOOK FOR BUYING OPPORTUNITIES
                buy = 0
                if (rsi[i] <= RSI_BUY):
                    buy += 1
                
                if (cci[i] <= -CCI_THRESHOLD):
                    buy += 1
                
                if (stoc[i] <= STOCHASTIC_SELL):
                    buy += 1
                
                if buy >= 2:
                    # calculate selling position
                    return 1
    

    return currentPos
            

