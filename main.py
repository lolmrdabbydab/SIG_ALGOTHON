import pandas as pd
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

threshold = 0.2

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50

# ======================================================================================
#                                  --- Signal Calculation ---
# ======================================================================================
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

# ======================================================================================
#                                   --- getMyPosition() ---
# ======================================================================================
def getMyPosition(prcSoFar):
    global currentPos
    
    EMAs = getEMA(prcSoFar)
    # prcsofar is 50 rows and 750 columns 

    if EMAs.shape[1] == 0:
        # Not enough data, so we can't make a decision.
        # Return the last known position without making any trades.
        return currentPos
    
    short_MA = EMAs[0]
    long_MA = EMAs[1]
    
    # last day prices
    last = prcSoFar[:, -1]

    # evaluate signals
    for i in range(nInst):
        per_change = percentageChange(short_MA[i], long_MA[i])
        
        # short signal
        if (short_MA[i] < long_MA[i]) and (per_change > threshold):
            currentPos[i] = round(-( (1 - short_MA[i]/long_MA[i]) * 10000 )/last[i]) # the smaller shortMA is than longMA the more stocks we borrow (short) 
        
        # long signal
        elif (short_MA[i] > long_MA[i]) and (per_change > threshold):
            currentPos[i] = round(( (1 - long_MA[i]/short_MA[i]) * 10000 )/last[i]) # the shorter longMA is the lesser the ratio is and the more 
    
    return currentPos
    
def percentageChange(x, y):
    return abs(((x - y)/y)*100)