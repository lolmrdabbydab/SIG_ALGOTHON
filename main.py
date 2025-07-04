import pandas as pd
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

threshold = 0.35
short_indicator = 65
buy_indicator = 35
scaleFactor = 100

# --- NEW PARAMETER FOR NORMALIZATION ---
# This is the % difference between EMAs that we consider a 'maximum strength' signal.
# A good starting point is 2.0. You can tune this value.
MAX_TREND_STRENGTH = 2.0

SHORT_TERM_EMA_DAYS = 50
LONG_TERM_EMA_DAYS = 180

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


crossed = np.zeros((2,50)) # check if the short (0) and long(1) rsi indicators have been crossed

# ======================================================================================
#                                   --- getMyPosition() ---
# ======================================================================================
def getMyPosition(prcSoFar):
    global currentPos
    global crossed

    EMAs = getEMA(prcSoFar)
    # prcsofar is 50 rows and 750 columns 

    if EMAs.shape[1] == 0:
        # Not enough data, so we can't make a decision.
        # Return the last known position without making any trades.
        return currentPos
    
    rsi_vals = getRSI(prcSoFar)
    
    short_MA = EMAs[0]
    long_MA = EMAs[1]
    
    # last day prices yea 
    last = prcSoFar[:, -1]

    
    # evaluate signals
    for i in range(nInst):
        per_change = percentageChange(short_MA[i], long_MA[i])
        
        # short signal
        if (short_MA[i] < long_MA[i]) and (per_change > threshold):
            if rsi_vals[i] > short_indicator: # 60
                if crossed[0][i] == 0:
                    crossed[0][i] = 1

            else:
                if crossed[0][i] == 1:
                    # Calculate a normalized strength score (0 to 1)
                    normalized_strength = min(per_change, MAX_TREND_STRENGTH) / MAX_TREND_STRENGTH
                    # Determine the dollar amount for the position based on strength
                    dollar_position = normalized_strength * 10000

                    # rsi indicator to short, as the value went up and now its coming down again
                    currentPos[i] = round((dollar_position / last[i])) # the smaller shortMA is than longMA the more stocks we borrow (short) 
                    crossed[0][i] = 0
            
        # long signal
        elif (short_MA[i] > long_MA[i]) and (per_change > threshold):
            if rsi_vals[i] < buy_indicator: # 30
                if crossed[1][i] == 0:
                    crossed[1][i] = 1
                else:
                    if crossed[1][i] == 1:
                        normalized_strength = min(per_change, MAX_TREND_STRENGTH) / MAX_TREND_STRENGTH
                    # Determine the dollar amount for the position based on strength
                        dollar_position = normalized_strength * 10000
                    # Convert to number of shares
                        currentPos[i] = -round(dollar_position / last[i])
                        # rsi indicator to buy, as the upward trend is continuing
                        crossed[1][i] = 0

        else: # per change less than threshold - no buy/sell zone (no clear trend)
            currentPos[i] = 0

    return currentPos


# ======================================================================================
#                                   --- Helper Function ---
# ======================================================================================
def percentageChange(x, y):
    return abs(((x - y)/y)*100)