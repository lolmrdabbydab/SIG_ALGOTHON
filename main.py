# main.py
import pandas as pd
import numpy as np

# ======================================================================================
# --- Parameters ---
# ======================================================================================
nInst = 50

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50

THRESHOLD_PERCENT = 0.2

DOLLAR_LIMIT = 10000

# ======================================================================================
#                               --- Helper Functions ---
# ======================================================================================

def getEMA(prices, lookback):
    prices_df = pd.DataFrame(prices.T)
    return prices_df.ewm(span=lookback, adjust=False).mean().iloc[-1].to_numpy()

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

# ======================================================================================
#                               --- getMyPosition() ---
# ======================================================================================

def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    # Create default position vector of all zeros
    positions = np.zeros(nInst)

    # --- Guard Clause ---
    # If there isn't enough data, return vector positions of zero
    if nt < LONG_TERM_EMA_DAYS:
        return positions.astype(int)

    # --- Calculate EMA Signals ---
    ema_short = getEMA(prcSoFar, SHORT_TERM_EMA_DAYS)
    ema_long = getEMA(prcSoFar, LONG_TERM_EMA_DAYS)

    # --- Trading Logic ---
    # Calculate percentage difference for all 50 stocks at once
    percentage_diff = np.abs(((ema_short - ema_long) / ema_long) * 100)

    # Boolean var to identify which stocks is tradable
    is_above_threshold = percentage_diff > THRESHOLD_PERCENT
    long_signal_mask = (ema_short > ema_long) & is_above_threshold
    short_signal_mask = (ema_short < ema_long) & is_above_threshold
    
    # --- Position Sizing ---
    latest_prices = prcSoFar[:, -1]

    # For stocks with a long signal, calculate their proportional size
    if np.any(long_signal_mask):
        long_ratios = 1 - (ema_long[long_signal_mask] / ema_short[long_signal_mask])
        dollar_allocations_long = long_ratios * DOLLAR_LIMIT
        positions[long_signal_mask] = dollar_allocations_long / latest_prices[long_signal_mask]

    # For stocks with a short signal, calculate their proportional size
    if np.any(short_signal_mask):
        short_ratios = 1 - (ema_short[short_signal_mask] / ema_long[short_signal_mask])
        dollar_allocations_short = -1 * short_ratios * DOLLAR_LIMIT
        positions[short_signal_mask] = dollar_allocations_short / latest_prices[short_signal_mask]
        
    return positions.astype(int)