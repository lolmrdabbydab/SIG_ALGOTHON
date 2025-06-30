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