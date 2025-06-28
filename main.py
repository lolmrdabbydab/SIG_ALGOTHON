# main.py
# This version is specifically for calculating and returning the SMA values.
# The output of this function will be used by a teammate's code to determine trade logic.

import numpy as np

# ======================================================================================
# --- 1. Global Parameters for Easy Tuning ---
# ======================================================================================
# Your team can easily change these values to test different MA combinations.

SHORT_TERM_MA_DAYS = 20
LONG_TERM_MA_DAYS = 50

# ======================================================================================
# --- 2. Main Function (Modified for Your Task) ---
# ======================================================================================

def getMyPosition(prices):
    """
    NOTE: For the final submission, this function must return a vector of integer
    share positions. This modified version is for the team's internal workflow,
    where this function's role is to ONLY calculate and return the moving averages.
    The output will be a (2, 50) NumPy array:
    - Row 0: Short-term SMAs for all 50 stocks.
    - Row 1: Long-term SMAs for all 50 stocks.
    """
    # Get the dimensions of the price data array
    nInst, nt = prices.shape
    
    # --- A. Guard Clause ---
    # If there isn't enough historical data, return nothing or an empty array.
    # Returning two empty arrays is a clean way to handle this.
    if nt < LONG_TERM_MA_DAYS:
        return np.array([[], []]) # Return an empty (2, 0) array

    # --- B. Calculate the Moving Averages ---
    
    # To calculate the SMAs for the most recent day, we only need the
    # last `LONG_TERM_MA_DAYS` worth of prices.
    price_history_for_ma = prices[:, -LONG_TERM_MA_DAYS:]

    # Calculate the short-term SMA for each stock
    # We take the last SHORT_TERM_MA_DAYS from our sliced history
    sma_short = np.mean(price_history_for_ma[:, -SHORT_TERM_MA_DAYS:], axis=1)
    
    # Calculate the long-term SMA for each stock
    sma_long = np.mean(price_history_for_ma, axis=1)
    
    # --- C. Return the Calculated MAs ---
    # Stack the two arrays into a single (2, 50) NumPy array.
    return np.array([sma_short, sma_long])