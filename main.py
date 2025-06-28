# main.py
# This version calculates EXPONENTIAL MOVING AVERAGES (EMA).
# The output of this function will be used by a teammate's code to determine trade logic.

import numpy as np
import pandas as pd # Pandas is used for its efficient EMA calculation

# ======================================================================================
# --- 1. Global Parameters for Easy Tuning ---
# ======================================================================================
# Your team can easily change these values to test different MA combinations.

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50

# ======================================================================================
# --- 2. Main Function (Modified for Your Task) ---
# ======================================================================================

def getMyPosition(prices):
    """
    NOTE: For the final submission, this function must return a vector of integer
    share positions. This modified version is for the team's internal workflow,
    where this function's role is to ONLY calculate and return the moving averages.
    The output will be a (2, 50) NumPy array:
    - Row 0: Short-term EMAs for all 50 stocks.
    - Row 1: Long-term EMAs for all 50 stocks.
    """
    # Get the dimensions of the price data array
    nInst, nt = prices.shape
    
    # --- A. Guard Clause ---
    # If there isn't enough historical data, return an empty array.
    if nt < LONG_TERM_EMA_DAYS:
        return np.array([[], []]) # Return an empty (2, 0) array

    # --- B. Calculate the Exponential Moving Averages (EMA) ---
    
    # Convert the entire price history to a pandas DataFrame to calculate EMA.
    # EMA is more stable when calculated over a longer series.
    # We transpose (.T) so that each stock is a column.
    prices_df = pd.DataFrame(prices.T)
    
    # Calculate the short-term EMA for each stock.
    # .ewm() creates an exponentially weighted moving window.
    # 'span' is the standard way to define the EMA period.
    # We take the last row of values (.iloc[-1]) which corresponds to today's EMA.
    ema_short = prices_df.ewm(span=SHORT_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    
    # Calculate the long-term EMA for each stock
    ema_long = prices_df.ewm(span=LONG_TERM_EMA_DAYS, adjust=False).mean().iloc[-1]
    
    # --- C. Return the Calculated MAs ---
    # Stack the two pandas Series and convert them back to a NumPy array.
    return np.array([ema_short.to_numpy(), ema_long.to_numpy()])