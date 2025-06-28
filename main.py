# Implementation of a Simple Moving Average (SMA) Crossover strategy.

import numpy as np

# ======================================================================================
# --- 1. Global Parameters for Easy Tuning ---
# ======================================================================================
# You can easily change these values to test different MA combinations later.

SHORT_TERM_MA_DAYS = 20
LONG_TERM_MA_DAYS = 50

# The dollar amount to allocate for each position. We'll use the max allowed.
DOLLAR_ALLOCATION = 10000

# ======================================================================================
# --- 2. Main Function (Required by the Competition) ---
# ======================================================================================

def getMyPosition(prices):
    """
    Calculates the desired position based on an SMA crossover.
    - If the short-term MA is above the long-term MA, we go long.
    - If the short-term MA is below the long-term MA, we go short.
    
    This function is stateless and returns the absolute target position each day.
    """
    # Get the dimensions of the price data array
    nInst, nt = prices.shape
    
    # --- A. Guard Clause ---
    # If there isn't enough historical data to calculate the longest moving average,
    # we return a position of all zeros to avoid errors.
    if nt < LONG_TERM_MA_DAYS:
        return np.zeros(nInst, dtype=int)

    # --- B. Calculate the Moving Averages ---
    
    # To calculate the SMAs for the most recent day, we only need the
    # last `LONG_TERM_MA_DAYS` worth of prices.
    price_history_for_ma = prices[:, -LONG_TERM_MA_DAYS:]

    # Calculate the short-term SMA for each stock
    # We take the last SHORT_TERM_MA_DAYS from our sliced history
    sma_short = np.mean(price_history_for_ma[:, -SHORT_TERM_MA_DAYS:], axis=1)
    
    # Calculate the long-term SMA for each stock
    sma_long = np.mean(price_history_for_ma, axis=1)

    # --- C. Generate the Trading Signal (Long/Short) ---
    
    # Create a signal vector: +1 for long, -1 for short.
    # np.where is an efficient way to do this for all 50 stocks at once.
    # If sma_short > sma_long, signal is 1 (long). Otherwise, it's -1 (short).
    long_short_signal = np.where(sma_short > sma_long, 1, -1)

    # --- D. Calculate Target Positions in Shares ---
    
    # Get the latest price for each stock to convert dollars to shares
    latest_prices = prices[:, -1]
    
    # Calculate the target dollar position for each stock
    target_dollar_positions = DOLLAR_ALLOCATION * long_short_signal
    
    # Convert the dollar amount to the number of shares
    # The evaluator will automatically clip this to the $10k limit if the
    # calculation is slightly off, but this gets us the target share count.
    target_positions = target_dollar_positions / latest_prices
    
    # --- E. Return Final Positions ---
    # The function must return a NumPy vector of integers.
    return target_positions.astype(int)