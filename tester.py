import numpy as np
import pandas as pd

SHORT_TERM_EMA_DAYS = 20
LONG_TERM_EMA_DAYS = 50

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def getSMA(prices):
    """
    - Row 0: Short-term SMAs for all 50 stocks.
    - Row 1: Long-term SMAs for all 50 stocks.
    """
    # Get the dimensions of the price data array
    nInst, nt = prices.shape
    
    # If there isn't enough historical data, return nothing or an empty array.
    if nt < LONG_TERM_EMA_DAYS:
        return np.array([[], []]) # Return an empty (2, 0) array

    # --- B. Calculate the Moving Averages ---
    
    # To calculate the SMAs for the most recent day, we only need the
    # last `LONG_TERM_MA_DAYS` worth of prices.
    price_history_for_ma = prices[:, -LONG_TERM_EMA_DAYS:] # last longterm days of each stock, price[i] is the ith stock's last 50 vals

    # Calculate the short-term SMA for each stock
    # We take the last SHORT_TERM_MA_DAYS from our sliced history
    sma_short = np.mean(price_history_for_ma[:, -SHORT_TERM_EMA_DAYS:], axis=1)
    
    # Calculate the long-term SMA for each stock
    sma_long = np.mean(price_history_for_ma, axis=1)
    
    # --- C. Return the Calculated MAs ---
    # Stack the two arrays into a single (2, 50) NumPy array.
    print(sma_long, '\n\n\nSHORT: \n\n\n' ,sma_short)
    return np.array([sma_short, sma_long])


def getEMA(prices):
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
    print(ema_short)
    # Calculate the long-term EMA for each stock
    ema_long = prices_df.ewm(span=LONG_TERM_EMA_DAYS, adjust=False).mean().iloc[-2]
    print('\n\n\n',ema_long)
    # --- C. Return the Calculated MAs ---
    # Stack the two pandas Series and convert them back to a NumPy array.
    return np.array([ema_short.to_numpy(), ema_long.to_numpy()])

prices = loadPrices('prices.txt')
ma = getEMA(prices)
