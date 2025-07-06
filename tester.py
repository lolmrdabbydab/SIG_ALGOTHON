import pandas as pd
import numpy as np

VOLATILITY_PERIOD = 20

def getVolatility(prices):

    price_history = prices[:, -(VOLATILITY_PERIOD + 1):]
   
    volatility = np.zeros(nInst)

    for n, stock in enumerate(price_history):
        per_change_arr = np.zeros(VOLATILITY_PERIOD)
        for i in range(1, VOLATILITY_PERIOD + 1):
            per_change_arr[i - 1] = calculatePerChange(stock[i], stock[i - 1])
        
        volatility[n] = np.std(per_change_arr)

    return volatility


def calculatePerChange(new_val, old_val):
    return ((new_val - old_val)/old_val)*100


def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T
   
prices = loadPrices("prices.txt")
print(getVolatility(prices))