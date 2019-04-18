import numpy as np
import pandas as pd


def getER(df, n):
    erList = []
    for i in range(n, len(df) + 1):
        tmp = df.iloc[i-n:i]
        nom = abs(tmp.iloc[-1] - tmp.iloc[0])
        denom = tmp.diff().abs().sum()
        erList.append((nom / denom).values[0])
    return erList
    

def getP(T):
    if isinstance(T, (pd.Series, pd.DataFrame)):
        T = len(T)
    return int(np.floor( 12 * (T/100)**(0.25) ))


def autocovariance(Xi, k):
    if k < 0:
        k = np.abs(k)
    Xs = np.average(Xi)
    N = np.size(Xi)
    autoCov = 0
    for i in np.arange(0, N-k):
        autoCov += (Xi[i+k] - Xs)*(Xi[i] - Xs)
    return np.array([[ (1/N-1) * autoCov ]])


def getBlock(y, p):
    blockList = []
    # 0 <= i, j <= p
    for i in range(0, p+1, 1):
        rowList = []
        for j in range(0, p+1, 1):
            rowList.append(
                autocovariance(y.values, i-j)
            )
        blockList.append(rowList)
    return np.block(blockList)
