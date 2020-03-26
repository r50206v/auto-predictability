import sys
import numpy as np
import pandas as pd
import scipy.stats as stats


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
    return np.array([[ 1/(N-1) * autoCov ]])


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


def _shift(df):
    constant = df[df > 0].mean().values[0]
    if all(df < 0):
        constant = 1
    if df.min().values[0] <= 0:
        df = df + (-1)*df.min().values[0] + constant
    return df


def boxcox(df):
    df = _shift(df)
    try:
        y, lam, ci = stats.boxcox(df, alpha=0.05)
    except RuntimeError: 
        # when boxcox cannot converge
        sys.stderr.write('[WARNING] BoxCox cannot converge\n')
        return np.log(df)
    
    if (ci[0] > lam) or (ci[1] < lam):
        # if the lambda is out of the confidence interval
        # return constant value 
        # (`get` will use logarithm as an alternative)
        sys.stderr.write('[WARNING] lambda not in confidence interval\nlambda: {0}, CI: {1}\n'.format(lam, ci))
        return np.log(df)
    return pd.DataFrame(y, index=df.index, columns=df.columns)