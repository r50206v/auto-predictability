import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import _pickle as pickle
import warnings
warnings.filterwarnings('ignore')
from functions import *


'''
calculating some predictability indicators for several assets
1) Auto-Predictability 
2) Efficient Ratio
'''
start_date, end_date = None, None
if len(sys.argv) == 2:
    start_date = sys.argv[1]
elif len(sys.argv) == 3:
    start_date = sys.argv[1]
    end_date = sys.argv[2]


n = 8 # for efficient ratio
probMap = dict({k: {} for k in ['外匯','先進國家','新興市場','ETF','原物料']})
erMap = dict({k: {} for k in ['外匯','先進國家','新興市場','ETF','原物料']})
path = sys.path[0]
fileList = sorted(os.listdir(path + '/dataset/'))
sys.stderr.write('start processing %d files from %s' % (len(fileList), path+'/dataset/'))
for fileName in tqdm(fileList):
    stat_id, category, name = fileName.split('-', 2)
    name = name.replace('.csv', '')

    data = pd.read_csv(path + '/dataset/' + fileName, index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    
    if start_date is None:
        start_date = '2009-07-01' #'2005-12-31'
    if end_date is None:
        end_date = data.index[-1].strftime('%Y-%m-%d') #'2018-01-01'
    data = data.loc[(data.index > start_date) & (data.index <= end_date)]


    # calculate Efficiency Ratio
    er = getER(data, n)
    erMap[category][name] = np.nanmean(er)


    # boxcox transformation
    data = data.pct_change()\
               .dropna()
    # data = boxcox(data)
    # data = data.loc[(data != 0).any(1)]

    # calculate Maximum Entropy
    p = getP(data[stat_id])
    nom = getBlock(data[stat_id], p)
    denom = getBlock(data[stat_id], p - 1)
    gamma = getBlock(data[stat_id], 0)[0][0]
    prob = np.log2(gamma / (np.linalg.det(nom) / np.linalg.det(denom)))
    probMap[category][name] = prob if pd.notnull(prob) else 0


    # pickle results
    file_dst = path + '/result/%s-%s-%s.pkl' % (stat_id, category, name)
    with open(file_dst, 'wb') as f:
        pickle.dump({
            'p': p,
            'nom': nom,
            'denom': denom,
            'gamma': gamma,
            'prob': prob,
            'er': er,
        }, f, protocol=True)
    # with open(file_dst, 'rb') as f:
    #     # probMap[category][name] = pickle.load(f)['prob'] 
    #     erMap[category][name] = np.nanmean(pickle.load(f)['er'])


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


for category, probDict in probMap.items():
    data = pd.DataFrame(probDict, index=['value']).sort_values(by=['value'], axis=1)
    plt.figure(figsize=(15, 8))
    plt.barh(np.arange(len(data.columns)), data.values[0], 0.35)
    plt.yticks(np.arange(len(data.columns)), data.columns.values)
    plt.xticks(np.arange(0, 1, 0.05))
    plt.title('%s: Auto Predictability - %s~%s' % (category, start_date, end_date))
    plt.xlabel('auto predictability')
    plt.grid(axis='x')
    plt.margins(y=0.01)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(path + '/%s-auto-predictability-%s-%s.png' % (category, start_date, end_date))


for category, erDict in erMap.items():
    data = pd.DataFrame(erDict, index=['value']).sort_values(by=['value'], axis=1)
    plt.figure(figsize=(15, 8))
    plt.barh(np.arange(len(data.columns)), data.values[0], 0.35)
    plt.yticks(np.arange(len(data.columns)), data.columns.values)
    plt.xticks(np.arange(0, 1, 0.05))
    plt.title('%s: Efficiency Ratio - %s~%s' % (category, start_date, end_date))
    plt.xlabel('efficiency ratio (n=%d)' % n)
    plt.grid(axis='x')
    plt.margins(y=0.01)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(path + '/%s-efficiency-ratio-%s-%s.png' % (category, start_date, end_date))