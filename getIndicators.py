import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import _pickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions import *


'''
calculating some predictability indicators for several assets
1) Auto-Predictability 
2) Efficient Ratio
'''

n = 5 # for efficient ratio
probMap, erMap = {}, {}
path = sys.path[0]
fileList = os.listdir(path + '/dataset/')
sys.stderr.write('start processing %d files from %s' % (len(fileList), path+'/dataset/'))
for fileName in tqdm(fileList):
    stat_id, name = fileName.split('-', 1)
    name = name.replace('.csv', '')

    data = pd.read_csv(path + '/dataset/' + fileName, index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    
    start_date = '2009-07-01' #'2005-12-31'
    end_date = data.index[-1].strftime('%Y-%m-%d') #'2018-01-01'
    data = data.loc[(data.index > start_date) & (data.index <= end_date)] #& (data.index < '2018-01-02')


    # calculate Efficiency Ratio
    er = getER(data, n)
    erMap[name] = np.nanmean(er)


    # boxcox transformation
    data = boxcox(data)

    # calculate Maximum Entropy
    data = data.pct_change()\
               .dropna()
    p = getP(data[stat_id])
    nom = getBlock(data[stat_id], p)
    denom = getBlock(data[stat_id], p - 1)
    gamma = getBlock(data[stat_id], 0)[0][0]
    prob = np.log2(gamma / (np.linalg.det(nom) / np.linalg.det(denom)))
    probMap[name] = prob


    # pickle results
    file_dst = path + '/result/%s-%s.pkl' % (stat_id, name)
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
        # probMap[name] = pickle.load(f)['prob'] 
        # erMap[name] = np.nanmean(pickle.load(f)['er'])


data = pd.DataFrame(probMap, index=['value']).sort_values(by=['value'], axis=1)
plt.figure(figsize=(15, 8))
plt.barh(np.arange(len(data.columns)), data.values[0], 0.35)
plt.yticks(np.arange(len(data.columns)), data.columns.values)
plt.xticks(np.arange(0, 1, 0.05))
plt.title('Maximum Extropy - %s~%s' % (start_date, end_date))
plt.xlabel('auto predictability')
plt.grid(axis='x')
plt.margins(y=0.01)
plt.autoscale()
plt.tight_layout()
plt.savefig(path + '/auto-predictability.png')


data = pd.DataFrame(erMap, index=['value']).sort_values(by=['value'], axis=1)
plt.figure(figsize=(15, 8))
plt.barh(np.arange(len(data.columns)), data.values[0], 0.35)
plt.yticks(np.arange(len(data.columns)), data.columns.values)
plt.xticks(np.arange(0, 1, 0.05))
plt.title('Efficiency Ratio - %s~%s' % (start_date, end_date))
plt.xlabel('efficiency ratio (n=%d)' % n)
plt.grid(axis='x')
plt.margins(y=0.01)
plt.autoscale()
plt.tight_layout()
plt.savefig(path + '/efficiency-ratio.png')