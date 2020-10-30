import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pymorphy2

stat_data = np.load('stat_data.npy')
signs_list = np.load('signs_list.npy')
chars_list = np.load('chars_list.npy')

groups = [i for i in range(len(chars_list))]
counts = np.array([num.sum(axis=0) for num in stat_data])
countsl = [num.sum(axis=0) for num in stat_data]
for i in range(len(counts)):
    try:
        if countsl[i] > 10000:
            countsl.pop(i)
            groups.pop(-1)
    except Exception:
        pass
plt.bar(groups, countsl)
