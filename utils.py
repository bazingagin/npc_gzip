from scipy.spatial.distance import cosine
import numpy as np
import torch
import scipy

class ToInt:
    def __call__(self, pic):
        return pic * 255

def NCD(c1, c2, c12):
    dis = (c12-min(c1,c2))/max(c1, c2)
    return dis

def CLM(c1, c2, c12):
    dis = 1 - (c1+c2-c12)/c12
    return dis

def CDM(c1, c2, c12):
    dis = c12/(c1+c2)
    return dis

def MSE(v1, v2):
    return np.sum((v1-v2)**2)/len(v1)

def agg_by_concat_space(t1, t2):
    return t1+' '+t2

def agg_by_jag_word(t1, t2, trunc=True):
    t1_list = t1.split(' ')
    t2_list = t2.split(' ')
    comb = []
    l = min([len(t1_list), len(t2_list)])
    for i in range(0,l-1,2):
        comb.append(t1_list[i])
        comb.append(t2_list[i+1])
    if len(t1_list) > len(t2_list):
        comb += t1_list[i:]
    return ' '.join(comb)

def agg_by_jag_char(t1, t2, trunc=True):
    t1_list = list(t1)
    t2_list = list(t2)
    comb = []
    l = min([len(t1_list), len(t2_list)])
    for i in range(0,l-1,2):
        comb.append(t1_list[i])
        comb.append(t2_list[i+1])
    if len(t1_list) > len(t2_list):
        comb += t1_list[i:]
    return ''.join(comb)

def agg_by_avg(i1, i2):
    return torch.div(i1+i2, 2, rounding_mode='trunc')

def agg_by_min_or_max(i1, i2, func_n):
    stacked = torch.stack([i1, i2], axis=0)
    if func_n == 'min':
        return torch.min(stacked, axis=0)[0]
    else:
        return torch.max(stacked, axis=0)[0]

def agg_by_stack(i1, i2):
    return torch.stack([i1, i2])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h