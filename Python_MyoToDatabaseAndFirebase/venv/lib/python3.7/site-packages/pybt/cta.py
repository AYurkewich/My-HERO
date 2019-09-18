import numpy as np
import pandas as pd
from . import talib
from operator import methodcaller

def get_pos_1l(prz:'arr', line:'arr'):
    pos = np.zeros_like(prz)
    pos[prz > line] = 1
    pos[prz <= line] = -1
    return pos

def get_pos_3l(prz:'arr', upper:'arr', lower:'arr', mid:'arr'):
    pos = np.zeros_like(prz)
    
    for i in range(1,len(prz)):
        if np.isnan(upper[i]): continue
            
        if pos[i-1] == 0:
            if prz[i] > upper[i]:# and price[i-1] < upper[i-1]:
                pos[i] = 1
            elif prz[i] < lower[i]:# and price[i-1] > lower[i-1]:
                pos[i] = -1
            else:
                pos[i] = pos[i-1]

        elif pos[i-1] == 1:
            if prz[i] < mid[i]:
                if prz[i] >= lower[i]:   # 未下穿 lower
                    pos[i] = 0
                else:
                    pos[i] = -1
            else:
                pos[i] = pos[i-1]
                
        elif pos[i-1] == -1:
            if prz[i] > mid[i]:
                if prz[i] <= upper[i]:
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                prz[i] = prz[i-1]
    return pos


def _get_pos(prz:'arr', st:'tuple'):
    st_name = st[0]
    st_params = st[1]
    ta = methodcaller(st_name, prz, *st_params)(talib)    
    if st_name in ['boll', 'don']:
        pos = get_pos_3l(prz, ta['upper'], ta['lower'], ta['mid'])
    else:
        pos = np.array()
    return pos

def get_pos(prz:'arr', st):
    if type(st)==tuple:
        if prz.ndim == 1:
            return _get_pos(prz,st)
        else:
            pos = np.zeros_like(prz)
            for i in range(pos.shape[1]):
                pos[:,i] = _get_pos(prz[:,i],st)
            return pos
    elif type(st)==list:
        return [_get_pos(prz,st_) for st_ in st]
    else:
        return []