import pandas as pd
import numpy as np

def calc_mdd(arr):
    dd = np.zeros_like(arr)
    dd_pct = np.zeros_like(arr)
    for i in range(len(arr)-1):
        dd[i] = arr[i+1:].min() - arr[i]
        dd_pct[i] = arr[i+1:].min()/arr[i] - 1
    return {'mdd_value': dd.min(), 'mdd_pct': dd_pct.min()}

def calc_ann_ret(arr):
    ret = arr[-1]/arr[0] - 1
    n = len(arr)
    return np.power(1 + ret, 365/n) - 1

def stats(arr):
    mdd_pct = calc_mdd(arr)['mdd_pct']
    ret = arr[-1]/arr[0] - 1
    return {'ret': ret, 'mdd': mdd_pct}

def stats_yearly(s, include_total = True):
    df = pd.DataFrame()
    df['nav'] = s
    df['year'] = df.index.to_series().apply(lambda x:x[:4])
    rst = df.groupby('year').apply(lambda x: pd.Series(stats(x.nav.values)))
    if include_total:
        total = pd.Series(stats(s.values))
        total.name = 'total'
        rst = rst.append(total)
    return rst

def stats_yearly_multi_assets(df):
    rsts = []
    for i in df:
        rsts.append(stats_yearly(df[i]))
    return pd.concat(rsts,axis=1,keys=df.columns)

if __name__ == '__main__':
    #test
    pass