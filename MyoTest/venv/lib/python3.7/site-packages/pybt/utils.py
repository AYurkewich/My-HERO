import pandas as pd
import numpy as np
from scipy.stats import rankdata
from . import StockDaily

def grouping1d(arr, nog, ST = 999):
    # 一维数组的分组
    # nan 分到第 0 组，ST 分到第 -1 组
    rst = np.zeros_like(arr, dtype = int)
    
    rnk = rankdata(arr)
    
    num_total = len(arr)
    num_nan = np.isnan(arr).sum()
    is_ST = np.array(pd.Series(abs(arr-ST))<0.000001)  # use Series to avoid RuntimeWarning
    num_ST = (is_ST).sum()
    n = num_total - num_nan - num_ST  # 除了 nan 和 ST 之外一共有多少元素
    d = round(n/nog)
    
    # 第 1，2，3，... ，nog -1 组
    for i in range(1,nog):
        r1 = 1 + (i-1)*d
        r2 = r1 + d
        rst[(rnk >= r1) & (rnk < r2)] = i
    
    # 最后一组
    r1 = 1 + (nog-1)*d
    r2 = n
    rst[(rnk >= r1) & (rnk <= r2)] = nog
    
    rst[is_ST] = -1
    
    return rst

def get_clean_factor_and_forward_returns(factor:'s',prices:'df',bins = 5,periods=(20,60)):
    factor_data = pd.DataFrame(factor)
    factor_data.columns=['factor']
    
    sd = StockDaily.StockDaily(prices)
    
    for p in periods:
        factor_data[str(p) + 'D'] = np.nan
    
    for date in factor_data.index.levels[0]:
        factor_data.loc[(date,),'factor_quantile'] = grouping1d(factor_data.loc[date].factor.values,bins)
        for p in periods:
            factor_data.loc[(date,), str(p) + 'D'].loc[:] = sd.get_forward_return(date,p)
    
    factor_data.factor_quantile = factor_data.factor_quantile.astype(int)
    return factor_data

def convert_date_in_factor_data(factor_data):
    factor_data.reset_index(inplace = True)
    factor_data.rename(columns={'level_0':'date','level_1':'asset'}, inplace = True)
    factor_data.date = pd.to_datetime(factor_data.date)
    factor_data.set_index(['date','asset'],inplace = True)
    return None

if __name__ == '__main__':
#    import alphalens
    df = pd.read_csv('Data/pe.csv',header=None)
    df.columns = ['date','asset','factor']
    df.date = df.date.astype(str)
    factor = pd.Series(df.factor.values, pd.MultiIndex.from_arrays([df.date.values,df.asset.values]))
    factor = factor
    factor_data = get_clean_factor_and_forward_returns(factor,prices)
    factor_data = factor_data.reset_index().rename(columns={'level_0':'date','level_1':'asset'})
    factor_data.date = pd.to_datetime(factor_data.date)
    factor_data = factor_data.set_index(['date','asset'])
#    alphalens.tears.create_full_tear_sheet(factor_data)

    
