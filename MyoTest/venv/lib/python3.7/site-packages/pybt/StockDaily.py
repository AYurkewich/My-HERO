import numpy as np
import pandas as pd
from . import general as gn
#import general as gn

class StockDaily:
    def __init__(self, df_closeAdj,
                 df_can_buy = pd.DataFrame(), 
                 df_can_sell = pd.DataFrame()):
        self.df_closeAdj = df_closeAdj
        self.df_can_buy = df_can_buy
        self.df_can_sell = df_can_sell
        self.universe = self.df_closeAdj.columns.tolist()
        self.dates = self.df_closeAdj.index.tolist()
        
        if not self.df_can_buy.empty:
            if not gn.check_df_struct(self.df_closeAdj, self.df_can_buy):
                print('Warning: df_closeAdj and df_can_buy has different struct.')
                
        if not self.df_can_sell.empty:
            if not gn.check_df_struct(self.df_closeAdj, self.df_can_sell):
                print('Warning: df_closeAdj and df_can_sell has different struct.')
        
    def get_pct_chg(self,startd, endd):
        if not startd in self.dates:
            print('StockDaily.get_return: startd not in dates | startd = {}'.format(startd))
            return pd.Series()
        if not endd in self.dates:
            print('StockDaily.get_return: endd not in dates | startd = {}'.format(endd))
            return pd.Series()
        if startd > endd:
            print('StockDaily.get_return: startd > endd | startd = {}, endd = {}'.format(startd,endd))
            return pd.Series()
        
        prz0 = self.df_closeAdj.loc[startd,:]
        prz1 = self.df_closeAdj.loc[endd,:]
        ret = prz1 / prz0 - 1
        if not self.df_can_buy.empty:
            can_buy = self.df_can_buy.loc[startd,:]
            ret[~can_buy] = np.nan
        return ret

    def get_forward_return(self, startd, ndays, auto_fit_end = True):
        # auto_fit_end: 当 endd 超过 self.dates[-1] 的时候，是否自动设置为 self.dates[-1]
        if not startd in self.dates:
            print('StockDaily.get_forward_return: startd not in dates | startd = {}'.format(startd))
            return pd.Series()
        
        starti = self.dates.index(startd)
        endi = starti + ndays
        
        if endi > len(self.dates) - 1:
            if not auto_fit_end:
                print('StockDaily.get_forward_return: end date exceeds the range.')
                return pd.Series()
            else:
                endi = len(self.dates) - 1
        
        endd = self.dates[endi]
        return self.get_pct_chg(startd,endd)