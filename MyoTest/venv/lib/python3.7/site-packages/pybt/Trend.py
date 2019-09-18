#import sys
#sys.path.append('D:\MyCloud\Research')

from pandas import *
import numpy as np
#import QTool as qtool
from scipy import stats
#import TA

'''data for test'''
#s = read_excel('Data/test.xlsx','HS300',index_col='date')['HS300']
#s = read_excel('Data/test.xlsx','rb',index_col='date')['close']
#prices = read_excel('Data/swidata.xlsx','swi1',index_col='date',skiprows=1)
#s = read_excel('Data/japan.xlsx','index',index_col='date')['日经225']
s = Series([58.90,51.53,50.23,56.43,64.48,51.82,36.13,47.64,57.88,52.50,53.62,61.95,77.86,71.11,81.03,95.18,96.38,121.58,138.58])
#holc = DataFrame({'high':s+5,'open':s-1,'low':s-5,'close':s})
#holc = read_csv('rb000.csv')
#price = holc.close
#st1 = ('boll',[20,2])
#st2 = ('ma',[20])
#st2 = ('break',[20])
'''-------------'''




def dt(s,interval=1,fee=0,invest=1,show=True):
    #定投回测
    df = DataFrame()
    df['price'] = s
    df['nav_price'] = df.price/df.price[0]
    invest = Series(data=0,index=df.index)
    invest[np.mod(range(len(df)),interval)==0] = 1
    df['invest'] = invest
    df['shs'] = df.invest/df.nav_price
    df['cum_shs'] = df.shs.cumsum()
    df['cum_invest'] = df.invest.cumsum()
    df['cum_mv'] = df.cum_shs * df.nav_price
    df['cum_pnl'] = df.cum_mv - df.cum_invest
    df['raw_rtn'] = df.nav_price - 1
    df['dt_rtn'] = df.cum_mv / df.cum_invest - 1
    if show:
        df[['raw_rtn','dt_rtn']].plot()
    return df


class rs():  #resample
        
    def fixed_interval(s,interval,fromTop=True):
        if fromTop:
            ss = s[np.mod(range(len(s)),interval)==0]    
        else:
            ss = s[np.mod(np.array(range(len(s))) + interval - np.mod(len(s)-1,interval),interval)==0]    
        return ss

    def fixed_vola(s,th):
        l_index = []
        l_price = []
        k = 0
        l_index.append(s.index[0])
        l_price.append(s[0])
        flag_end = False
        while not flag_end:
            for i in range(k,len(s)):
                if abs(s[i]/s[k]-1) >= th:
                    l_index.append(s.index[i])
                    l_price.append(s[i])
                    k = i
                    break
            if i >= len(s)-1:
                flag_end = True
        return Series(index=l_index,data=l_price)
                
                
class ti():
    def ma(s,N):
        df = DataFrame()
        df['price'] = s
        df['ma'] = df.price.rolling(N).mean()
        return df
    
    def ma_leading(s,N):
        df = DataFrame()
        df['price'] = s
        df['ma'] = df.price.rolling(N).mean().shift()
        return df
    
    def maAlign(s,maLength):
        df = DataFrame()
        df['price'] = s
        for i in range(len(maLength)):
            N = maLength[i]
            df['ma'+str(N)] = df.price.rolling(N).mean()    
        for i in range(1,len(maLength)):
            N1 = maLength[i-1]
            N2 = maLength[i]
            df['ma'+str(N1) + '-ma' +str(N2)] = 100*(df['ma'+str(N1)] / df['ma'+str(N2)] - 1)
        return df
    
    def boll(s,N,offset):
        df = DataFrame()
        df['price'] = s
        df['mid'] = df.price.rolling(N).mean()
        df['std'] = df.price.rolling(N).std()
        df['upper'] = df['mid'] + offset * df['std']
        df['lower'] = df['mid'] - offset * df['std']
        del df['std']
        return df
    
    def boll_leading(s,N,offset):
        df = DataFrame()
        df['price'] = s
        df['mid'] = df.price.rolling(N).mean().shift()
        df['std'] = df.price.rolling(N).std().shift()
        df['upper'] = df['mid'] + offset * df['std']
        df['lower'] = df['mid'] - offset * df['std']
        del df['std']
        return df

    def don(s,N):    
        df = DataFrame()
        df['price'] = s
        df['upper'] = df.price.rolling(N).max().shift()
        df['lower'] = df.price.rolling(N).min().shift()
        df['mid'] = (df.upper + df.lower)/2
        return df

    def brk(s,N):
        df = DataFrame()
        df['price'] = s
        df['level'] = df.price.shift(N)
        return df
    
    def cntmaupx(s,M): #count ma up cross
        df = DataFrame({'count':0,1:s})
        for i in range(2,M+1): 
            df[i]=s.rolling(i).mean()
            df['count'] += np.sign((df[i-1]-df[i]).fillna(0))
        return df
    
    def slope(s,N):
        df = DataFrame()
        df['price'] = s
        slp = Series(data=np.nan,index=s.index)
        x = np.array(range(1,N+1))
        for i in range(N-1,len(df)):
            y = np.array(s[i-N+1:i+1])
            reg = np.polyfit(x,y,1)
            slp[i] = reg[0]
        df['slope'] = slp
        return df
    
    def slope_qk(s,N):
        df = DataFrame()
        df['price'] = s
        df['sxy'] = 0
        x = np.arange(1,N+1)
        df['sx'] = x.sum()
        df['sxx'] = (x**2).sum()
        df['sy'] = df.price.rolling(N).sum()
        for i in range(N):
            df['sxy'] = df['sxy'] + df.price.shift(i)*(N-i)
        df['slope'] = (N*df.sxy-df.sx*df.sy) / (N*df.sxx-df.sx**2)
        return df
    
    def dualSlope(s,N1,N2):
        df = DataFrame()
        df['price'] = s
        df['slope1'] = ti.slope_qk(s,N1).slope
        df['slope2'] = ti.slope_qk(s,N2).slope
        return df
    
    def linreg(s,N):
        df = DataFrame()
        df['price'] = s
        for i in range(N-1,len(s)):
            y = np.array(s[i-N+1:i+1])
            y = y/y[0] - 1   #cumulative return
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            df.at[i,'slope'] = slope
            df.at[i,'p_value'] = p_value
        return df 
    
    def convex(s,N):
        df = DataFrame()
        df['price'] = s
        slp = Series(data=np.nan,index=s.index)
        cvx = Series(data=np.nan,index=s.index)
        x = np.array(range(1,N+1))
        for i in range(N-1,len(df)):
            y = np.array(s[i-N+1:i+1])
            reg = np.polyfit(x,y,2)
            slp[i] = reg[0]
            cvx[i] = reg[1]
        df['slope'] = slp
        df['convex'] = cvx
        return df

    def mom(s,N):
        df = DataFrame()
        df['price'] = s
        df['mom'] = df.price.pct_change(N)
        return df

    def rsi(s,N):
        df = DataFrame()
        df['price'] = s
        change = df.price.diff()
        df['PC'] = change.apply(lambda x:x if x>0 else 0)
        df['NC'] = change.apply(lambda x:-x if x<0 else 0)
        s_N = Series(data=np.nan,index=s.index)
        s_P = Series(data=np.nan,index=s.index)
        s_P[N] = df.PC[1:N+1].mean()
        s_N[N] = df.NC[1:N+1].mean()
        for i in range(N+1,len(s)):
            s_P[i] = (s_P[i-1] * (N-1) + df.PC[i])/N
            s_N[i] = (s_N[i-1] * (N-1) + df.NC[i])/N
        df['P'] = s_P
        df['N'] = s_N
        df['RS'] = df.P/df.N
        df['RSI'] = 100 - 100/(1+df.RS)
        return df
    
    def srsi(s,N):
        df = DataFrame()
        df['price'] = s
        change = df.price.diff()
        df['PC'] = change.apply(lambda x:x if x>0 else 0)
        df['NC'] = change.apply(lambda x:-x if x<0 else 0)
        df['P'] = df.PC.rolling(N+1).mean()
        df['N'] = df.NC.rolling(N+1).mean()
        df['RS'] = df.P/df.N
        df['RSI'] = 100 - 100/(1+df.RS)
        return df

    def env(s,N,offset):
        df = DataFrame()
        df['price'] = s
        df['ma'] = df.price.rolling(N).mean()
        df['upper'] = df.ma * (1 + offset)
        df['lower'] = df.ma * (1 - offset)
        return df

    def dma(s,N1,N2):
        if N1 >= N2:
            print('WARNING: Dual MA N1 >= N2')
        df = DataFrame()
        df['price'] = s
        df['ma1'] = df.price.rolling(N1).mean()
        df['ma2'] = df.price.rolling(N2).mean()
        return df

    def macd(s,K1=12,K2=26,N=9):
        df = DataFrame()
        df['price'] = s
        df['ma1'] = df.price.rolling(K1).mean()
        df['ma2'] = df.price.rolling(K2).mean()
        df['dif'] = df.ma1 - df.ma2
        df['dea'] = df.dif.rolling(N).mean()
        df['macd'] = 2 * (df.dif - df.dea)
        return df

class xti():
    def don(holc,N):
        df = holc.copy()
        df['upper'] = holc.high.rolling(N).max().shift()
        df['lower'] = holc.low.rolling(N).min().shift()
        df['mid'] = (df.upper + df.lower)/2
        return df
    
    def bollShft(holc,N,offset):
        df = holc.copy()
        df['std'] = df.close.rolling(N).std()
        df['mid'] = df.close.rolling(N).mean()
        df['upper'] = (df['mid'] + offset * df['std']).shift()
        df['lower'] = (df['mid'] - offset * df['std']).shift()
        return df
    
    def boll(holc,N,offset,shift=0):
        df = holc.copy()
        df['std'] = df.close.rolling(N).std()
        df['mid'] = df.close.rolling(N).mean()
        df['upper'] = (df['mid'] + offset * df['std']).shift(shift)
        df['lower'] = (df['mid'] - offset * df['std']).shift(shift)
        return df

        
def map_sin(x,th):
    if x>=th or x<=0:
        return 0
    else:
        return np.sin(np.pi*x/th)

def tmode_filter(position,tmode):
    position_f = position.copy()
    if tmode == 1:
        position_f[position_f < 0] = 0
    elif tmode == -1:
        position_f[position_f > 0] = 0
    return position_f


class sl():
    def normal(s,pos,th):
        df = DataFrame({'price':s,'position':pos})
        for i in df.index[1:]:
            pass

def stoploss_trailing(price,position,th,return_ext=False):
    #th为百分比
    df = DataFrame(index=price.index)
    df['price'] = price
    df['position'] = position
    low_ae = Series(data=0.0,index=price.index)
    high_ae = Series(data=0.0,index=price.index)
    stop_price = Series(data=0.0,index=price.index)
    position_sl = position.copy()
    stopped = True
    for i in range(1,len(price)):
        if position[i] != 0:
            if position[i-1] == 0:
#                print('enter')
                low_ae[i] = price[i]
#                print(low_ae[i])
                high_ae[i] = price[i]
                stopped = False
            else:          
                low_ae[i] = min(price[i],low_ae[i-1])
                high_ae[i] = max(price[i],high_ae[i-1])
            if position[i] > 0:
                stop_price[i] = high_ae[i] * (1-th)
                if price[i] < stop_price[i]:
                    stopped = True
            else:
                stop_price[i] = low_ae[i] * (1+th)
                if price[i] > stop_price[i]:
                    stopped = True
            if stopped:
                position_sl[i] = 0
            
    df['low_ae'] = low_ae
    df['high_ae'] = high_ae
    df['stop_price'] = stop_price
    df['position_sl'] = position_sl
    if return_ext:
        return position_sl,df
    else:
        return position_sl

def stoploss_kextreme(holc,position,return_ext=False):
    df = holc.copy()
    df['position'] = position
    price = df.close
    stop_price = Series(data=0.0,index=price.index)
    position_sl = position.copy()
    stopped = True
    for i in range(1,len(price)):
        if position[i] != 0:
            if position[i-1] == 0:
                if position[i] > 0:
                    stop_price[i] = holc.at[i,'low']
                else:
                    stop_price[i] = holc.at[i,'high']
                stopped = False
            else:          
                stop_price[i] = stop_price[i-1]
                
            if position[i] > 0:
                if price[i] < stop_price[i]:
                    stopped = True
            else:
                if price[i] > stop_price[i]:
                    stopped = True
            if stopped:
                position_sl[i] = 0
            
    df['stop_price'] = stop_price
    df['position_sl'] = position_sl
    if return_ext:
        return position_sl,df
    else:
        return position_sl        
            
def PG_1line(price:Series,line:Series) -> Series:
    position = Series(index=price.index,data=0)
    position[price>line] = 1
    position[price<line] = -1
    return position

def PG_2line(price:Series,upper:Series,lower:Series,reversal=False) -> Series:
    position = Series(index=price.index,data=0)
    
    if type(upper) in [int,float]:
        upper = Series(data=upper,index=price.index)

    if type(lower) in [int,float]:
        lower = Series(data=lower,index=price.index)

    for i in range(1,len(price)):
        if np.isnan(upper[i]):
            continue
        if position[i-1] == 0:
            if reversal:
                if price[i] < upper[i] and price[i-1] > upper[i-1] and price[i] > lower[i]:
                    position[i] = -1
                elif price[i] > lower[i] and price[i-1] < lower[i-1] and price[i] < upper[i]:
                    position[i] = 1
                else:
                    position[i] = position[i-1]
            else:
                if price[i] > upper[i] and price[i-1] < upper[i-1]:
                    position[i] = 1
                elif price[i] < lower[i] and price[i-1] > lower[i-1]:
                    position[i] = -1
                else:
                    position[i] = position[i-1]
        elif position[i-1] == 1:
            if reversal:
                if price[i] > upper[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if price[i] < lower[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
        elif position[i-1] == -1:
            if reversal:
                if price[i] < lower[i]:
                    position[i] = 1
                else:
                    position[i] = position[i-1]
            else:
                if price[i] > upper[i]:
                    position[i] = 1
                else:
                    position[i] = position[i-1]
    return position

#def PG_2line_X(price:Series,upper:Series,lower:Series,shortable=False,reversal=False) -> Series:
#    position = Series(index=price.index,data=0)
#           
#    for i in range(2,len(price)):
#        if np.isnan(upper[i]):
#            continue
#        if position[i-1] == 0:
#            if reversal:
#                if price[i-1] > upper[i-1] and price[1] <= upper[i] and shortable:
#                    position[i] = -1
#                elif price[i-1] < lower[i-1] and price[i] >= lower[i]:
#                    position[i] = 1
#                else:
#                    position[i] = position[i-1]
#            else:
#                if price[i] >= upper[i] and price[i-1] < upper[i-1]:
#                    position[i] = 1
#                elif price[i] <= lower[i] and price[i-1] > lower[i-1] and shortable:
#                    position[i] = -1
#                else:
#                    position[i] = position[i-1]
#        elif position[i-1] == 1:
#            if reversal:
#                if price[i] > upper[i]:
#                    position[i] = -1 if shortable else 0
#                else:
#                    position[i] = position[i-1]
#            else:
#                if price[i] < lower[i]:
#                    position[i] = -1 if shortable else 0
#                else:
#                    position[i] = position[i-1]
#        elif position[i-1] == -1:
#            if reversal:
#                if price[i] < lower[i]:
#                    position[i] = 1
#                else:
#                    position[i] = position[i-1]
#            else:
#                if price[i] > upper[i]:
#                    position[i] = 1
#                else:
#                    position[i] = position[i-1]
#    return position

def PG_3line(price:Series,mid:Series,upper:Series,lower:Series,reversal=False) -> Series:
    position = Series(index=price.index,data=0)
    if type(mid) in [int,float]:
        mid = Series(data=mid,index=price.index)
        
    if type(upper) in [int,float]:
        upper = Series(data=upper,index=price.index)
    
    if type(lower) in [int,float]:
        lower = Series(data=lower,index=price.index)
    
    for i in range(1,len(price)):
        if np.isnan(mid[i]):
            continue
        if position[i-1] == 0:
            if reversal:
                if price[i-1] > upper[i-1] and price[i] < upper[i] and price[i] > mid[i]: #防止直接穿越中线
                    position[i] = -1
                elif price[i-1] < lower[i-1] and price[i] > lower[i] and price[i] < mid[i]:
                    position[i] = 1
                else:
                    position[i] = position[i-1]
            else:
                if price[i] > upper[i]:# and price[i-1] < upper[i-1]:
                    position[i] = 1
                elif price[i] < lower[i]:# and price[i-1] > lower[i-1]:
                    position[i] = -1
                else:
                    position[i] = position[i-1]
        elif position[i-1] == 1:
            if reversal:
                if price[i] > mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if price[i] < mid[i]:
                    position[i] = 0 if price[i] >= lower[i] else -1
#                    position[i] = -1 if price[i] < lower[i] else 0

                else:
                    position[i] = position[i-1]
                
        elif position[i-1] == -1:
            if reversal:
                if price[i] < mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if price[i] > mid[i]:
                    position[i] = 0 if price[i] <= upper[i] else 1
                else:
                    position[i] = position[i-1]
    return position


def PG_3line_X(price:Series,mid:Series,upper:Series,lower:Series,shortable=False,reversal=False) -> Series:
    position = Series(index=price.index,data=0)
           
    for i in range(2,len(price)):
        if np.isnan(mid[i]):
            continue
        if position[i-1] == 0:
            if reversal:
                if price[i] <= upper[i] and price[i-1] > upper[i-1] and price[i] > mid[i] and shortable:
                    position[i] = -1
                elif price[i] >= lower[i] and price[i-1] < lower[i-1] and price[i] < mid[i]:
                    position[i] = 1
                else:
                    position[i] = position[i-1]
            else:
                if price[i] >= upper[i] and price[i-1] < upper[i-1] and price[i] < mid[i]:
                    position[i] = 1
                elif price[i] <= lower[i] and price[i-1] > lower[i-1] and price[i] > mid[i] and shortable:
                    position[i] = -1
                else:
                    position[i] = position[i-1]
        elif position[i-1] == 1:
            if reversal:
                if price[i] > mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if price[i] < mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
        elif position[i-1] == -1:
            if reversal:
                if price[i] < mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if price[i] > mid[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
    return position


def ds(position:Series,timer:int) -> Series:
    #定时退出
    barsSinceEntry = Series(index=position.index,data=0)
    position_ds = position.copy()
    for i in range(1,len(position)):
        if position[i] * position[i-1] > 0:
            barsSinceEntry[i] = barsSinceEntry[i-1] + 1
    position_ds[barsSinceEntry>=timer] = 0                     
    return position_ds
            
def srsi(s,N=5,th=30,shortable=False,fee=0):
    df = DataFrame()
    df['price'] = s
    df['mid'] = 50
    df['oversold'] = th
    df['overbought'] = 100-th
    df['chg'] = df.price.diff()
    df['loss'] = df.chg.apply(lambda x:-x if x<0 else 0)
    df['gain'] = df.chg.apply(lambda x:x if x>0 else 0)
    df['count_loss'] = (df.loss>0).rolling(N).sum()
    df['count_gain'] = (df.gain>0).rolling(N).sum()
    df['avg_gain'] = df.gain.rolling(N).sum()/df.count_gain
    df['avg_loss'] = df.loss.rolling(N).sum()/df.count_loss
    df['rs'] = df.avg_gain / df.avg_loss
    rsi = 100 - 100/(df.rs + 1)
    rsi[isnull(df.avg_loss)] = 100
    rsi[isnull(df.avg_gain)] = 0
    rsi[:N-1] = np.nan
    df['rsi'] = rsi
    df['position'] = PG_3line_X(df.rsi,df.mid,df.overbought,df.oversold,shortable=shortable,reversal=True)
    df_nav = qtool.calc_nav(df.price,df.position,fee)
    df_nav['rsi'] = df['rsi']
    df_nav['rs'] = df['rs']
    df_nav['avg_gain'] = df['avg_gain']
    df_nav['avg_loss'] = df['avg_loss']
    df_nav['gain'] = df['gain']
    df_nav['loss'] = df['loss']
    
    return df_nav        

#def don(s,N,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['ma'] = df.price.rolling(N).mean()
#    df['upper_band'] = df.price.rolling(N).max()
#    df['lower_band'] = df.price.rolling(N).min()
#    df['position'] = PG_3line(df.price,df.ma,df.upper_band,df.lower_band,shortable=shortable)
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma'] = df['ma']
#    df_nav['upper_band'] = df['upper_band']
#    df_nav['lower_band'] = df['lower_band']
#    return df_nav
#
#def don_leading(s,N,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['ma'] = df.price.rolling(N).mean().shift()
#    df['upper_band'] = df.price.rolling(N).max().shift()
#    df['lower_band'] = df.price.rolling(N).min().shift()
#    df['position'] = PG_3line(df.price,df.ma,df.upper_band,df.lower_band,shortable=shortable)
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma'] = df['ma']
#    df_nav['upper_band'] = df['upper_band']
#    df_nav['lower_band'] = df['lower_band']
#    return df_nav
#
#def donmid(s,N,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['upper_band'] = df.price.rolling(N).max()
#    df['lower_band'] = df.price.rolling(N).min()
#    df['mid'] = (df.upper_band + df.lower_band)/2
#    
#    df['position'] = 0
#    df.loc[df.price > df.mid,'position'] = 1
#    if shortable:
#        df.loc[df.price < df.mid, 'position'] = -1
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['mid'] = df['mid']
#    return df_nav
#
#
#
#def don2(s,N,shortable=False,fee=0):
#    #不使用中轨
#    df = DataFrame()
#    df['price'] = s
#    df['upper_band'] = df.price.rolling(N).max()
#    df['lower_band'] = df.price.rolling(N).min()
#    
#    df['position'] = 0
#    df.loc[df.price>=df.upper_band] = 1
#    if shortable:
#        df.loc[df.price<=df.lower_band] = -1
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['upper_band'] = df['upper_band']
#    df_nav['lower_band'] = df['lower_band']
#    return df_nav
#
#def don_ds(s,N,timer,shortable=False,fee=0):
#    #定时退出
#    df = DataFrame()
#    df['price'] = s
#    df['ma'] = df.price.rolling(N).mean()
#    df['upper_band'] = df.price.rolling(N).max()
#    df['lower_band'] = df.price.rolling(N).min()
#
#    df['position_original'] = PG_3line(df.price,df.ma,df.upper_band,df.lower_band,shortable=shortable)
#    df['position_ds'] = ds(df.position_original,timer)
#    
#    df_nav = qtool.calc_nav(df.price,df.position_ds,fee)
#    df_nav['ma'] = df['ma']
#    df_nav['upper_band'] = df['upper_band']
#    df_nav['lower_band'] = df['lower_band']
#    df_nav['position_original'] = df['position_original']
#    df_nav['position_ds'] = df['position_ds']
#    
#    return df_nav
#
#
#def macd(s,N1,N2,M,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['ma1'] = df.price.rolling(N1).mean()
#    df['ma2'] = df.price.rolling(N2).mean()
#    df['MACDValue'] = df.ma1 - df.ma2
#    df['AvgMACD'] = df.MACDValue.rolling(M).mean()
#    df['MACDDiff'] = df.MACDValue - df.AvgMACD
#    df['position'] = 0
#    df.loc[df.MACDDiff>0,'position'] = 1
#    if shortable:
#        df.loc[df.MACDDiff<0,'position'] = -1  
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma1'] = df['ma1']
#    df_nav['ma2'] = df['ma2']
#    df_nav['MACDValue'] = df['MACDValue']
#    df_nav['AvgMACD'] = df['AvgMACD']
#    df_nav['MACDDiff'] = df['MACDDiff']
#    return df_nav
#
#def macd_ds(s,N1,N2,M,timer,shortable=False,fee=0):
#    df = macd(s,N1,N2,M,shortable=shortable,fee=fee)
#    df['position_original'] = df['position']
#    df['position'] = ds(df.position_original,timer)
#    df_nav = qtool.calc_nav(df.price,df.position,fee=fee)
#    df_nav['position_original'] = df['position_original']
#    df_nav['ma1'] = df['ma1']
#    df_nav['ma2'] = df['ma2']
#    df_nav['MACDValue'] = df['MACDValue']
#    df_nav['AvgMACD'] = df['AvgMACD']
#    df_nav['MACDDiff'] = df['MACDDiff']
#
#    return df_nav
#
#
#
#def trend(s,N,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['lb'] = df.price.pct_change(N)
#    df['return'] = df.price.pct_change()
#    df['position'] = 0
#    df.loc[df.lb>0,'position'] = 1 
#    if shortable:
#        df.loc[df.lb<0,'position'] = -1
#    else:
#        df.loc[df.lb<0,'position'] = 0
#    
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['lb'] = df['lb']
#    return df_nav
#    
#
#def ma(s,N,shortable=False,fee=0):
##s：资产价格序列
##N: 均线参数
##返回： dataframe，包含 price, position, nav（简单法）
#    df = DataFrame()
#    df['price'] = s
#    df['ma'] = df.price.rolling(N).mean()
#    df['return'] = df.price.pct_change()
#    df['position'] = 0
#    df.loc[df.price>df.ma,'position'] = 1 
#    if shortable:
#        df.loc[df.price<df.ma,'position'] = -1  
#    else:
#        df.loc[df.price<df.ma,'position'] = 0 
#              
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma'] = df['ma']
#    df_nav['x'] = df_nav.price/df_nav.ma-1
#    return df_nav
#
#def ma_leading(s,N,shortable=False,fee=0):
##s：资产价格序列
##N: 均线参数
##返回： dataframe，包含 price, position, nav（简单法）
#    df = DataFrame()
#    df['price'] = s
#    df['ma'] = df.price.rolling(N).mean().shift()
#    df['return'] = df.price.pct_change()
#    df['position'] = 0
#    df.loc[df.price>df.ma,'position'] = 1 
#    if shortable:
#        df.loc[df.price<df.ma,'position'] = -1  
#    else:
#        df.loc[df.price<df.ma,'position'] = 0 
#              
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma'] = df['ma']
#    return df_nav
#
#    
#def dma(s,N1,N2,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['ma1'] = df.price.rolling(N1).mean()
#    df['ma2'] = df.price.rolling(N2).mean()
#    df['position'] = 0
#    df.loc[df.ma1>df.ma2,'position'] = 1 
#          
#    df['return'] = df.price.pct_change()
#    
#    if shortable:
#        df.loc[df.ma1<df.ma2,'position'] = -1  
#    
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma1'] = df['ma1']
#    df_nav['ma2'] = df['ma2']
#    return df_nav
#
#def altdma(s,N1,N2,shortable=False,fee=0):
#    df = DataFrame()
#    df['price'] = s
#    df['ma1'] = df.price.rolling(N1).mean()
#    df['ma2'] = df.price.rolling(N2).mean()
#    df['position'] = 0
#    df.loc[(df.price>df.ma1) & (df.price>df.ma2),'position'] = 1 
#    if shortable:
#        df.loc[(df.price<df.ma1) & (df.price<df.ma2),'position'] = -1
#    df['return'] = df.price.pct_change()
#    df_nav = qtool.calc_nav(df.price,df.position,fee)
#    df_nav['ma1'] = df['ma1']
#    df_nav['ma2'] = df['ma2']
#    return df_nav



def dualMom(data,N,fee=0):
    rst = data.copy()
    a = data.columns
    a1 = a[0]
    a2 = a[1]
    LB1 = a1 + '_LB'
    LB2 = a2 + '_LB'
    R1 = a1 + '_R'
    R2 = a2 + '_R'
    rst[LB1] = rst[a1].pct_change(N)
    rst[LB2] = rst[a2].pct_change(N)
    rst[R1] = rst[a1].pct_change()
    rst[R2] = rst[a2].pct_change()
    rst.reset_index(inplace=True)
    rst['nav1'] = rst[a1]/rst.at[0,a1]
    rst['nav2'] = rst[a2]/rst.at[0,a2]
    rst.at[0,'pos'] = '-'
    rst.at[0,'returnGross'] = 0.0
    rst.at[0,'fee'] = 0.0
    rst.at[0,'returnNet'] = 0.0
    rst.at[0,'nav'] = 1.0
    for i in rst.index[1:]:
        rLB1 = -1e6 if isnull(rst.at[i,LB1]) else rst.at[i,LB1]
        rLB2 = -1e6 if isnull(rst.at[i,LB2]) else rst.at[i,LB2]
        rst.at[i,'pos'] = '-' if max(rLB1,rLB2)<=0 else a1 if rLB1 > rLB2 else a2
        rst.at[i,'returnGross'] = 0 if rst.at[i-1,'pos'] == '-' else rst.at[i,R1] if rst.at[i-1,'pos'] == a1 else rst.at[i,R2]
        rst.at[i,'fee'] = 0 if rst.at[i-1,'pos'] == rst.at[i,'pos'] else fee if rst.at[i,'pos']=='-' or rst.at[i-1,'pos']=='-' else fee*2
        rst.at[i,'returnNet'] = rst.at[i,'returnGross'] - rst.at[i,'fee']
        rst.at[i,'nav'] = rst.at[i-1,'nav'] * (1+rst.at[i,'returnNet'])
    rst['nav1'] = rst[a1]/rst.at[0,a1]
    rst['nav2'] = rst[a2]/rst.at[0,a2]
    rst.set_index('index',inplace=True)
    return rst
    
    
def turn(data,N,timing=True,fee=0,timing_shrd=1):
#测试多资产轮动策略
#选择 N期 涨幅最高的资产进行持有
#data 为含有资产价格的 DataFrame, 每一列为一个资产
#N 为回看的期数，用来计算涨幅
#timing 指示是否进行择时，如果不择时，则始终满仓轮动；
#  如果择时，则需要计算下跌资产的个数，如果下跌资产超过 timing_shrd 比例，则空仓
    df=data.copy()
    cols = df.columns
    noa = len(cols)
    for col in cols:
        df[col+'_lb'] = df[col].pct_change(N)
        df[col+'_return'] = df[col].pct_change()
    lbs = [col+'_lb' for col in cols]
    df.reset_index(inplace=True)
    for i in df.index: #i=0,1,2,3,...
        s = df.ix[i][lbs]
        if all(isnull(s)):
            position = 'null'
        else:
            s.sort_values(ascending=False,inplace=True)
            nod = (s<0).sum()  #number of decline 下跌的资产个数
            if timing and nod >= timing_shrd * noa: 
                position = 'null'
            else:
                position = s.index[0].split('_')[0]
        df.at[i,'position'] = position
        if i == 0:
            strategy_return_before_fee = 0
            
            strategy_return = 0
            nav_before_fee = 1
            nav = 1
        else:
            position_1 = df.at[i-1,'position']
            strategy_return_before_fee = 0 if position_1 == 'null' else df.at[i,position_1+'_return']
            trading_fee = 0 if position_1 == position else fee
            strategy_return = strategy_return_before_fee - trading_fee
            nav_before_fee = df.at[i-1,'nav_before_fee'] * (1+strategy_return_before_fee)
            nav = df.at[i-1,'nav'] * (1+strategy_return)
        df.at[i,'strategy_return_before_fee'] = strategy_return_before_fee
        df.at[i,'strategy_return'] = strategy_return
        df.at[i,'nav_before_fee'] = nav_before_fee
        df.at[i,'nav'] = nav
    df.set_index('date',inplace=True)
    return df

def switch_ratio(data,N):
    df = data.copy()
    if len(df.columns) != 2:
        return 0
    s = df.mean()
    if s[0] > s[1]:
        L = s.index[0]
        S = s.index[1]
    else:
        L = s.index[1]
        S = s.index[0]
    df[L+'_return'] = df[L].pct_change()
    df[S+'_return'] = df[S].pct_change()
    df['ratio'] = np.log(df[L]/df[S])
    df['ratio_ma'] = rolling_mean(df['ratio'],N)
    df.reset_index(inplace=1)
    for i in df.index:
        ratio_ma = df.get_value(i,'ratio_ma')
        ratio = df.get_value(i,'ratio')
        if isnull(ratio_ma):
            position = 'null'
        else:
            if ratio >= ratio_ma:
                position = L
            else:
                position = S
        df.set_value(i,'position',position)
        if i == 0:
            nav = 1
            strategy_return = 0
        else:
            position_1 = df.get_value(i-1,'position')
            if position_1 == 'null':
                strategy_return = 0
            else:
                strategy_return = df.get_value(i,position_1+'_return')
            nav = df.get_value(i-1,'nav') * (1+strategy_return)
        df.set_value(i,'strategy_return',strategy_return)
        df.set_value(i,'nav',nav)
    df.set_index('date',inplace=1) 
    return df


class pg():
    def ma(s,N,tmode=1):
        df = ti.ma(s,N)
        position = PG_1line(s,df.ma)
        position = tmode_filter(position,tmode)
        return position
    
    def maAlign(s,maLength,tmode=1):
        df = ti.maAlign(s,maLength)
        df_madiff = df.iloc[:,-len(maLength)+1:]
        position = Series(0,index=df.index)
        position[(df_madiff>0).all(1)] = 1
        position[(df_madiff<0).all(1)] = -1
        return tmode_filter(position,tmode)
    
    def ma_with_upxfilter(s,N,M,tmode=1):
        df = ti.ma(s,N)
        df['count'] = ti.cntmaupx(s,M)['count']
        df['position'] = 0
        df.at[(df['count']>0) & (df.price > df.ma),'position']=1
        df.at[(df['count']<0) & (df.price < df.ma),'position']=-1
        position = tmode_filter(df.position,tmode)
        return position
    
    def ma_leading(s,N,tmode=1):
        df = ti.ma_leading(s,N)
        position = PG_1line(s,df.ma)
        position = tmode_filter(position,tmode)
        return position
    
    def linreg(s,N,tmode=1):
        df = ti.linreg(s,N)
        df['position'] = 0
        df.at[df.slope > 0, 'position'] = 1
        df.at[df.slope < 0, 'position'] = -1
        df.at[df.p_value > 0.001, 'position'] = 0
        return df.position
    
    def boll(s,N,offset,tmode=1):
        df = ti.boll(s,N,offset)
        position = PG_3line(s,df.mid,df.upper,df.lower)
        position = tmode_filter(position,tmode)
        return position
    
    def boll_sl(s,N,offset,stoploss=0.01,tmode=1):
        df = ti.boll(s,N,offset)
        df['position'] = 0
        pos = 0
        for i in df.index[1:]:
            price_last = df.at[i-1,'price']
            price = df.at[i,'price']
            upper = df.at[i,'upper']
            lower = df.at[i,'lower']
            mid = df.at[i,'mid']
            if pos == 0:
                if price_last < upper and price > upper:
                    pos = 1
                    price_stoploss = price * (1-stoploss)
                elif price_last > lower and price < lower:
                    pos = -1
                    price_stoploss = price * (1+stoploss)
            elif pos == 1:
                if price < price_stoploss or price < mid:
                    pos = 0
            elif pos == -1:
                if price > price_stoploss or price > mid:
                    pos = 0
            df.at[i,'position'] = pos
        return tmode_filter(df.position,tmode)
    
    def boll_leading(s,N,offset,tmode=1):
        df = ti.boll_leading(s,N,offset)
        position = PG_3line(s,df.mid,df.upper,df.lower)
        position = tmode_filter(position,tmode)
        return position

    def don(s,N,tmode=1):    
        df = ti.don(s,N)
        position = PG_3line(s,df.mid,df.upper,df.lower)
        position = tmode_filter(position,tmode)
        return position

    def brk(s,N,tmode=1):
        df = ti.brk(s,N)
        position = PG_1line(s,df.level)
        position = tmode_filter(position,tmode)
        return position
    
    def boll_upper(s,N,offset,tmode=1):
        df = ti.boll_leading(s,N,offset)
        position = PG_1line(s,df.upper)
        position = tmode_filter(position,tmode)
        return position
    
    def slope(s,N,tmode=1):
        df = ti.slope(s,N)
        position = PG_1line(df.slope,0)
        position = tmode_filter(position,tmode)
        return position
    
    def mom(s,N,tmode=1):
        df = ti.mom(s,N)
        position = PG_1line(df.mom,0)
        position = tmode_filter(position,tmode)
        return position

    def env(s,N,offset,tmode=1):
        df = ti.env(s,N,offset)
        position = PG_3line(s,df.ma,df.upper,df.lower)
        position = tmode_filter(position,tmode)
        return position

    def dma(s,N1,N2,tmode=1):
        df = ti.dma(s,N1,N2)
        position = PG_1line(df.ma1,df.ma2)
        position = tmode_filter(position,tmode)
        return position

    def dualSlope(s,N1,N2,tmode=1):
        df = ti.dualSlope(s,N1,N2)
        position = PG_1line(df.slope1,df.slope2)
        position = tmode_filter(position,tmode)
        return position

    def dualSlope2(s,N1,N2,tmode=1):
        df = ti.dualSlope(s,N1,N2)
        df['position'] = 0
        pos = 0
        for i in df.index:
            slope1 = df.at[i,'slope1']
            slope2 = df.at[i,'slope2']
            if pos == 0:
                if slope1 > slope2 and slope2 > 0:
                    pos = 1
                elif slope1 < slope2 and slope2 < 0:
                    pos = -1
            elif pos == 1:
                if slope2 < 0:
                    pos = 0
            elif pos == -1:
                if slope2 > 0:
                    pos = 0
            df.at[i,'position'] = pos
        return tmode_filter(df.position,tmode)


    def macd(s,K1=12,K2=26,N=9,tmode=1):
        df = ti.macd(s,K1,K2,N)
        df['position'] = 0
        pos = 0
        for i in df.index:
            dif = df.at[i,'dif']
            macd = df.at[i,'macd']
            if pos == 0:
                if dif > 0 and macd > 0:
                    pos = 1
                elif dif < 0 and macd < 0:
                    pos = -1
            elif pos == 1:
                if dif < 0:
                    pos = 0
            elif pos == -1:
                if dif > 0:
                    pos = 0
            df.at[i,'position'] = pos
        return tmode_filter(df.position,tmode)

    def convex(s,N,tmode=1):
        df = ti.convex(s,N)
        df['position'] = 0
        pos= 0
        for i in df.index:
            slope = df.at[i,'slope']
            convex = df.at[i,'convex']
            if pos == 0:
                if slope > 0 and convex > 0:
                    pos = 1
                elif slope < 0 and convex < 0:
                    pos = -1
            elif pos == 1:
                if slope < 0:
                    pos = 0
            elif pos == -1:
                if slope > 0:
                    pos = 0
            df.at[i,'position'] = pos
        return tmode_filter(df.position,tmode)
    
    def xdon(df,N,tmode=1):
        xdf = xti.don(df,N)
        position = PG_3line(xdf.close,xdf.mid,xdf.upper,xdf.lower)
        position = tmode_filter(position,tmode)
        return position

    

    def rsrs(df,N,S1,S2,retAll=False):
        xdf = df.copy()
        sumx = xdf.low.rolling(N).sum()
        sumy = xdf.high.rolling(N).sum()
        sumxx = (xdf.low**2).rolling(N).sum()
        sumxy = (xdf.low*xdf.high).rolling(N).sum()
        xdf['slope']=(N*sumxy-sumx*sumy)/(N*sumxx-sumx**2)
        xdf['slope']=xdf.slope.shift()
        position = PG_2line(xdf.slope,S1,S2)
        position = tmode_filter(position,1)
        xdf['position'] = position
        if retAll:
            return xdf
        else:
            return position

    def poly(s,N,d,tmode = 1, retAll = False):
        n = len(s)
        rst = TA.poly(s.values,N,d)
        deriv1 = np.zeros(n)
        deriv2 = np.zeros(n)
        for i in range(n):
            if i < N - 1:
                deriv1[i] = np.nan
                deriv2[i] = np.nan
            else:
                p = np.poly1d(rst[i])
                deriv1[i] = p.deriv(1)(s.values[i])
                deriv2[i] = p.deriv(2)(s.values[i])
        position = np.zeros(n)
        position[(deriv1 > 0) & (deriv2 > 0)] = 1
        position[(deriv1 < 0) & (deriv2 < 0)] = -1
        position = Series(position,index=s.index)
        position = tmode_filter(position,tmode)
        df = DataFrame({'price':s,'deriv1':deriv1,'deriv2':deriv2,'position':position})
        if retAll:
            return df
        else:
            return df.position
        
class ol():
    #order list
    def boll_leading_sl(holc,N,offset):
        df = xti.bollShft(holc,N,offset)
        ordLst = DataFrame()
        pos = 0
        for i in df.index:
            traded = False
            przOpen = df.at[i,'open']
            przClose = df.at[i,'close']
            przHigh = df.at[i,'high']
            przLow = df.at[i,'low']
            mid = df.at[i,'mid']
            upper = df.at[i,'upper']
            lower = df.at[i,'lower']
            date = df.at[i,'date']
            
            if pos == 1:
                if przOpen < mid:
                    pos = 0
                    prz = przOpen
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
            elif pos == -1:
                if przOpen > mid:
                    pos = 0
                    prz = przOpen
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
            if pos == 1:
                if przLow < mid:
                    pos = 0
                    prz = mid
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
            elif pos == -1:
                if przHigh > mid:
                    pos = 0
                    prz = mid
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
            if pos == 0:
                if przClose > upper:
                    pos = 1
                    prz = przClose
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
                elif przClose < lower:
                    pos = -1
                    prz = przClose
                    ordLst = ordLst.append({'date':date,'prz':prz,'pos':pos},ignore_index=True)
                    traded = True
        
            if not traded and pos != 0:
                ordLst = ordLst.append({'date':date,'prz':przClose,'pos':pos},ignore_index=True)
                
        return ordLst

#def boll(s,N,offset,tmode=1,fee=0):
#    position = pg.boll(s,N,offset,tmode=tmode)
#    df_nav = qtool.calc_nav(s,position,fee)

def PG_dma(s,N1,N2,shortable=False):
    return dma(s,N1,N2,shortable=shortable,fee=0).position


def PG_altdma(s,N1,N2,shortable=False):
    return altdma(s,N1,N2,shortable=shortable,fee=0).position

def PG_rsi(s,N,th,shortable=False):
    df = rsi(s,N,th,shortable=shortable,fee=0)
    return df.position


def PG_don2(s,N,shortable=False):
    return don2(s,N,shortable,0).position


def PG_don_leading(s,N,shortable=False):
    return don_leading(s,N,shortable,0).position

def PG_donmid(s,N,shortable=False):
    return donmid(s,N,shortable,0).position

def mix_position(positions,th = 1) -> Series:
#'''
#positions can be a list of position or a DataFrame
#example:
#    s1 = Series([1,1,1,0,0,1])
#    s2 = Series([0,0,1,0,1,0])
#    mixed = mix_position([s1,s2])
#'''

#    if type(positions) == Series:
#        return positions
    if type(positions) == DataFrame:
        summ = positions.sum(1)
#        n = len(positions.columns)
        mixed = Series(index=positions.index,data=0)
    else:
        df = DataFrame(positions).T
        summ = df.sum(1)
#        n = len(positions)
        mixed = Series(index=df.index,data=0)
    mixed[summ >= th] = 1
    mixed[summ <= -th] = -1
    return mixed
    
#def strategy_single_param(strategy_name,s,N,shortable=False,fee=0):
#    DICT_FUNC = {'trend':trend,'ma':ma}
#    func = DICT_FUNC[strategy_name]
#    return func(s,N,shortable,fee)
#
#def strategy_dual_param(strategy_name,s,P1,P2,shortable=False,fee=0):
#    DICT_FUNC = {'boll':boll}
#    func = DICT_FUNC[strategy_name]
#    return func(s,P1,P2,shortable,fee)
#    
#def mega_strategy(s,strategy,shortable=False,fee=0):
#    DICT_FUNC = {'trend':trend,'ma':ma,'boll':boll,'dma':dma,'break':trend}
#    strategy_name = strategy[0]
#    strategy_param = strategy[1]
#    func = DICT_FUNC[strategy_name]
#    nop = len(strategy_param)
#    if nop == 1:
#        return func(s,strategy_param[0],shortable,fee)
#    elif nop == 2:
#        return func(s,strategy_param[0],strategy_param[1],shortable,fee)
#    else:
#        print('mega_strategy 目前仅支持两个参数的策略')
#        return 0



DICT_PG = {'trend':pg.brk,'break':pg.brk,'ma':pg.ma,\
           'boll':pg.boll,'don':pg.don,'altdma':PG_altdma,'rsi':PG_rsi,\
           'don2':PG_don2,'don_leading':PG_don_leading,'donmid':PG_donmid,'boll_leading':pg.boll_leading,\
           'ma_leading':pg.ma_leading,'boll_upper':pg.boll_upper,'slope':pg.slope,'mom':pg.mom,'env':pg.env,\
           'ma_with_upxfilter':pg.ma_with_upxfilter,'dma':pg.dma,'xdon':pg.xdon,'macd':pg.macd,'boll_sl':pg.boll_sl,\
           'linreg':pg.linreg,'convex':pg.convex,'dualSlope':pg.dualSlope,'dualSlope2':pg.dualSlope2,'maAlign':pg.maAlign,\
           }


DICT_RS = {'fixed_interval':rs.fixed_interval,'fixed_vola':rs.fixed_vola}


def PG(s:Series,strategy:tuple,tmode=1) -> Series:
    strategy_name = strategy[0]
    strategy_param = strategy[1]
    if len(strategy) < 3:
        shs = 1
    else:
        shs = strategy[2]
    func = DICT_PG[strategy_name]
    nop = len(strategy_param)
    if nop == 1:
        return func(s,strategy_param[0],tmode=tmode) * shs
    elif nop == 2:
        return func(s,strategy_param[0],strategy_param[1],tmode=tmode) * shs
    elif nop == 3:
        return func(s,strategy_param[0],strategy_param[1],strategy_param[2],tmode=tmode) * shs
    else:
        print('mega_PG 目前仅支持最多三个参数的策略')
        return Series(index=s.index)

def PG_multi_strategies(s:Series,strategies:list,tmode=1) -> DataFrame:
    positions = []
    for strategy in strategies:
        print('PG_multi_strategies: %s' % str(strategy))
        p = PG(s,strategy,tmode=tmode)
        p.name = str(strategy)
        positions.append(p.copy())
    return DataFrame(positions).T

def mega_PG(s:Series,strategies,tmode=1):
    if type(strategies) == list:  #reture a DataFrame
        return PG_multi_strategies(s,strategies,tmode=tmode)
    else: #strategies is a single tuple, return a Series
        return PG(s,strategies,tmode=tmode)

def mega_PGRS(s:Series,rs_method,strategies,tmode=1):
    #with ReSample
    rs_name = rs_method[0]
    rs_param = rs_method[1]
    rs_func = DICT_RS[rs_name]
    s_rs = rs_func(s,rs_param[0])    
    positions = mega_PG(s_rs,strategies,tmode)
    positions = positions.reindex(s.index)
    if type(positions)==DataFrame:
        positions.iloc[0,:] = 0
    else:
        positions[0] = 0
    return positions.fillna(method='ffill')

def PG_mix(s:Series,strategies:list,tmode=1,th = -1) -> Series:
    if th == -1:
        th = len(strategies)
    positions = mega_PG(s,strategies,tmode)
    position = mix_position(positions,th)
    return position

def bc_PG(prices:DataFrame,strategy:tuple,tmode=1) -> DataFrame:
    positions = DataFrame(index=prices.index,columns=prices.columns)
    for col in positions.columns:
        positions[col] = mega_PG(prices[col],strategy,tmode=tmode)
    return positions
    
def mapping_strategy(price:Series,strategy:tuple,mapping:tuple) -> Series:
    DICT_MAP = {'map_sin':map_sin}
    x = mega_strategy(price,strategy)['x']
    map_name = mapping[0]
    map_params = mapping[1]
    map_func = DICT_MAP[map_name]
    if len(map_params) == 1:
        p = x.apply(lambda x: map_func(x,map_params[0]))
        p.name = 'position'
        return p
    elif len(map_params) == 2:
        p = x.apply(lambda x: map_func(x,map_params[0],map_params[1]))
        p.name = 'position'
        return p
    else:
        print('mapping_strategy 目前仅支持两个参数的 mapping function')
        return Series()

def calc_nav_single_position(price:Series,position:Series,fee=0,include_price=False):
    df_nav = qtool.calc_nav(price,position,fee=fee)
    if include_price:
        return DataFrame({'price':price/price[0],'nav':df_nav.nav})
    else:
        return df_nav.nav

def calc_nav_multi_positions(price:Series,positions:DataFrame,fee=0,include_price=False) -> DataFrame:
    navs = DataFrame()
    for col in positions:
        nav = calc_nav(price,positions[col],fee=fee)
        navs[col] = nav
    if include_price:
        navs['price'] = price/price[0]
    return navs


def calc_nav(price:Series,positions,fee=0,include_price=False):
    if type(positions) == Series:
        return calc_nav_single_position(price,positions,fee,include_price)
    else:
        return calc_nav_multi_positions(price,positions,fee,include_price)

def calc_nav_mat2mat(prices:DataFrame,positions:DataFrame,fee=0)->DataFrame:
    navs = DataFrame(index=prices.index,columns=prices.columns)
    for col in navs.columns:
        navs[col] = calc_nav_single_position(prices[col],positions[col],fee=fee)
    return navs

def periodize(s,st,M,tmode=1):
    df = DataFrame({'price':s})
    df['g'] = 1 + np.mod(range(len(s)),M)
    for i in range(1,M+1):
        s1 = df[df.g==i].price
        idx = s1.index
        p1 = PG(s1.reset_index().price,st,tmode=tmode)
        p1 = p1.shift().fillna(0)
        df[i] = Series(p1.values,index=idx)
        df = df.fillna(method='ffill').fillna(0)
    return df.iloc[:,2:].astype(int)


class mp():
    def cube(positions):
        n = len(positions.columns)
        x = positions.sum(1)/n
        return x**3

    def sine(positions):
        n = len(positions.columns)
        x = positions.sum(1)/n * np.pi
        return np.sin(x)
    
    def linear(positions):
        n = len(positions.columns)
        return positions.sum(1)/n

    def one(positions,th):
        w = mp.linear(positions)
        w_rst = Series(0,index=w.index)
        w_rst[w>th] = 1
        w_rst[w<th] = -1
        return w_rst

    def linearPT(positions,th):
        n = len(positions.columns)
        x = positions.sum(1)/n
        return x.apply(lambda x: 2*th-x if x>th else -2*th-x if x<-th else x)
    
    def linearPT2(positions,th):
        n = len(positions.columns)
        x = positions.sum(1)/n
        return x.apply(lambda x: 2*th + 1 - 2*x if x>th else -2*th - 1 - 2*x if x<-th else x/th)