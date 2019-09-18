from pandas import *
import numpy as np
import scipy.optimize as sco
from scipy import linalg

#data = read_excel('Data/data.xlsx','w',skiprows=1,index_col='date')        
#data=read_excel('收益测算.xlsx','精选指数',index_col='date')


def get_cum_return(data):
    data1 = data.copy()
    for i in data1.index:
        data1.ix[i]=data1.ix[i]/data.ix[0]-1
    return data1

def get_annualized_return_covar_and_freq(data):
    #返回各资产的 “年化收益率” 和 “协方差矩阵”
    freq = get_freq(data)
    if freq == 0:
        print('Alert in risk_budget: 无法识别的周期！')
        return 0
    returns = data.pct_change()
    covs = returns.cov() * freq
    return returns.mean() * freq, covs, freq

def get_annualized_return_covar(data):
    r,covar,_ = get_annualized_return_covar_and_freq(data)
    return r.values, covar.values


def dates_resample(dates,freq):
#对时间序列重采样
    if type(freq) == str:
        if freq == 'm':
            df = DataFrame(dates)
            df['month'] = df.date.apply(lambda x:datetime(x.year,x.month,1))
            rst = df.groupby('month').last()
            return rst.reset_index().date
        elif freq == 'q':
            df = DataFrame(dates)
            df['month'] = df.date.apply(lambda x:datetime(x.year,x.month,1))
            rst = df.groupby('month').last()
            rst['qend']=rst['date'].apply(lambda x:x.month==3 or x.month==6 or x.month==9 or x.month==12)
            return rst[rst.qend].reset_index().date
        elif freq == 'y':
            df = DataFrame(dates)
            df['year'] = df.date.apply(lambda x:x.year)
            rst = df.groupby('year').last()
            return rst.reset_index().date
        else:
            print('Alert in dates_resample:freq参数错误！')
            return Series()
    elif type(freq) == int:
        l = []
        i = 0
        for date in dates:
            i += 1
            if np.mod(i,freq) == 0:
                l.append(date)
        return Series(l)
    else:
        print('Alert in dates_resample:freq参数错误！')
        return Series()

def get_freq(data):
    t = data.index.values
    t = to_datetime(t)
    d = ((t[-1]-t[0])/len(t)).days
    if (abs(d)-1) < 2:
        return 220
    elif (abs(d) - 7) < 2:
        return 52
    elif (abs(d) - 12) < 2:
        return 12
    else:
        return 0

def dvola_parity(data):
    pass

def risk_budget(data,b,lev = 1):
#b: 风险权重
    if type(b)==list:
        b = np.array(b)
    
    freq = get_freq(data)
    if freq == 0:
        print('Alert in risk_budget: 无法识别的周期！')
        s = Series(0,index=data.columns)
        s.name = data.index[-1]
        return s
    
    noa = len(data.columns)
    returns = data.pct_change()
    covs = returns.cov() * freq    
              
    weights = b

    bnds = tuple((lev,lev) if weights[x]==lev else (0,0) if weights[x]==0 else (0,lev) for x in range(noa))
        
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-lev})

    def risk_budget_obj(covs,b,weights):
        TR = np.sqrt(np.dot(weights.T, np.dot(covs,weights)))
        return sum((np.dot(covs,weights) * weights / TR - b * TR) ** 2)

    opts = sco.minimize(lambda weights: risk_budget_obj(covs,b,weights), weights, method = 'SLSQP', bounds = bnds, constraints = cons,tol=1e-25)
    s = Series(opts.x,index=data.columns)
    s.name = data.index[-1]
    return s

def active_risk_budget(data,b,pi,c=2):
#b: 风险权重
    if type(b)==list:
        b = np.array(b)

    freq = get_freq(data)
    if freq == 0:
        print('Alert in risk_budget: 无法识别的周期！')
        s = Series(0,index=data.columns)
        s.name = data.index[-1]
        return s
    
    noa = len(data.columns)
    returns = data.pct_change()
    covs = returns.cov() * freq    
              
    weights = b

    bnds = tuple((1,1) if weights[x]==1 else (0,0) if weights[x]==0 else (0,1) for x in range(noa))
        
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})

    def active_risk_budget_obj(covs,b,weights):
        sigma = np.sqrt(np.dot(weights.T, np.dot(covs,weights)))
        TR =  sigma * c - np.dot(weights,pi)
        return sum( (np.multiply((c * np.dot(covs,weights)/ sigma  - pi),weights)  - b * TR) ** 2)

    opts = sco.minimize(lambda weights: active_risk_budget_obj(covs,b,weights), weights, method = 'SLSQP', bounds = bnds, constraints = cons,tol=1e-25)
    s = Series(opts.x,index=data.columns)
    s.name = data.index[-1]
    return s


def mean_var_optimization(r:'arr', covar:'arr2d',rrr:'num'):
    # both r and covar should be numpy arrays
    # rrr = required rate of return
    noa = len(r)
    w = np.zeros(noa)
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1},
            {'type':'eq', 'fun':lambda x: (x.reshape(-1,1)*r.reshape(-1,1)).sum()-rrr})
    bnds = tuple((0,1) for i in range(noa))
    def obj(covar, w):
        return np.dot(np.dot(w.T, covar), w)
    opts = sco.minimize(lambda w: obj(covar, w), w, method = 'SLSQP', bounds = bnds, constraints = cons, tol = 1e-25)
    return opts.x, opts.fun

def mean_var(data:'df',rrr:'num')->'arr':
    freq = get_freq(data)
    if freq == 0:
        print('Alert in risk_budget: 无法识别的周期！')
        return 0
    
    ret = data.pct_change()
    r = ret.mean().values * freq
    covar = ret.cov().values * freq
    return mean_var_optimization(r,covar,rrr)

#def mean_var_model(**kwargs):
#    if 'data' in kwargs:
#        return mean_var(kwargs['data'])
#    elif 'r' in kwargs:
#        if 'covar' in kwargs:
#            return mean_var_optimization(kwargs['r'], kwargs['covar'])
#        else:
#            return -1
#    else:
#        return -1

def risk_parity(data):
    n = len(data.columns)
    b = np.array([1/n] * n)
    return risk_budget(data,b)

def active_risk_parity(data,pi=[],c=2):
    n = len(data.columns)
    b = np.array([1/n] * n)
    if pi == []:
        pi = get_annualized_return_covar_and_freq(data)[0]
    return active_risk_budget(data,b,pi,c)

def ARP(data,dev,v_target,sharpe_target=1,tau=0.05):
    hr,covs,freq = get_annualized_return_covar_and_freq(data)
    w = risk_parity(data)
    noa = len(data.columns)
    
    P = np.eye(noa)
    Q = np.array([hr]).T   #使用资产历史收益率作为观点收益率
    tauV = tau * covs
#    Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])  
    Omega = np.dot(np.dot(P,covs),P.T) * np.eye(Q.shape[0])  
    A = linalg.inv(tau*covs)
    B = np.dot(P.T,linalg.inv(Omega))
#    lmbda = ((hr*w).sum() - rf) / np.sqrt(np.dot(w,np.dot(covs,w)))   #风险厌恶系数
    lmbda = sharpe_target / np.sqrt(np.dot(w,np.dot(covs,w)))
    PI = lmbda * np.dot(covs,w)    #均衡收益率
    mu = np.dot(linalg.inv(A+np.dot(B,P)),(np.dot(A,np.array([PI]).T) + np.dot(B,Q)))  #后验收
    
    weights = w
    
    cr = get_cum_return(data)  #由 data 计算出的各个资产从第一天开始的 “累计收益率”
    bm_r = (cr*weights).sum(1).diff()   #按照 weights 配比的组合的每日收益率
    
    def dev_cons(cr,x,bm_r):    #跟踪误差（为计算方便，使用方法）函数
    #cr 为各个资产的累计收益率
    #x 为配置权重
    #bm_r 为日基准收益率
        r = (cr * x).sum(1).diff()
        return ((r-bm_r)**2).dropna().mean()

#    bnds = tuple((max(0,wi-dev),min(1,wi+dev)) for wi in w)   
    bnds = tuple((0,1) for wi in w) 
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1},\
            {'type':'ineq', 'fun':lambda x:  dev*dev - freq * dev_cons(cr,x,bm_r)},\
            {'type':'ineq', 'fun':lambda x:  v_target*v_target - np.dot(x,np.dot(covs,x))})
    
    def bl_obj(mu,covs,lmbda,weights):
        return  np.dot(np.array(weights),mu)[0] - lmbda * np.dot(weights,np.dot(covs,weights))/2

    opts = sco.minimize(lambda weights: -bl_obj(mu,covs,lmbda,weights), weights, method = 'SLSQP', bounds = bnds, constraints = cons,tol=1e-25)
    s = Series(opts.x,index=data.columns)
    s.name = data.index[-1]
    return s


#def risk_budget_weight(data,b,lb):
#    n = len(data)
#    df_weights = DataFrame()
#    for i in range(lb,n):
#        weights = risk_budget(data[i-lb+1:i+1],b)
#        df_weights = df_weights.append(weights)
#    return df_weights


def active_risk_parity_resample(data,lb,freq,c=2):
    #lb: look back period
    dates = Series(data.index);
    l_dates = list(dates)
    dates_r = dates_resample(dates,freq)
    df_weights = DataFrame()
    for date in dates_r:
        k = l_dates.index(date)
        if k+1 < lb: continue   #前几期样本不足则过滤
        weights = active_risk_parity(data[k-lb+1:k+1],pi=[],c=c)
        df_weights = df_weights.append(weights)
    return df_weights


#def risk_budget_bt(data,b,lb,freq,lev=1):
#    return risk_budget_weight_resample(data,b,lb,freq,lev)


def risk_budget_weight_resample(data,b,lb,freq,lev=1):
    #lb: look back period
    dates = Series(data.index)
    l_dates = list(dates)
    dates_r = dates_resample(dates,freq)
    df_weights = DataFrame()
    for date in dates_r:
        k = l_dates.index(date)
        if k+1 < lb: continue   #前几期样本不足则过滤
        weights = risk_budget(data[k-lb+1:k+1],b,lev=lev)
        df_weights = df_weights.append(weights)
    return df_weights

def rp_weight_resample(data,lb,freq):
    #lb: look back period
    noa = len(data.columns)
    b = np.array([1/noa] * noa)
    return risk_budget_weight_resample(data,b,lb,freq)      

def ARP_weight_resample(data,dev,v_target,sharpe_target,tau,lb,freq):
    #lb: look back period
    s = Series(index=['dev','v_target','sharpe_target','tau','lb'],data=[dev,v_target,sharpe_target,tau,lb])
    print(s)
    dates = Series(data.index)
    l_dates = list(dates)
    dates_r = dates_resample(dates,freq)
    df_weights = DataFrame()
    for date in dates_r:
        k = l_dates.index(date)
        if k+1 < lb: continue    #前几期样本不足则过滤
        weights = ARP(data[k-lb+1:k+1],dev,v_target,sharpe_target,tau)
        df_weights = df_weights.append(weights)
    return df_weights

def equalweight_resample(data,freq):
    #lb: look back period
    dates = Series(data.index)
    noa = len(data.columns)
    l_dates = list(dates)
    dates_r = dates_resample(dates,freq)
    df_weights = DataFrame()
    for date in dates_r:
        weights = Series(data=[1/noa] * noa,index=data.columns)
        weights.name = date
        df_weights = df_weights.append(weights)
    return df_weights

def calc_nav(data,weights,fee=0):
    print(data)
    print(weights)
    
    assets = data.columns
    noa = len(assets)
    level1 = np.array(['data']*noa + ['returns']*noa + ['weights']*noa)
    level2 = list(assets)*3
    df=DataFrame(columns=[level1,level2],index=weights.index)
    df['data'] = data
    df['returns'] = df.data.pct_change()
    df['weights'] = weights
    
    df_nav = DataFrame()
    
    
#    df_nav['fee'] = abs(df.weights-df.weights.shift(1)).sum(1)*fee
#    df_nav['fee'][0]= fee
#    df_nav['fee']= df_nav['fee'].shift(1)
#    df_nav['fee'][0]=0

    fee_deduct = (abs(df.weights.diff()) * df.weights.shift(1)).sum(1)*fee
    fee_deduct[0] = fee
    df_nav['fee'] = fee_deduct.shift(1)
    df_nav['fee'][0] = 0

    df_nav['strategy_return_before_fee'] = (df.weights.shift(1)*df.returns).sum(1)
    df_nav['strategy_return_before_fee'][0] = 0
    df_nav['nav_before_fee'] = (df_nav['strategy_return_before_fee']+1).cumprod()
    
    df_nav['strategy_return'] = df_nav['strategy_return_before_fee'] - df_nav['fee']
    df_nav['nav'] = (df_nav['strategy_return']+1).cumprod()
    df_nav.index.name = 'date'
    return df_nav,df



#def calc_nav2(data,weights,fee=0):
#    assets = data.columns
#    noa = len(assets)
#    level1 = np.array(['data']*noa + ['nav']*noa + ['weights']*noa + ['shs']*noa)
#    level2 = list(assets)*4
#    df=DataFrame(columns=[level1,level2],index=data.index)
#    df['data'] = data
#    df['weights'] = weights
#    df['weights'] = df['weights'].fillna(method='ffill')
#    df['nav'] = 0
#    df['shs'] = 1.0
#    df.dropna(inplace=True)
#    df['nav'] = df.data / df.data.ix[0]   #各个资产价格归一
#    df.reset_index(inplace=True)
#    df['strategy','nav']=1
#      
#    for i in df.index:    
#        if  i == 0:
#            df.loc[i,('strategy','nav')] = 1
#            shs = df.weights.ix[i] / df.nav.ix[i]
#            shs -= shs * fee
#            df.loc[0,'shs']=list(shs)
#        else:
#            df.loc[i,('strategy','nav')] = sum(df.loc[i-1,'shs'] * df.loc[i,'nav'])
#            if all(df.weights.ix[i-1]==df.weights.ix[i]):   #权重都不变
#                df.loc[i,'shs'] = list(df.loc[i-1,'shs'])
#            else:
#                shs_last = df.shs.ix[i-1]
#                shs = df.loc[i,('strategy','nav')] * df.weights.ix[i] / df.nav.ix[i]
#                shs -= abs(shs-shs_last) * fee    #这个可能不太对, 如果shs 从100 降低到 0 , 那么shs 可能为负数
#                df.loc[i,'shs'] = list(shs)
#               
#    df.set_index('date',inplace=True)
#    df_nav = df[('strategy')]
#    return df_nav,df



#def calc_nav_reb(data,weights,reb_dates,fee=0):
#    pass
    
#def get_shs(price:Series,weight:Series,fee=0)->Series:
#    shs = weight / price
#    reb = weight.shift() != weight                                
#    shs[~reb] = np.nan
#    if isnull(shs[0]):
#        shs[0] = 0    
#    shs = shs.fillna(method='ffill')
#    
#    shs_change = abs(shs.shift()-shs)
#    shs_change[0] = shs[0]
#    fees = shs_change * price * fee
#    
#    mv = shs * price
##    return_contri = mv.pct_change() - fees
#    return return_contri


def test_all(data,lb,freq,dev,v_target,c,fee=5/1000):
    noa = len(data.columns)
    b = np.array([1/noa]*noa)
    
    weights_ew = equalweight_resample(data,freq)
    nav_eq = calc_nav(data[lb:],weights_ew,fee)[0]
    
    weights_rp = risk_budget_weight_resample(data,b,lb,freq)
    nav_rp =  calc_nav(data,weights_rp,fee)[0]
    
    weights_htarp = ARP_weight_resample(data,dev,v_target,1,0.025,lb,freq)
    nav_htarp =  calc_nav(data,weights_htarp,fee)[0]

    weights_arp = active_risk_parity_resample(data,lb,freq,c)
    nav_arp = calc_nav(data,weights_arp,fee)[0]
    
    df_nav = DataFrame()
    df_nav['arp'] = nav_arp['nav']
    df_nav['HT_arp'] = nav_htarp['nav']
    df_nav['RP'] = nav_rp['nav']
    df_nav['EW'] = nav_eq['nav']
    
    return df_nav
    

#df = test_all(data,lb=100,freq='m',dev=0.1,v_target=0.1,c=8,fee=3/1000)

#data=read_excel('BB.xlsx','RP_data',index_col='date')
#weights = risk_budget_weight_resample(data,b=np.array([1/3,1/3,1/3]),lb=250,freq='m')
