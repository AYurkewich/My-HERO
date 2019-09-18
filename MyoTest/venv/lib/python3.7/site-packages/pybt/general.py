import numpy as np
import pandas as pd

def to_db_code(ts_code):
    l = ts_code.split('.')
    return l[1]+'_'+l[0]

def to_ts_code(db_code):
    return db_code[2:] + '.' + db_code[:2]

def real_round_2(x):
    return int((int(x*1000)+5)/10)/100

def check_df_struct(df1, df2):
    if not df1.shape == df2.shape:
        return False
    if np.not_equal(np.array(df1.index),np.array(df2.index)).any():
        return False
    if np.not_equal(np.array(df1.columns),np.array(df2.columns)).any():
        return False
    return True

