import numpy as np

def abs2percF(s, p):
    '''
    将绝对量数据转化为增长率，前移 ( e.g. [1,2,3] -> [2,1.5,NA] )
    s : array
    p : look-back period
    '''
    return np.append((s[p:] - s[:-p])/s[:-p], np.repeat(np.nan, p))

def abs2percB(s, p):
    '''
    将绝对量数据转化为增长率，后移 ( e.g. [1,2,3] -> [NA,2,1.5] )
    s : array
    p : look-back period
    '''
    return np.append(np.repeat(np.nan, p), (s[p:] - s[:-p])/s[:-p])

def laggingF(s, l):
    '''
    向前平移时间序列 ( e.g. [1,2,3] -> [2,3,NA] )
    s : array
    l : lagging period
    '''
    return np.append(s[l:], np.repeat(np.nan, l))

def laggingB(s, l):
    '''
    向后平移时间序列 ( e.g. [1,2,3] -> [NA,1,2] )
    s : array
    l : lagging period
    '''
    return np.append(np.repeat(np.nan, l), s[:-l])
