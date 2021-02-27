import numpy as np
import pickle
import datetime
with open("../data/dicOfMatrix.pickle",'rb') as f:
    data = pickle.load(f)
with open("../data/factor_in_2020.pickle",'rb') as f:
    factor_in = pickle.load(f)
with open("../data/factor_out_2020.pickle",'rb') as f:
    factor_out = pickle.load(f)

coef = 9.3 # 迁徙规模修正系数
def L_iw():
    """
    international to WuHan
    :return: # of people fly to WuHan from international.
    """
    return 0 #just my toy number

def L_cw(t):
    """
    China to WuHan
    :param t: time
    :return: # of people come to WuHan from China.
    """
    if t <= 31:
        time = datetime.timedelta(int(t)) + datetime.date(2020,1,1)
        mat = data[time]
        fvector = factor_out[:,(time-datetime.date(2020,1,1)).days]*coef
        result = 0.01*mat[:,0] #取武汉这一列
        # print(result)
        result = fvector.dot(result)
        return result
    elif t >= 32:
        return np.mean([L_cw(i) for i in range(25,31)])*np.exp(-1*(t-31))




def L_wi():
    """
    WuHan to international
    :return: # of people fly from WuHan to international.
    """
    return 0 # just my toy number

def L_wc(t):
    """
    WuHan to China
    :param t: time
    :return:# of people from WuHan to other China cities.
    """
    begin = datetime.date(2020,1,1)
    if t <= 31:
        time = datetime.timedelta(int(t)) + datetime.date(2020, 1, 1)
        mat = data[time]
        fvector = factor_out[0,(time-begin).days] * coef
        result = 0.01 * mat[0,:]  # 取武汉这一行
        # print(result)
        result = np.sum(result) * fvector
        return result
    elif t >= 32:
        return np.mean([L_wc(i) for i in range(25, 31)])*np.exp(-1*(t-32))

def z(t):
    """
    zoonotic force of infection
    :param t: time
    :return: # of people got sick because of zoonotic.
    """
    if t < 1:
        return 82
    else:
        return 0

def R0(t):
    """
    基本感染数 : 平均一个患者感染几个人
    :param t:
    :return:
    """
    if t < 60:
        return 2.5
    else:
        return 0.5
Di = 7.5 # mean infectious period 感染期 (平均一个病人感染多久就会死亡或是痊愈)
De = 6.5 # mean latent period 潜伏期 (平均一个感染者需要多久才会表现出症状)

