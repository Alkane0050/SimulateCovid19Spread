import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
with open("../data/factor_out_2020.pickle",'rb') as f:
    factor_out = pickle.load(f)
with open("../data/factor_in_2020.pickle",'rb') as f:
    factor_in = pickle.load(f)
with open("../data/dicOfMatrix.pickle",'rb') as f:
    matrices = pickle.load(f)
coef = 9.3*10000 # 迁徙规模修正系数
def Lout(i,t):
    """
    城市i在第t天的人口流出
    :param i:
    :param t:
    :return: [Lo_0,Lo_1,...] 第j位表示流出至城市j的数量
    """
    begin = datetime.date(2020, 1, 1)
    day = begin + datetime.timedelta(days=t)
    matrix = matrices[day]
    outRate = matrix[i,:]
    factor = factor_out[i,t]*coef
    out = outRate*factor*0.01
    return out

def Lin(i,t):
    """
    第t天流入城市i的人口
    :param i:
    :param t:
    :return:number
    """
    begin = datetime.date(2020, 1, 1)
    day = begin + datetime.timedelta(days=t)
    # matrix = matrices[day]
    IN = factor_in[i,t]*coef
    return IN

def dSpread(Lo,vector,i):
    """
    第i个城市处于vector状态[S,E,I1,I2,R]时,向外流动的人口
    :param Lo:
    :param vector:
    :param i:
    :return:[dLSin,dLEin,...,dLRin]
    其中dLSin,dLEin,...都是向量
    """
    S,E,I1,I2,R = vector
    N = np.sum(vector)
    dLSin = Lo * S/N
    dLEin = Lo * E/N
    dLI1in = Lo * I1/N
    dLI2in = Lo * I2/N
    dLRin = Lo * R/N
    return np.array([dLSin,dLEin,dLI1in,dLI2in,dLRin])







