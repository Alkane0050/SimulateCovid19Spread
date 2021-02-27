import pickle
from ode.Model import SEIR
import numpy as np


def dCitySEIR(vector,t,params):
    """
    建立城市之间流动模型的SEIR
    :param vector:
    :param t: time
    :param params:R0,Di,De,Lsi,Lei,Lii,Lri,Lo
    :return:dF
    """
    S,E,I1,I2,R = vector
    R0,Di,De,Da,Lsi,Lei,Li1i,Li2i,Lri,Lo= params
    N = S+E+I1+I2+R
    I = I1+I2
    t = np.ceil(t)
    dSdt = -(R0/Di)*S*I/N + Lsi - Lo*S/N
    dEdt = (R0/Di)*S*I/N- E/De + Lei - Lo*E/N
    dI1dt = E/De - I1/(Di-Da) + Li1i - Lo*I1/N
    dI2dt = I1/(Di-Da) - I2/Da +Li2i - Lo*I2/N
    dRdt = I2/Da + Lri - Lo*R/N
    return np.array([dSdt,dEdt,dI1dt,dI2dt,dRdt])




