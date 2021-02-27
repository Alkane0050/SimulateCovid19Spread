import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from ode.migration import *
from ode.solver import odeSolver
import traceback
import warnings
# warnings.filterwarnings('error')

def dSEIR(vector,t,params):
    """

    :param vector:
    :param t: time
    :param params:
    :return:
    """
    S,E,I1,I2,R = vector
    I = I1+ I2
    R0,Di,De,Da= params
    N = S+E+I+R
    if N>1e15 or N<1e5 : # Overflow
        print("Overflow",params)
        return np.array([0,0,0,0,0])
    t = np.ceil(t)
    dSdt = -(R0 / Di) * S * I / N
    dEdt = (R0 / Di) * S * I / N - E / De
    dI1dt = E / De - I1 / (Di - Da)
    dI2dt = I1 / (Di - Da) - I2 / Da
    dRdt = I2 / Da
    return np.array([dSdt, dEdt, dI1dt, dI2dt, dRdt])

def SEIR(initVector=None,t=None,params=None):
    if initVector is None:
        print("Error: initVector can't be none!")
    if params is None:
        print("params can't be None!")
    P = odeint(dSEIR,initVector,t,args=(params,))
    S,E,I1,I2,R = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4]
    return S,E,I1,I2,R

