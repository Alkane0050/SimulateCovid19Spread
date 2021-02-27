import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from ode.migration import *
from ode.solver import odeSolver

def dSEIR(vector,t,params):
    """

    :param vector:
    :param t: time
    :param params:
    :return:
    """
    S,E,I,R = vector
    R0,Di,De= params
    N = S+E+I+R
    if N>1e15 or N<1e5 : # Overflow
        print("Overflow",params)
        return np.array([0,0,0,0])
    t = np.ceil(t)
    dSdt = -(R0/Di)*S*I/N
    dEdt = (R0/Di)*S*I/N - E/De
    dIdt = E/De - I/Di
    dRdt = I/Di
    return np.array([dSdt,dEdt,dIdt,dRdt])

def SEIR(initVector=None,t=None,params=None):
    if initVector is None:
        initVector = [8000000,0,1,0]
    if params is None:
        print("params can't be None!")
    P = odeint(dSEIR,initVector,t,args=(params,))
    S,E,I,R = P[:,0],P[:,1],P[:,2],P[:,3]
    return S,E,I,R
def drawSEIR(initVector,t,params):
    S, E, I, R = SEIR(initVector, t, params)
    plt.figure()
    # plt.plot(S)
    plt.plot(E)
    plt.plot(I)
    plt.plot(R)
    plt.legend(['E','I','R'])
    plt.show()


if __name__ == "__main__":
    print("Start Test SEIR Model")
    initVector = [10000,10,2,3]
    t = np.arange(0,30)
    params = [1,2,3]
    S,E,I,R = SEIR(initVector,t,params)