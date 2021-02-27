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
    R0,Di,De,L_iw,L_cw,L_wi,L_wc,z = params
    N = S+E+I+R
    t = np.ceil(t)
    dSdt = -S/N*(R0(t)/Di*(I) + z(t)) + L_cw(t) - (L_wc(t))*S/N
    dEdt = S/N*(R0(t)/Di*(I) + z(t)) - E/De - (L_wc(t))*E/N
    dIdt = E/De - I/Di - (L_wc(t))*I/N
    dRdt = I/Di -(L_wc(t))*R/N
    return np.array([dSdt,dEdt,dIdt,dRdt])


S,E,I,R = 11080000,105,20,0
vector1 = [11080000,105,20,0]
vector2 = [11080000,105,20,0]
params = (R0,Di,De,L_iw,L_cw,L_wi,L_wc,z)

t = np.arange(0,200)

P1 = odeint(dSEIR,vector1,t,args=([R0,Di,De,L_iw,L_cw,L_wi,L_wc,z],))
P2 = odeSolver(dSEIR,vector2,t,args=([R0,Di,De,L_iw,L_cw,L_wi,L_wc,z]))
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,P1[:,1]),plt.plot(t,P1[:,2])
plt.legend(['E','I']),plt.title("scipy")
plt.subplot(1,2,2)
plt.plot(t,P2[:,1]),plt.plot(t,P2[:,2])
plt.legend(['E','I']),plt.title("hypy")
plt.show()



