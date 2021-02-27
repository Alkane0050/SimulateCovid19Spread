import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,norm,poisson
from scipy.misc import factorial
import random

x = np.linspace(1,60,60)
y = 5*x + np.random.normal(0,15,60) + 20

## y = beta0 + beta1 * x
# beta0,beta1 = 20, 5

plt.scatter(x,y)
plt.show()
eps = 1e-10
def likelyHood(beta0,beta1,sigma=1):
    R = y - x*beta1 - beta0
    res = np.sum(norm.logpdf(R,loc=0,scale=sigma))
    if res is np.nan:
        print(beta0,beta1,sigma)
    return res

def prior(beta0,beta1,sigma=1):
    beta0prior = uniform.logpdf(beta0,loc=-100,scale=200)
    beta1prior = uniform.logpdf(beta1,loc=-100,scale=200)
    sigmaprior = norm.logpdf(sigma,loc=0,scale=10)
    # print(beta0prior,beta1prior,sigmaprior)
    return beta0prior+beta1prior+sigmaprior

def posterior(beta0,beta1,sigma=1):
    return likelyHood(beta0,beta1,sigma) + prior(beta0,beta1,sigma)

def genVal(vector):
    return (norm.rvs(loc=vector,scale=(0.1,0.5,0.3)))

T = 50000
MChain = np.zeros((T,3))
MChain[0,:] = np.array([4,20,1])
for t in range(T-1):
    proVal = genVal(MChain[t,:])
    a = posterior(proVal[0],proVal[1],proVal[2])
    b = posterior(MChain[t,:][0],MChain[t,:][1],MChain[t,:][2])
    # print(MChain[t,:],proVal)
    # print(posterior(proVal[0],proVal[0],proVal[0]),posterior(MChain[t,:][0],MChain[t,:][1],MChain[t,:][2]))
    prob = min(np.exp(a-b+eps),1)
    u = random.random()
    if u < prob:
        MChain[t+1,:] = proVal
        # print(t)
    else:
        MChain[t+1,:] = MChain[t,:]

plt.subplot(1,3,1),plt.plot(MChain[:,0])
plt.subplot(1,3,2),plt.plot(MChain[:,1])
plt.subplot(1,3,3),plt.plot(MChain[:,2])
plt.show()

