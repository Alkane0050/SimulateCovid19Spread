from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,norm,poisson,exponweib,lognorm
from scipy.misc import factorial
import random
import seaborn as sns
from ode.Model import SEIR,drawSEIR
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.optimize import minimize
data = pd.read_csv("../data/originalData.csv")
xd = np.array(data).reshape(-1)[0:]
initVector = [10800000,0,1, 0]
incubation = 7
start = 30
T = 100
alpha = 1
params = (4.03535257, 6.63128893,10 )
def deltaArray(array):
    res = []
    for i in range(len(array)-1):
        a0 = array[i]
        a1 = array[i+1]
        res.append(a1-a0)
    return np.array(res)

def LoglikeliHood(params):
    beta,sigma,gamma = params
    t = np.arange(0,len(xd)+incubation)
    _,E,I,_ = SEIR(initVector,t,params)
    Lambda = deltaArray((I)[incubation-1:])
    M = np.sum(xd[start:])
    eps = 1e-6
    indexVector = np.array(abs(xd-Lambda)<1,dtype=int)
    prob = np.sum((np.log(eps+alpha*poisson.pmf(k=xd[start:],mu=Lambda[start:])+(1-alpha)*indexVector[start:]*xd[start:]/M)))
    return prob

def negLoglikelihood(params):
    return -LoglikeliHood(params)

def prior(params):
    R0,Di,De = params
    # DiPrior = norm.logpdf(Di,loc = 12, scale = 5)
    # DePrior = norm.logpdf(De,loc = 6.5, scale = 3)
    DiPrior = uniform.logpdf(Di,loc =2, scale =10)
    DePrior = uniform.logpdf(De,loc=2, scale =10)
    R0Prior = uniform.logpdf(R0,loc=1,scale=5)
    return DiPrior + DePrior + R0Prior

def genVal(params):
    R0,Di,De = params
    di,de,r0 = norm.rvs(loc=(Di,De,R0),scale=(0.1,0.1,0.1))
    return (r0,di,de)

def posterior(params):
    return prior(params) + LoglikeliHood(params)

def MCMC(initParam,iters):
    T = iters
    MChain = np.zeros((T,len(initParam)))
    MChain[0,:] = np.array(initParam).reshape(-1)
    cnt = 0
    for t in range(T-1):

        u = random.random()
        prob = min(np.exp(posterior(newVector)-posterior(MChain[t,:])),1)
        if u < prob:
            MChain[t+1,:] = newVector
            cnt +=1
        else:
            MChain[t+1,:] = MChain[t,:]

    return MChain,cnt

def estimate(params):
    t = np.arange(0, len(xd) + 10 + incubation)
    S, E, I, R = SEIR(initVector=initVector, t=t, params=params)
    begin = datetime.date(2020,12,8) - datetime.timedelta(days=incubation)
    realDays = [begin + datetime.timedelta(days=i + incubation) for i in range(len(xd))]
    days = [begin + datetime.timedelta(days=i) for i in range(len(t))]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
    plt.gcf().autofmt_xdate()
    plt.plot(days,(I))
    plt.scatter(realDays, np.cumsum(xd))
    plt.legend(
        ['Estimated Number of Infections',
         'Real Number of Infections', ])
    print(params,"\nlog(prior):{}\nlog(likelihood):{}".format(prior(params),LoglikeliHood(params)))

def showChain(chain):
    plt.figure()
    plt.subplot(3, 1, 1), plt.plot(chain[:, 0]), plt.title("$R0$")
    plt.subplot(3, 1, 2), plt.plot(chain[:, 1]), plt.title("$Di$")
    plt.subplot(3, 1, 3), plt.plot(chain[:, 2]), plt.title("$De$")
    plt.show()

if __name__ == "__main__":
    print("Start")
    paramSet = []
    count = 0
    for i in range(T):
        Chain,cnt = MCMC(params,10000)
        count+=cnt
        params = Chain[-5:].mean(axis=0)
        if i % 10 ==0:
            showChain(Chain)
            plt.figure(),plt.title("iter:{} :accept rate:{}".format(i,count*1.0/(10000*(i+1))))
            estimate(params)
            plt.show()
        np.random.shuffle(Chain)
        paramSet.extend([[Chain[i,0],Chain[i,1],Chain[i,2]] for i in range(0,1000)])
    estimate(params)
    plt.savefig("result.svg")
    paramSet = np.array(paramSet)
    plt.subplot(1, 3, 1), sns.distplot(paramSet[:, 0],norm_hist=True),plt.title("$R0$")
    plt.subplot(1, 3, 2), sns.distplot(paramSet[:, 1],norm_hist=True),plt.title("$Di$")
    plt.subplot(1, 3, 3), sns.distplot(paramSet[:, 2],norm_hist=True),plt.title("$De$")
    plt.show()

