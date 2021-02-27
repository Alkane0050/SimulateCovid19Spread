import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ode.Model2 import dSEIR,SEIR
from simulate.tools import deltaArray,showChain
import datetime
from scipy.stats import uniform,norm,poisson
import random
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
data = pd.read_csv("../data/originalData.csv")
data = np.array(data,dtype=float)
xd = data[0:].reshape(-1) # wuhan data
initVector = [10800000,0,0,1, 0]
params = (5,  8, 10,   2) #     R0,Di,De,Da = params
incubation = 7
start = 30
T = 100
alpha =1

def LoglikeliHood(params):
    t = np.arange(0,len(xd)+incubation)
    _,E,I1,I2,_ = SEIR(initVector,t,params)
    I = I1+I2
    eps = 1e-5
    Lambda = deltaArray((I)[incubation-1:])
    M = np.sum(xd[start:])
    indexVector = np.array(abs(xd-Lambda+0.5)<1,dtype=int)

    prob = np.sum((np.log(eps+alpha*poisson.pmf(k=xd[start:],mu=Lambda[start:])) + (1-alpha)*indexVector[start:]*xd[start:]/M)) #
    return prob

def prior(params):
    R0,Di,De,Da = params
    DiPrior = uniform.logpdf(Di,loc =2, scale =10)
    DePrior = uniform.logpdf(De,loc=2, scale =10)
    DaPrior = uniform.logpdf(Da, loc=1, scale=Di)
    R0Prior = uniform.logpdf(R0,loc=1,scale=5)
    return DiPrior + DePrior + R0Prior + DaPrior

def genVal(params):
    R0,Di,De,Da = params
    r0,di,de,da = norm.rvs(loc=(R0,Di,De,Da),scale=(0.1,0.05,0.05,0.05))
    return (r0,di,de,da)

def posterior(params):
    return prior(params) + LoglikeliHood(params)

def MCMC(initParam,iters):
    T = iters
    MChain = np.zeros((T,len(initParam)))
    MChain[0,:] = np.array(initParam).reshape(-1)
    cnt = 0
    for t in range(T-1):
        newVector = genVal(MChain[t,:])
        u = random.random()
        a = posterior(newVector)
        b = posterior(MChain[t,:])
        prob = min(np.exp(a-b),1)
        if u < prob:
            MChain[t+1,:] = newVector
            cnt +=1
        else:
            MChain[t+1,:] = MChain[t,:]

    return MChain,cnt

def estimate(params):
    t = np.arange(0, len(xd) + 10 + incubation)
    S, E, I1,I2, R = SEIR(initVector=initVector, t=t, params=params)
    I = I1+I2
    begin = datetime.date(2019,12,8) - datetime.timedelta(days=incubation)
    realDays = [begin + datetime.timedelta(days=i + incubation) for i in range(len(xd))]
    days = [begin + datetime.timedelta(days=i) for i in range(len(t))]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
    plt.gcf().autofmt_xdate()
    plt.plot(days, (I))
    plt.scatter(realDays, np.cumsum(xd))
    plt.legend(
        ['Estimated Number of I',
         'Real Number of Infections', ])
    print(params,"\nlog(prior):{}\nlog(likelihood):{}".format(prior(params),LoglikeliHood(params)))

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
        paramSet.extend([[Chain[i,0],Chain[i,1],Chain[i,2],Chain[i,3]] for i in range(0,1000)])
    estimate(params)
    plt.savefig("result.svg")
    paramSet = np.array(paramSet)
    plt.subplot(4, 1, 1), plt.hist(paramSet[:, 0],density=True),plt.title("$R0$")
    plt.subplot(4, 1, 2), plt.hist(paramSet[:, 1],density=True),plt.title("$Di$")
    plt.subplot(4, 1, 3), plt.hist(paramSet[:, 2],density=True),plt.title("$De$")
    plt.subplot(4, 1, 4), plt.hist(paramSet[:, 3],density=True),plt.title("$Da$")
    plt.show()

