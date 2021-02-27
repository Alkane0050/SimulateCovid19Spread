import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_style('darkgrid')
with open("../sample/SEIRparamSet.pickle",'rb') as f:
    paramSet = pickle.load(f)
sigma = stats.sem(paramSet)
mu = np.mean(paramSet,axis=0)
lo,hi = stats.t.interval(0.95, len(paramSet[:,0]), loc=mu, scale=sigma)
pl,ph = stats.t.interval(0.99, len(paramSet[:,0]), loc=mu, scale=sigma)
width = hi - lo
prior = [(i,j) for i,j in zip(pl,ph)]
name = ['$R_0$','$D_i$','$D_e$','$D_a$']
def plotHist(param,lo,hi,priorRange,name=None):
    paramS1 = []
    paramS2 = []
    st,ed = priorRange
    for i in param:
        if st < i and i < ed:
            paramS1.append(i)
            if lo <= i and i<=hi:
                paramS2.append(i)
    sns.distplot(paramS1)
    # plt.hist(paramS2, range=[lo, hi],)
    plt.title(name),plt.show()

for i in range(3):
    l = lo[i]
    h = hi[i]
    param = paramSet[:,i]
    priorRange = prior[i]
    Pname = name[i]
    plotHist(param,l,h,priorRange,Pname)
