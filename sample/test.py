from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import numpy as np

lambdaa = 0.5
def targetFunc(x):
    return 1/np.pi*(lambdaa/(lambdaa**2+(x-0)**2))

T = 10000
pi = [0 for i in range(T)]

for t in range(T-1):
    p_ = norm.rvs(loc=pi[t], scale=1, size=1, random_state=None)[0]   #状态转移进行随机抽样
    u = random.random() # U[0,1]
    tmp = min(targetFunc(p_)/targetFunc(pi[t]),1)
    if u < tmp:
        pi[t+1] = p_
    else:
        pi[t+1] = pi[t]
num_bins = 50
t = np.linspace(-5,5,1000)
plt.scatter(t,targetFunc(t))
plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7,label='Samples Distribution')
plt.show()


