from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ode.Model import SEIR
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sample.estimate import deltaArray,prior,LoglikeliHood
params = (2.2,12.5,5.2)
data = pd.read_csv("../data/originalData.csv")
xd = np.array(data).reshape(-1)[0:]
incubation = 14
start = 0
T = 12
alpha = 1
t = np.arange(0, len(xd) + 10 + incubation)
def fit(E,I):
    im = []
    e,i = E,I
    initVector = [10800000,E,I, 0]
    S, E, I, R = SEIR(initVector=initVector, t=t, params=params)
    begin = datetime.date(2020, 12, 8) - datetime.timedelta(days=incubation)
    realDays = [begin + datetime.timedelta(days=i + incubation) for i in range(len(xd))]
    days = [begin + datetime.timedelta(days=i) for i in range(len(t))]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
    plt.gcf().autofmt_xdate()
    im=plt.plot(days[incubation:], (I + E)[incubation:],color= 'b')
    im+=plt.plot(days[incubation:], (I)[incubation:],color = 'g')
    im+=plt.scatter(realDays, np.cumsum(xd),color = 'r').findobj()
    im+= plt.text(0.5,1.01,(e,i), ha="center",va="bottom",color=np.random.rand(3),
                     transform=ax.transAxes, fontsize="large").findobj()
    # im+=plt.title("E is {}".format(e)).findobj()
    # plt.legend(
    #     ['Estimated Number of Infections + latents',
    #      'Estimated Number of Infections',
    #      'Real Number of Infections', ])
    print(params, "\nlog(prior):{}\nlog(likelihood):{}".format(prior(params), LoglikeliHood(params)))
    return im

if __name__ == "__main__":
    print("hhh")
    fig,ax = plt.subplots()
    ims = []
    T = 500
    Es = range(1,T)
    Is = [1 for i in range(T)]
    for e,i in zip(Es,Is):
        ims.append(fit(e,i))
    ani = animation.ArtistAnimation(fig, ims,interval=200, repeat_delay=1000)
    ani.save("test.gif",writer='pillow')