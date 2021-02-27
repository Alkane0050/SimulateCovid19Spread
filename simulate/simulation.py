from matplotlib import animation
from simulate.citySEIR import dCitySEIR
from simulate.migration import Lout,Lin,dSpread
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['PingFang SC'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def simulate(initSEIR,days,hparams):
    """
    模拟2019-nCoV在46个重要城市之间的传播
    :param days:
    :param hparams:
    :return:
    """
    ## params
    T = days #模拟天数
    R0,Di,De,Da = hparams
    ## init
    with open("../data/cases.pickle",'rb') as f:
        cases = pickle.load(f)
    with open("../data/cityPopulation.json",encoding='utf-8') as f:
        population = json.load(f)

    numCities = len(population)
    citysSEIR = np.zeros((numCities,5,T))
    for i,S in enumerate(population):
        citysSEIR[i,0,0] = population[S]*10000
    citysSEIR[0,:,0] = initSEIR
    LSin,DLSin   = np.array([Lin(i,0) for i in range(numCities)],dtype=float),np.array([0 for i in range(numCities)],dtype=float)
    LEin,DLEin   = np.array([0 for i in range(numCities)],dtype=float),np.array([0 for i in range(numCities)],dtype=float)
    LI1in,DLI1in = np.array([0 for i in range(numCities)],dtype=float),np.array([0 for i in range(numCities)],dtype=float)
    LI2in,DLI2in = np.array([0 for i in range(numCities)],dtype=float),np.array([0 for i in range(numCities)],dtype=float)
    LRin,DLRin   = np.array([0 for i in range(numCities)],dtype=float),np.array([0 for i in range(numCities)],dtype=float)
    for t in range(T-1):
        for i in range(numCities):
            Lo = Lout(i,t) # 第t天流出城市i的人数
            Lsi,Lei,Li1i,Li2i,Lri= LSin[i],LEin[i],LI1in[i],LI2in[i],LRin[i]
            params = R0,Di,De,Da,Lsi,Lei,Li1i,Li2i,Lri,np.sum(Lo)
            initVector = citysSEIR[i,:,t]
            dF = dCitySEIR(initVector,t,params=(params))
            citysSEIR[i,:,t+1] = citysSEIR[i,:,t] + dF
            DLSin += dSpread(Lo,initVector,i)[0]
            DLEin += dSpread(Lo,initVector,i)[1]
            DLI1in+= dSpread(Lo,initVector,i)[2]
            DLI2in+= dSpread(Lo,initVector,i)[3]
            DLRin += dSpread(Lo,initVector,i)[4]
        LSin,LEin,LI1in,LI2in,LRin = DLSin,DLEin,DLI1in,DLI2in,DLRin
        DLSin = np.zeros_like(DLSin)
        DLEin = np.zeros_like(DLEin)
        DLI1in = np.zeros_like(DLI1in)
        DLI2in = np.zeros_like(DLI2in)
        DLRin = np.zeros_like(DLRin)
        # print(t)
    return citysSEIR

def visualize(citysSEIR):
    with open("../data/cityAndId.json",encoding='utf-8') as f:
        citys = json.load(f)
    citys = [i for i in citys]
    y_pos = np.arange(len(citys))
    numcities,_,T = citysSEIR.shape
    fig, ax = plt.subplots(2,2,figsize=(30,20))
    for i in range(2):
        for j in range(2):
            ax[i,j].set_yticks(y_pos)
            ax[i,j].set_yticklabels(citys)
    x = range(numcities)
    # S = ax[0, 0].barh(x, citysSEIR[:,0,T-1])
    E = ax[0,0].barh(x,citysSEIR[:,1,T-1])
    I1 = ax[0, 1].barh(x,citysSEIR[:,2,T-1])
    I2 = ax[1, 0].barh(x,citysSEIR[:,3,T-1])
    R = ax[1, 1].barh(x,citysSEIR[:,4,T-1])
    # N = ax[1, 2].barh(x,np.sum(citysSEIR[:,:,T-1],axis=1))
    SEIR = [E,I1,I2,R]
    def animate(i):
        # tex1 = ax[0, 0].set_title("S day{}".format(str(i + 1)))
        tex2 = ax[0, 0].set_title("E day{}".format(str(i + 1)))
        tex3 = ax[0, 1].set_title("I1 day{}".format(str(i + 1)))
        tex4 = ax[1, 0].set_title("I2 day{}".format(str(i + 1)))
        tex5 = ax[1, 1].set_title("R day{}".format(str(i + 1)))
        # tex6 = ax[1, 2].set_title("N day{}".format(str(i + 1)))
        for j,bar in enumerate(SEIR):
            if j ==5:
                y =  np.sum(citysSEIR[:,:,i],axis=1)
            else:
                y = citysSEIR[:,j+1,i]
            for t, b in enumerate(bar):
                b.set_width(y[t])

    anim = animation.FuncAnimation(fig, animate, repeat=True, blit=False, frames=T-1,
                                   interval=200)
    anim.save('mymovie.gif', writer='pillow')
    plt.show()


if __name__ == "__main__":
    R0 = 2.2
    Di = 5
    De = 12
    Da = 3
    T = 30
    initSEIR = [1108.1*10000,10,100,1,0] # 武汉的S,E,I1,I2,R

    citysSEIR = simulate(initSEIR,days=T,hparams=(R0,Di,De,Da))
    visualize(citysSEIR)









