import numpy as np
import matplotlib.pyplot as plt
def deltaArray(array):
    res = []
    for i in range(len(array)-1):
        a0 = array[i]
        a1 = array[i+1]
        res.append(a1-a0)
    return np.array(res)
def showChain(chain):
    plt.figure()
    plt.subplot(4, 1, 1), plt.plot(chain[:, 0]), plt.title("$R0$")
    plt.subplot(4, 1, 2), plt.plot(chain[:, 1]), plt.title("$Di$")
    plt.subplot(4, 1, 3), plt.plot(chain[:, 2]), plt.title("$De$")
    plt.subplot(4, 1, 4), plt.plot(chain[:, 3]), plt.title("$Da$")
    plt.show()