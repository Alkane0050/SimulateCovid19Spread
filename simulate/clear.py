import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1,10000)
y = np.sin(np.pi*10**2/(np.sqrt(x**2+10**2)+x))
plt.plot(x,y)
plt.show()