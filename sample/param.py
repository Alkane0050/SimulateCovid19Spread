from sample.estimate import negLoglikelihood
from scipy.optimize import minimize
import numpy as np
init = np.array([0.5,0.18,0.18])
print(minimize(negLoglikelihood,init,options={'gtol': 1e-6, 'disp': True}))