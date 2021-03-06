import numpy as np
from scipy.integrate import odeint

"""
定义常微分方程，给出各方向导数,即速度
"""


def dmove(Point, t, sets):
    """
    p：位置矢量
    sets：其他参数
    """
    p, r, b = sets
    x, y, z = Point
    return np.array([p * (y - x), x * (r - z)-y, x * y - b * z])


t = np.arange(0, 30, 0.01)
# 调用odeint对dmove进行求解，用两个不同的初始值
P1 = odeint(dmove, (0., 1.011, 0.), t, args=([10., 28., 3.],))  # (0.,1.,0.)是point的初值
# ([10.,28.,3.],)以元祖的形式给出 point,t之后的参数
P2 = odeint(dmove, (0., 1.01, 0.), t, args=([10., 28., 3.],))

"""
画3维空间的曲线
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(P1[:, 0], P1[:, 1], P1[:, 2])
ax.plot(P2[:, 0], P2[:, 1], P2[:, 2])
plt.show()