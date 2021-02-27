import numpy as np
def odeSolver(dF,initVector,solveSpace,args):
    """
    一阶模拟微分方程组
    :param dF:
    :param initVector:
    :param solveSpace:
    :param args:
    :return:
    """
    resultSet = np.zeros((len(solveSpace)+1,len(initVector)))
    vector = np.array(initVector,dtype='float64')  # 起始的值
    resultSet[0] = vector
    for t in solveSpace:
        delVector = dF(vector,t,args)
        vector += delVector
        resultSet[t+1] = vector
    return resultSet[:-1]


