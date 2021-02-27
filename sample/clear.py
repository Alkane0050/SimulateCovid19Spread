import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import exponweib,lognorm
data = pd.read_csv("data.csv")
data = data.drop_duplicates() # 去重
data['time'] = pd.to_datetime(data['time'],format='%Y/%m/%d')
data['time'] = data['time'].dt.date
data = data.sort_values(by=['time'])
data = data.set_index('time').groupby('time').max()
res = []
begin = datetime.date(2020,1,24)
for i in range(len(data)-1):
    day0 = begin + datetime.timedelta(days=i)
    day1 = begin + datetime.timedelta(days=i+1)
    initDay = day1 - datetime.timedelta(days=13)
    deltaCases = data[day1:day1]['cases'].max() - data[day0:day0]['cases'].max()
    res.append(deltaCases)
    print(initDay,deltaCases)
print(res)
res = np.array(res)
np.savetxt("res.csv",res,delimiter='\n',fmt='%d')

