import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("DXYArea.csv")
# data = data.groupby(['provinceName','cityName','updateTime']).max
sCity = []
for i in data['provinceName']:
    if i.find('市')!=-1:
        sCity.append(i)
sCity = set(sCity)
citysData = []
for city in sCity:
    cityData = data[data['provinceName']== city ][['provinceName','province_confirmedCount','updateTime']]
    cityData['cityName'] = city
    cityData[['provinceName','cityName','city_confirmedCount','updateTime']] = cityData[['provinceName','cityName','province_confirmedCount','updateTime']]
    cityData = cityData[['provinceName','cityName','city_confirmedCount','updateTime']]
    data = data[data['provinceName']!=city]
    citysData.append(cityData)
data = data[['provinceName','cityName','city_confirmedCount','updateTime']]
for cityData in citysData:
    data = data.append(cityData)
data['updateTime'] = pd.to_datetime(data['updateTime'],format='%Y-%m-%d')
data['updateTime'] = data['updateTime'].dt.date
data = data.groupby(['provinceName','cityName','updateTime']).max()
table = []
citys = []
tmp = []

for provinceName,cityName,updateTime in data.index:
    val = data.loc[provinceName,cityName,updateTime].get_values()
    s = [provinceName+cityName,updateTime,int(val)]
    table.append(s)
    citys.append((provinceName,cityName))
data = pd.DataFrame(table,columns=['城市','时间','病例数'])
tmp = list(set(citys))
tmp.sort(key = citys.index)
citys = tmp
dic = dict()
for city in citys:
    cityDic = dict()
    a,b = city
    cityName = a+b
    for day in data[data['城市']==cityName]['时间']:
        val = data[(data['城市'] == cityName) & (data['时间'] == day)].max()
        val = list(val)[-1]
        cityDic[day] = val
    dic[cityName] = cityDic
df = pd.DataFrame(dic,dtype=int)
df[df.isna()] = 0
with open("../data/cases.pickle",'wb') as f:
    pickle.dump(df,f)
df = df.transpose()
df.to_excel("cases.xlsx",encoding='utf-8')




