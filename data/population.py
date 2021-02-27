# import bs4
# import pickle
# File = open('data.html')
# Soup = bs4.BeautifulSoup(File.read(),'html5lib')
# elems = Soup.select('.info_table')
# tags = elems[0].select('tr')
# population = dict()
# for tag in tags:
#     diji = tag.select('[name="diJi"]')
#     renko = tag.select('[name="renKou"]')
#     if len(diji)>= 1:
#         name = diji[0].get_text()
#         num = renko[0].get_text()
#         if len(name)*len(num) != 0 :
#             population[name] = float(num)*10000
#         print(name,num)
# with open("population.pickle",'wb') as f:
#     pickle.dump(population,f)
# f.close()
#
#
import numpy as np
import json
f = open("../data/cityPopulation.json",'r',encoding='utf-8')
population = json.load(f)
