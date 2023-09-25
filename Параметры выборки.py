import openpyxl
import pandas as pd
from math import log10, sqrt, ceil
from statistics import mode
import numpy as np
v = pd.read_excel('Параметры выборки.xlsx')
data= pd.DataFrame(v['Выборка 2'])
print ('Счет n (размер) =',v['Выборка 2'].count())
print ('Мин Xmin  =',v['Выборка 2'].min())
print ('Макс Xmax =',v['Выборка 2'].max())
print ('Сумма =',v['Выборка 2'].sum())
print ('Размах R =',v['Выборка 2'].max()-v['Выборка 2'].min())
m = 1 + 3.322 * log10(v['Выборка 2'].count())
print ('Кол-во интервалов m =', m)
print ('Оптимальная ширина интервала ∆X =', (v['Выборка 2'].max()-v['Выборка 2'].min())/m)
print ('Среднее значение =', v['Выборка 2'].mean())
print ('Математическое ожидание =', v['Выборка 2'].sum()/v['Выборка 2'].count())
print ('Медиана =', v['Выборка 2'].median())
print ('Мода =', v['Выборка 2'].mode())
print ('Дисперсия =', v['Выборка 2'].var())
print ('Стандартное отклонение =',v['Выборка 2'].std())

s = pd.Series(v['Выборка 2'], name='Границы интервалов и кол-во значений в них')
bins = list(np.arange(v['Выборка 2'].min(), v['Выборка 2'].max()+1, ((v['Выборка 2'].max()-v['Выборка 2'].min())/m)+0.00000000000001))
tab = s.groupby(pd.cut(s, bins=bins, right=False)).size()
print (tab)


znach = pd.DataFrame({'Значения интервалов': np.linspace(v['Выборка 2'].min(), v['Выборка 2'].max(), num = ceil(1 + 3.22 * log10(len(v['Выборка 2'])))+ 1)})
print(znach)

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.hist(v['Выборка 2'], bins = np.linspace(v['Выборка 2'].min(), v['Выборка 2'].max(), num = ceil(1 + 3.22 * log10(len(v['Выборка 2']))) + 1), color ='pink', edgecolor='brown')
ax.xaxis.set_major_locator(plt.IndexLocator(base = (v['Выборка 2'].max()-v['Выборка 2'].min())/m, offset = 0))
for i in ax.patches:
    ax.annotate('%1.0f'%(i.get_height()), (i.get_x()+2.8,i.get_height()+0.2))
ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
fig.set_figwidth(12)
fig.set_figheight(6)
plt.title('Выборка')
plt.xlabel('Интервалы')
plt.ylabel('Кол-во значений в интервалах')
plt.show()
