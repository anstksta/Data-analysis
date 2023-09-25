import pandas as pd
import numpy as np
import scipy.stats as st
from math import sqrt
from scipy.stats import t
import openpyxl
h = pd.read_excel ('Проверка гипотез.xlsx', sheet_name='Листок')
s1 = h['Выборка 1'].var()*(h['Выборка 1'].count()/(h['Выборка 1'].count()-1))
s2 = h['Выборка 2'].var()*(h['Выборка 2'].count()/(h['Выборка 2'].count()-1))
print ('Проверка гипотезы Фишера F')
print ('Несмещенная дисперсия первой выборки равна ', s1)
print ('Несмещенная дисперсия второй выборки равна ', s2)
if s1>s2:
    f1 = s1/s2
else:
    f1 = s2/s1
print ('F фактическая = ', f1)
print ('При alpha = 0,1:')
alpha = 0.1
f2 =  st.f.ppf(1-alpha/2, len(h['Выборка 1'])-1, len(h['Выборка 2'])-1)
print ('F критическая = ', f2)
if f1>f2 :
    print('Гипотеза Фишера согласуетя с выборкой, так как Fф>Fк(p=0,05)')
else:
    print('Гипотеза Фишера согласуетя с выборкой, так как Fф>Fк(p=0,05)')
print ('Проверка гипотезы о равенстве статистических средних значений (Критерий t-Стьюдента)')
mean1, mean2 = h['Выборка 1'].mean(), h['Выборка 2'].mean()
std1, std2 = h['Выборка 1'].std(ddof=1), h['Выборка 2'].std(ddof=1)
n1, n2 = h['Выборка 1'].count(), h['Выборка 2'].count()
se1, se2 = std1/sqrt(n1), std2/sqrt(n2)
sed = sqrt(se1**2.0 + se2**2.0)
t_stat = (mean1 - mean2) / sed
df = n1 + n2 - 2
alpha = 0.05
cv = t.ppf(1.0 - alpha, df)      
p = (1 - t.cdf(abs(t_stat), df)) * 2
print ('При alpha = 0.05')
print ('p =', p)
if p > alpha:
	print('Принимается нулевая гипотеза о том, что средние значения равны.')
else:
	print('Отвергается нулевая гипотеза о том, что средние значения равны.')
print ('Расчет критерия согласия-Пирсона')
print ('Для первой выборки:')
stat, p = st.normaltest(h['Выборка 1']) # Критерий согласия Пирсона
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принимается гипотеза о нормальности, так как p > alpha(0.05)')
else:
    print('Овергается гипотеза о нормальности, так как p < alpha(0.05)')
print ('Для второй выборки:')
stat, p = st.normaltest(h['Выборка 2']) # Критерий согласия Пирсона
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принимается гипотеза о нормальности, так как p > alpha(0.05)')
else:
    print('Овергается гипотеза о нормальности, так как p < alpha(0.05)')
