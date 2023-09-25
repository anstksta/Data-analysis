# Step 1: Импорт пакетов, функций и классов
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

# Step 2:Получение данных
df = pd.read_excel ('Коэффициенты логистической регрессии.xlsx', sheet_name='Лист3')
X = df[['Выборка 1', 'Выборка 2']]
y = df['Признак']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Step 3:Вычисление модели
w = np.random.randn(X_train.shape[1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, w):
    h = sigmoid(X @ w)
    J = -1/m * (y @ np.log(h) + (1-y) @ np.log(1-h))
    grad = 1/m * X.T @ (h-y)
    return J, grad

m = X_train.shape[0]
alpha = 0.01
num_iterations = 1000

J_history = []
w_history = []

for i in range(num_iterations):
    J, grad = cost_function(X_train, y_train, w)
    w = w - alpha * grad
    J_history.append(J)
    w_history.append(w)

print('Коэффициенты логистической регрессии: ', w)
