import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
df = pd.read_excel ('Индекс Джини.xlsx', sheet_name='Sheet1')
X = df[['Выборка 1', 'Выборка 2']]y = df['Признак']
# Разделение набора данных на обучающий и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
      
# Функция для выполнения тренировок с использованием индекса Джини.
def train_using_gini(X_train, X_test, y_train):
    # Создание объекта классификатора
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Проведение тренинга
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Функция для выполнения обучения с энтропией
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Дерево решений с энтропией
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Проведение тренинга
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
  
# Функция для составления прогнозов
def prediction(X_test, clf_object):
  
    # Прогноз по тесту с индексом Джини
    y_pred = clf_object.predict(X_test)
    print("Прогнозируемые значения:")
    print(y_pred)
    return y_pred
      
# Функция для расчета точности
def cal_accuracy(y_test, y_pred):
      
    print("Матрица путаницы: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Точность : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Отчет : ",
    classification_report(y_test, y_pred))
  
# Код драйвера
def main():
      
    # Этап строительства
    data = pd.read_excel ('Выборка ДР.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    # Этап "оперативная фаза"
    print("Результаты с использованием индекса Джини:")
      
    # Прогнозирование с использованием Джини
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
      
    print("Результаты с использованием энтропии:")
    # Прогнозирование с использованием энтропии
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
      
      
# Вызов основной функции
if __name__=="__main__":
    main()
