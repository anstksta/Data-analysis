# Step 1: Импорт пакетов, функций и классов
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
print('-------------Программа вычисления коэффициентов логистической регрессии-------------')
# Step 2:Получение данных
df = pd.read_excel ('Коэффициенты логистической регрессии.xlsx', sheet_name='Лист3')
X = df[['Выборка 1', 'Выборка 2']]
y = df['Признак']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Step 3:Вычисление матрицы ошибок
w = np.random.randn(X_train.shape[1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = sigmoid(X_test @ w)
y_pred_class = np.round(y_pred)

cm = confusion_matrix(y_test, y_pred_class)
print("Матрица ошибок:")
print(cm)

# Step 4:Рассчет метрик качества логистической регрессии
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
print("Метрики качетва")
print("Точность: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Полнота: {:.2f}%".format(recall*100))
print("F1 Score: {:.2f}%".format(f1*100))


# Step 4:Построение ROC-кривой
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



