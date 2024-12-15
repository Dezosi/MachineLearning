import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('p1_diamonds.csv')

print(df.head(10).to_string())

#Удаление Unnamed столбца. axis = 1, удаление столбца 1.
df = df.drop(['Unnamed: 0'], axis = 1)

#Создание переменных для категорий
categorial_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

#Замена категорий на численные значения
for i in range(3):
    new = le.fit_transform(df[categorial_features[i]])
    df[categorial_features[i]] = new

#print(df.head(10).to_string())

x = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = df[['price']]

#Разделение данных на тренировочный и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 25, random_state = 150)

#Тренировка
regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 150)
regr.fit(x_train, y_train.values.ravel())

#Прогнозирование
predictions = regr.predict(x_test)

result = x_test
result['preice'] = y_test
result['prediction'] = predictions.tolist()

print(result.to_string())

#Определение оси X
x_axis = x_test.carat

#Построение графиков
plt.scatter(x_axis, y_test, c = 'b', alpha =0.5, marker = '.', label = 'Real')
plt.scatter(x_axis, predictions, c = 'r', alpha =0.5, marker = '.', label = 'Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color = '#D3D3D3', linestyle = 'solid')
plt.legend(loc = 'lower right')
plt.show()







