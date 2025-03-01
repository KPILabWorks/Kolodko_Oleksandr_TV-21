import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from functools import reduce


# Функція для обчислення добутку елементів списку
def multiply(x, y):
    return x * y


# Створюємо список чисел
numbers = [2, 3, 5, 7, 11]
product = reduce(multiply, numbers)
print(f"Добуток елементів списку: {product}")

# Генеруємо часовий ряд даних про споживання енергії
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
data = np.random.randint(50, 150, size=len(dates))
df = pd.DataFrame({'Дата': dates, 'Споживання': data})

# Перетворення щоденних даних на тижневі середні значення
df.set_index('Дата', inplace=True)
df_weekly = df.resample('W').mean()

# Вивід результату
print("Перші 5 рядків агрегованих даних:")
print(df_weekly.head())



# Функція для прогнозування та оцінки точності
def evaluate_forecast(df, label):
    df['Лаг_1'] = df['Споживання'].shift(1)
    df.dropna(inplace=True)

    X = df[['Лаг_1']]
    y = df['Споживання']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Середня абсолютна помилка для {label}: {error}")


# Оцінка точності для оригінальних і агрегованих даних
evaluate_forecast(df, "щоденних даних")
evaluate_forecast(df_weekly, "тижневих середніх значень")