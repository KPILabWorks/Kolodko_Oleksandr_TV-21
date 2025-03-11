import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Завантажте дані про енергоспоживання
data = pd.read_csv('energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Вкажіть частоту даних (наприклад, 'H' для погодинних, 'D' для щоденних)
# Якщо ви не знаєте частоту, спробуйте її вивести:
# inferred_freq = pd.infer_freq(data.index)
# if inferred_freq:
#     data = data.asfreq(inferred_freq)
# else:
#     print("Не вдалося визначити частоту даних.  Вкажіть її явно, наприклад data = data.asfreq('H')")
#     exit() # або обробіть цю ситуацію інакше

# Приклад явного встановлення частоти (розкоментуйте, якщо потрібно)
data = data.asfreq('H')  # Замініть 'H' на фактичну частоту ваших даних

# Розділіть дані на навчальну та тестову вибірки
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]


# 1. Експоненційне згладжування (Holt-Winters)
def exponential_smoothing(train, test, seasonal_periods, trend='add', seasonal='add'):
    model = ExponentialSmoothing(train['energy_consumption'],
                                 seasonal_periods=seasonal_periods,
                                 trend=trend,
                                 seasonal=seasonal).fit()
    predictions = model.forecast(len(test))
    rmse = mean_squared_error(test['energy_consumption'], predictions, squared=False)
    mae = mean_absolute_error(test['energy_consumption'], predictions)
    return predictions, rmse, mae

# 2. ARIMA
def arima_forecast(train, test, order=(5,1,0)):  # Параметри p, d, q - потребують налаштування
    model = ARIMA(train['energy_consumption'], order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(len(test))
    rmse = mean_squared_error(test['energy_consumption'], predictions, squared=False)
    mae = mean_absolute_error(test['energy_consumption'], predictions)
    return predictions, rmse, mae

# 3. Facebook Prophet
def prophet_forecast(train, test):
    # Підготуйте дані для Prophet (потрібні колонки 'ds' (datetime) та 'y' (value))
    train_prophet = train.reset_index().rename(columns={'timestamp': 'ds', 'energy_consumption': 'y'})
    test_prophet = test.reset_index().rename(columns={'timestamp': 'ds', 'energy_consumption': 'y'})

    model = Prophet()
    model.fit(train_prophet)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    # Отримайте прогнози для тестової вибірки.  Увага до індексів!
    # Ми беремо зріз forecast, починаючи з кінця train_prophet.  Також важливо
    # щоб розмір зрізу відповідав розміру test.
    predictions = forecast['yhat'][len(train_prophet):len(train_prophet) + len(test)].values
    rmse = mean_squared_error(test['energy_consumption'], predictions, squared=False)
    mae = mean_absolute_error(test['energy_consumption'], predictions)
    return predictions, rmse, mae


# --- Виклик функцій та оцінка ---

# Параметри для експоненційного згладжування
seasonal_periods = 24  # Налаштуйте відповідно до сезонності ваших даних

# Експоненційне згладжування
exp_smooth_predictions, exp_smooth_rmse, exp_smooth_mae = exponential_smoothing(train, test, seasonal_periods)

# ARIMA
arima_predictions, arima_rmse, arima_mae = arima_forecast(train, test, order=(5,1,0)) # Налаштуйте порядок (p, d, q)

# Prophet
prophet_predictions, prophet_rmse, prophet_mae = prophet_forecast(train, test)


# Виведіть результати
print("Експоненційне згладжування (Holt-Winters):")
print(f"  RMSE: {exp_smooth_rmse:.3f}")
print(f"  MAE: {exp_smooth_mae:.3f}")

print("\nARIMA:")
print(f"  RMSE: {arima_rmse:.3f}")
print(f"  MAE: {arima_rmse:.3f}")

print("\nFacebook Prophet:")
print(f"  RMSE: {prophet_rmse:.3f}")
print(f"  MAE: {prophet_mae:.3f}")


# --- Візуалізація ---

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['energy_consumption'], label='Фактичні значення', color='blue')
plt.plot(test.index, exp_smooth_predictions, label='Експоненційне згладжування', color='green')
plt.plot(test.index, arima_predictions, label='ARIMA', color='red')
plt.plot(test.index, prophet_predictions, label='Prophet', color='purple')
plt.xlabel('Час')
plt.ylabel('Енергоспоживання')
plt.title('Прогнозування енергоспоживання')
plt.legend()
plt.show()