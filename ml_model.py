import pandas as pd
from sklearn.model_selection import train_test_split

# Завантаження даних
file_path = "data/household_power_consumption.txt"
df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])

# Перетворення дати та часу у формат datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.drop(columns=['Date', 'Time'], inplace=True)

# Конвертація числових колонок у float
numeric_cols = df.columns.difference(['Datetime'])
df[numeric_cols] = df[numeric_cols].astype(float)
df.dropna(inplace=True)

# Створення фіч (предикторів)
df['Hour'] = df['Datetime'].dt.hour  # Година доби
df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # День тижня (0 - понеділок, 6 - неділя)
df['Previous_Hour'] = df['Global_active_power'].shift(1)  # Споживання за попередню годину

# Видаляємо рядки з NaN (перший рядок буде пустим через shift)
df.dropna(inplace=True)

# Визначаємо X (фічі) та y (цільову змінну)
X = df[['Hour', 'DayOfWeek', 'Previous_Hour']]
y = df['Global_active_power']

# Розділяємо на train і test (80% тренувальні, 20% тестові)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Вивід розміру вибірок
print(f"Train set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Створюємо та навчаємо модель
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = model.predict(X_test)

# Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

# Вивід метрик
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

import matplotlib.pyplot as plt

# Візьмемо перші 100 точок для зручності
num_points = 100

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:num_points], label="Фактичне значення", marker='o', linestyle='-', color='blue')
plt.plot(y_pred[:num_points], label="Прогноз", marker='x', linestyle='--', color='red')

plt.xlabel("Часові інтервали")
plt.ylabel("Споживання енергії (кВт⋅год)")
plt.title("Порівняння фактичного та прогнозованого споживання енергії")
plt.legend()
plt.grid(True)
plt.show()

import joblib

# Збереження моделі
model_filename = "linear_regression_model.pkl"
joblib.dump(model, model_filename)

print(f"Модель збережено у {model_filename}")