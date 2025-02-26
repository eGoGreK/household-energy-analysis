import pandas as pd
import matplotlib.pyplot as plt

# Завантаження оброблених даних
file_path = "data/household_power_consumption.txt"
df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])

# Перетворення дати та часу у формат datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.drop(columns=['Date', 'Time'], inplace=True)

# Конвертація числових колонок у float
numeric_cols = df.columns.difference(['Datetime'])
df[numeric_cols] = df[numeric_cols].astype(float)
df.dropna(inplace=True)

# Вибираємо дані за один тиждень (щоб графіки були читабельними)
df_sample = df[(df['Datetime'] >= '2007-02-01') & (df['Datetime'] < '2007-02-08')]

# Додаємо колонку з годиною
df_sample['Hour'] = df_sample['Datetime'].dt.hour

# Обчислюємо середнє споживання за кожну годину доби
hourly_avg = df_sample.groupby('Hour')['Global_active_power'].mean()

# Побудова графіка середнього споживання за годину
plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-', color='red', label='Hourly Avg Power (kW)')

# Налаштування осей
plt.xlabel('Hour of the Day')
plt.ylabel('Avg Power Consumption (kW)')
plt.title('Average Power Consumption by Hour of the Day')
plt.legend()
plt.grid(True)

# Відображення графіка
plt.show()

# Агрегуємо дані по днях
df_sample['Date'] = df_sample['Datetime'].dt.date
daily_avg = df_sample.groupby('Date')['Global_active_power'].mean()

# Побудова графіка середнього споживання за день
plt.figure(figsize=(10, 5))
plt.bar(daily_avg.index, daily_avg.values, color='green', label='Daily Avg Power (kW)')

# Налаштування осей
plt.xlabel('Date')
plt.ylabel('Avg Power Consumption (kW)')
plt.title('Average Power Consumption by Day')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Відображення графіка
plt.show()

# Побудова графіка споживання електроенергії
plt.figure(figsize=(12, 6))
plt.plot(df_sample['Datetime'], df_sample['Global_active_power'], label='Global Active Power (kW)', color='blue')

# Налаштування осей
plt.xlabel('Date')
plt.ylabel('Power Consumption (kW)')
plt.title('Household Energy Consumption Over Time')
plt.legend()
plt.xticks(rotation=45)

# Відображення графіка
plt.show()

# Побудова гістограми розподілу споживання енергії
plt.figure(figsize=(10, 5))
plt.hist(df_sample['Global_active_power'], bins=30, color='purple', edgecolor='black', alpha=0.7)

# Налаштування осей
plt.xlabel('Power Consumption (kW)')
plt.ylabel('Frequency')
plt.title('Distribution of Power Consumption')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Відображення графіка
plt.show()

# Визначення пікових годин (години з найбільшим середнім споживанням)
peak_hours = hourly_avg.sort_values(ascending=False).head(5)

# Вивід у консоль
print("Top 5 Peak Consumption Hours:")
print(peak_hours)

# Побудова графіка пікових годин
plt.figure(figsize=(8, 5))
plt.bar(peak_hours.index, peak_hours.values, color='orange', label='Peak Hours')

# Налаштування осей
plt.xlabel('Hour of the Day')
plt.ylabel('Avg Power Consumption (kW)')
plt.title('Top 5 Peak Power Consumption Hours')
plt.xticks(peak_hours.index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Відображення графіка
plt.show()

import seaborn as sns

# Видаляємо колонку Date, оскільки вона не є числовою
df_corr = df_sample.drop(columns=['Date'])

# Обчислення кореляційної матриці
correlation_matrix = df_corr.corr()

# Побудова heatmap (теплової карти кореляції)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Налаштування
plt.title('Correlation Matrix of Energy Consumption Variables')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Відображення графіка
plt.show()

# Визначаємо межі для аномалій за допомогою статистики
q1 = df_sample['Global_active_power'].quantile(0.25)  # Перший квартиль (25%)
q3 = df_sample['Global_active_power'].quantile(0.75)  # Третій квартиль (75%)
iqr = q3 - q1  # Міжквартильний розмах

# Визначаємо межі аномалій
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Відбираємо аномальні значення
anomalies = df_sample[(df_sample['Global_active_power'] < lower_bound) | 
                      (df_sample['Global_active_power'] > upper_bound)]

# Вивід у консоль
print(f"Number of anomalies detected: {len(anomalies)}")
print(anomalies[['Datetime', 'Global_active_power']].head())

# Візуалізація аномалій на графіку
plt.figure(figsize=(12, 6))
plt.plot(df_sample['Datetime'], df_sample['Global_active_power'], label='Normal Consumption', color='blue', alpha=0.5)
plt.scatter(anomalies['Datetime'], anomalies['Global_active_power'], color='red', label='Anomalies', marker='o')

# Налаштування графіка
plt.xlabel('Date')
plt.ylabel('Power Consumption (kW)')
plt.title('Anomalies in Power Consumption')
plt.legend()
plt.xticks(rotation=45)

# Відображення графіка
plt.show()

# Побудова графіка розсіяння для активної та реактивної потужності
plt.figure(figsize=(8, 6))
plt.scatter(df_sample['Global_active_power'], df_sample['Global_reactive_power'], alpha=0.5, color='purple')

# Налаштування графіка
plt.xlabel('Global Active Power (kW)')
plt.ylabel('Global Reactive Power (kW)')
plt.title('Relationship Between Active and Reactive Power')
plt.grid(True)

# Відображення графіка
plt.show()

# Створюємо ковзне середнє для прогнозування (7-годинне середнє)
df_sample['Moving_Avg'] = df_sample['Global_active_power'].rolling(window=7).mean()

# Побудова графіка прогнозу споживання
plt.figure(figsize=(12, 6))
plt.plot(df_sample['Datetime'], df_sample['Global_active_power'], label='Actual Consumption', color='blue', alpha=0.5)
plt.plot(df_sample['Datetime'], df_sample['Moving_Avg'], label='7-Hour Moving Avg', color='red', linestyle='dashed')

# Налаштування графіка
plt.xlabel('Date')
plt.ylabel('Power Consumption (kW)')
plt.title('Power Consumption Forecast Using Moving Average')
plt.legend()
plt.xticks(rotation=45)

# Відображення графіка
plt.show()