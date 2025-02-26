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