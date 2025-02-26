import pandas as pd

# Завантаження даних
file_path = "data/household_power_consumption.txt"
df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])

# Перетворення дати та часу у формат datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Видалення старих колонок Date і Time
df.drop(columns=['Date', 'Time'], inplace=True)

# Конвертація числових колонок у float
numeric_cols = df.columns.difference(['Datetime'])
df[numeric_cols] = df[numeric_cols].astype(float)

# Видалення пропущених значень
df.dropna(inplace=True)

# Вивід результатів
print(df.info())
print(df.head())