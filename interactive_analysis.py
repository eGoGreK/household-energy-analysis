import pandas as pd
import plotly.express as px

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

# Вибірка даних за один тиждень для зручного перегляду
df_sample = df[(df['Datetime'] >= '2007-02-01') & (df['Datetime'] < '2007-02-08')].copy()

# Інтерактивний графік
fig = px.line(df_sample, x='Datetime', y='Global_active_power', title='Interactive Power Consumption Over Time')

# Відображення графіка
fig.show()

# Додаємо колонку з годиною
df_sample['Hour'] = df_sample['Datetime'].dt.hour

# Обчислюємо середнє споживання за кожну годину доби
hourly_avg = df_sample.groupby('Hour')['Global_active_power'].mean().reset_index()

# Інтерактивний барчарт пікових годин
fig2 = px.bar(hourly_avg, x='Hour', y='Global_active_power', 
              title='Average Power Consumption by Hour',
              labels={'Global_active_power': 'Avg Power Consumption (kW)', 'Hour': 'Hour of the Day'},
              color='Global_active_power', color_continuous_scale='viridis')

# Відображення графіка
fig2.show()

import plotly.figure_factory as ff

# Обчислення кореляційної матриці
df_corr = df_sample.drop(columns=['Datetime'])
correlation_matrix = df_corr.corr()

# Інтерактивна теплова карта (heatmap)
fig3 = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    colorscale='Viridis',
    showscale=True
)

# Налаштування заголовка
fig3.update_layout(title='Interactive Correlation Heatmap')

# Відображення графіка
fig3.show()