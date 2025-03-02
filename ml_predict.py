import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Завантаження моделі
loaded_model = joblib.load("linear_regression_model.pkl")

# Створення нового набору даних для прогнозу
new_data = pd.DataFrame([
    [8, 0, 1.2],  # Ранок у понеділок, попереднє споживання 1.2 кВт⋅год
    [14, 3, 2.1],  # День у четвер, попереднє споживання 2.1 кВт⋅год
    [19, 5, 1.8],  # Вечір у суботу, попереднє споживання 1.8 кВт⋅год
    [23, 6, 1.0],  # Пізній вечір у неділю, попереднє споживання 1.0 кВт⋅год
], columns=['Hour', 'DayOfWeek', 'Previous_Hour'])

# Прогноз
predictions = loaded_model.predict(new_data)

# Виведення прогнозів у вигляді списку
for i, pred in enumerate(predictions):
    print(f"Прогнозоване споживання енергії для набору {i+1}: {pred:.4f} кВт⋅год")

# Візуалізація прогнозу
plt.figure(figsize=(8, 5))
plt.plot(new_data["Hour"], predictions, marker='o', linestyle='-', color='b', label="Прогнозоване споживання")
plt.xlabel("Година доби")
plt.ylabel("Споживання енергії (кВт⋅год)")
plt.title("Прогнозоване споживання енергії")
plt.xticks(new_data["Hour"])  # Відображаємо лише ті години, які є в даних
plt.grid(True)
plt.legend()
plt.show()
