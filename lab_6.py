import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Исходные данные (Таблица 1)
# ==========================================
data = {
    '2019': [154.7, 159.6, 176.6, 176.0, 168.8, 175.8, 171.8, 172.1, 166.2, 199.0, 148.1, 155.5],
    '2020': [159.2, 168.1, 183.3, 161.9, 159.9, 178.7, 178.4, 177.9, 172.8, 199.5, 150.8, 153.7],
    '2021': [148.7, 160.7, 192.2, 178.2, 165.9, 176.4, 166.1, 168.2, 165.2, 184.3, 147.3, 151.3],
    '2022': [149.5, 153.4, 179.7, 170.7, 169.2, 181.3, 163.5, 168.5, 162.4, 179.1, 146.3, 143.4]
}

months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
df = pd.DataFrame(data, index=months)

# Формируем обучающую выборку (2019-2021) и тестовую (2022)
train_series = np.concatenate([df['2019'].values, df['2020'].values, df['2021'].values])
test_series = df['2022'].values

# ==========================================
# 1. Автокорреляционная функция (АКФ) вручную
# ==========================================
# Реализация по формулам (2.5) и (2.6) из лекции
def calculate_acf(y, max_lag):
    acf_vals = []
    for tau in range(max_lag + 1):
        if tau == 0:
            acf_vals.append(1.0)
        else:
            # Смещаем ряды на шаг tau
            y_t = y[tau:]
            y_t_minus_tau = y[:-tau]
            
            # Средние значения для смещенных рядов
            mean_y_t = np.mean(y_t)
            mean_y_t_minus_tau = np.mean(y_t_minus_tau)
            
            # Вычисление коэффициента Пирсона (формула 2.5)
            numerator = np.sum((y_t - mean_y_t) * (y_t_minus_tau - mean_y_t_minus_tau))
            denominator = np.sqrt(np.sum((y_t - mean_y_t)**2) * np.sum((y_t_minus_tau - mean_y_t_minus_tau)**2))
            
            r_tau = numerator / denominator if denominator != 0 else 0
            acf_vals.append(r_tau)
    return np.array(acf_vals)

lag_max = 12
acf_values = calculate_acf(train_series, lag_max)

print("--- 1. Коэффициенты автокорреляции ---")
for i, val in enumerate(acf_values):
    print(f"Лаг {i}: {val:.4f}")

# Построение коррелограммы (визуализация АКФ)
plt.figure(figsize=(10, 5))
plt.bar(range(lag_max + 1), acf_values, width=0.1, color='blue')
plt.scatter(range(lag_max + 1), acf_values, color='blue')
plt.axhline(0, color='black', linewidth=1)
plt.title("Коррелограмма (АКФ)")
plt.xlabel("Лаг $\\tau$ (месяцы)")
plt.ylabel("Коэффициент $r(\\tau)$")
plt.xticks(range(lag_max + 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# 2. Выбор лага и построение модели вручную
# ==========================================
# Ищем лаг с максимальной автокорреляцией (пропуская нулевой)
tau = np.argmax(np.abs(acf_values[1:])) + 1
print(f"\nНаибольшая автокорреляция достигается при лаге τ = {tau}")

# Подготовка данных для модели: y_t зависит от y_{t-tau}
y_train = train_series[tau:]
x_train = train_series[:-tau]

# Метод наименьших квадратов (МНК) для парной линейной регрессии
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

# Вычисление параметров theta_1 и theta_0
theta_1 = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean)**2)
theta_0 = y_mean - theta_1 * x_mean

print("\n--- 2. Парная авторегрессионная модель ---")
print(f"Уравнение модели: y_t = {theta_0:.4f} + {theta_1:.4f} * y_{{t-{tau}}}")

# ==========================================
# 4. Прогноз на 2022 год и оценка точности
# ==========================================
# Данные за 2021 год (последние 12 месяцев обучающей выборки) выступают как y_{t-tau} для прогноза на 2022
x_test = train_series[-tau:] 
y_pred = theta_0 + theta_1 * x_test

# Качество прогноза: R^2 = r^2 между фактическим рядом за 2022 и прогнозом.
r_fact_pred = np.corrcoef(test_series, y_pred)[0, 1]
r2 = r_fact_pred**2

print("\n--- 4. Прогнозные оценки и R^2 ---")
print("Прогноз на 2022 год:")
for m, val in zip(months, y_pred):
    print(f"{m}: {val:.1f}")
print(f"\nКоэффициент детерминации (R^2 = r^2 факт–прогноз за 2022): {r2:.4f}")

# ==========================================
# 5. График прогнозных и фактических данных
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(months, test_series, marker='o', color='blue', label='Фактические данные (2022)')
plt.plot(months, y_pred, marker='x', linestyle='--', color='red', label='Прогнозная модель')
plt.title("Сравнение прогнозных и фактических данных за 2022 год")
plt.xlabel("Месяц")
plt.ylabel("Количество зарегистрированных преступлений")
plt.legend()
plt.grid(True)
plt.show()