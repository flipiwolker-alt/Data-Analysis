import numpy as np
import matplotlib.pyplot as plt

# 1. Исходные данные (Таблица 1: Количество преступлений в РФ)
# Данные за 2019, 2020 и 2021 годы по месяцам (в тысячах)
data_2019 = np.array([154.7, 159.6, 176.6, 176.0, 168.8, 175.8, 171.8, 172.1, 166.2, 199.0, 148.1, 155.5])
data_2020 = np.array([159.2, 168.1, 183.3, 161.9, 159.9, 178.7, 178.4, 177.9, 172.8, 199.5, 150.8, 153.7])
data_2021 = np.array([148.7, 160.7, 192.2, 178.2, 165.9, 176.4, 166.1, 168.2, 165.2, 184.3, 147.3, 151.3])

# Объединяем в один временной ряд (36 месяцев)
y_true = np.concatenate((data_2019, data_2020, data_2021))
n_total = len(y_true)
t_full = np.arange(1, n_total + 1)

print("Исходный временной ряд y (36 месяцев, тыс. преступлений):")
print(np.array2string(y_true, precision=1, separator=", ", max_line_width=120))
print()

# 2. Вычисление коэффициентов ряда Фурье
# Находим средние значения для каждого из 12 месяцев (период T = 12)
y_avg = (data_2019 + data_2020 + data_2021) / 3
T = 12
t_period = np.arange(1, T + 1)

# a_0 / 2 (среднее значение за период)
a0_half = np.mean(y_avg)

# Вычисление коэффициентов a_k и b_k до 4-го порядка (k=4)
k_max = 4
a = np.zeros(k_max + 1)
b = np.zeros(k_max + 1)

for k in range(1, k_max + 1):
    # Формулы из лекции для дискретного ряда
    a[k] = (2 / T) * np.sum(y_avg * np.cos(2 * np.pi * k * t_period / T))
    b[k] = (2 / T) * np.sum(y_avg * np.sin(2 * np.pi * k * t_period / T))

print("Коэффициенты ряда Фурье:")
print(f"a0/2 = {a0_half:.4f}")
for k in range(1, k_max + 1):
    print(f"k={k}: a_{k} = {a[k]:.4f}, b_{k} = {b[k]:.4f}")

print("\nРяд Фурье (T = 12 — период в месяцах; t — номер месяца в сквозной нумерации):")
print("ŷ(t) = a₀/2 + Σ_{k=1}^{4} [ a_k·cos(2πkt/T) + b_k·sin(2πkt/T) ]")
print("\nС подставленными коэффициентами:")
_terms = [f"{a0_half:.4f}"]
for _k in range(1, k_max + 1):
    _terms.append(f"{a[_k]:+.4f}*cos(2π*{_k}*t/{T})")
    _terms.append(f"{b[_k]:+.4f}*sin(2π*{_k}*t/{T})")
print("ŷ(t) = " + " ".join(_terms).replace("+ -", "- "))
print()

# 3. Моделирование значений для всего ряда (36 месяцев)
y_model = np.zeros(n_total)
for t in range(n_total):
    time_point = t_full[t]
    sum_fourier = 0
    for k in range(1, k_max + 1):
        sum_fourier += a[k] * np.cos(2 * np.pi * k * time_point / T) + \
                       b[k] * np.sin(2 * np.pi * k * time_point / T)
    y_model[t] = a0_half + sum_fourier

print("Модельные значения ŷ(t) по ряду Фурье (k=4, 36 месяцев, тыс.):")
print(np.array2string(y_model, precision=4, separator=", ", max_line_width=120))
print()

# 4. Оценка качества модели (Коэффициент детерминации R^2)
ss_res = np.sum((y_true - y_model)**2) # Сумма квадратов остатков
ss_tot = np.sum((y_true - np.mean(y_true))**2) # Общая сумма квадратов
r2 = 1 - (ss_res / ss_tot)

print(f"\nКоэффициент детерминации R^2: {r2:.4f}")

# 5. Построение графика
plt.figure(figsize=(12, 6))
plt.plot(t_full, y_true, marker='o', label='Фактические данные', color='blue', linewidth=2)
plt.plot(t_full, y_model, marker='x', label='Модель (Ряд Фурье, k=4)', color='red', linestyle='--', linewidth=2)

plt.title('Моделирование количества зарегистрированных преступлений в РФ (2019-2021)')
plt.xlabel('Месяц (от начала 2019 года)')
plt.ylabel('Количество преступлений (тыс.)')
plt.xticks(np.arange(1, 37, step=1))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()

# Отображение графика
plt.show()