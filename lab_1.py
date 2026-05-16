import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

with open('Москва_2021.txt', 'r') as f:
    data = [int(line.strip()) for line in f if line.strip()]

n = len(data)
x_min = min(data)
x_max = max(data)
R = x_max - x_min

m = int(np.ceil(1 + 3.322 * np.log10(R)))
h = int(np.ceil(R / m))

print(f"Число групп (m): {m}, Шаг интервала (h): {h}")
print()

disc = pd.Series(data).value_counts().sort_index().reset_index()
disc.columns = ['Возраст', 'Частота']

bins = np.array([x_min + i * h for i in range(m + 1)])
interval_counts, _ = np.histogram(data, bins=bins)
intervals = [f"[{int(bins[i])} - {int(bins[i+1])})" for i in range(len(bins)-1)]
interval_series = pd.DataFrame({'Интервал': intervals, 'Частота': interval_counts})

print("Дискретный ряд:")
print(disc.to_string(index=False))
print()

print("Интервальный ряд:")
print(interval_series.to_string(index=False))
print()

def calculate_stats(data_array):
    mean = np.mean(data_array)
    variance = np.var(data_array)
    std_dev = np.sqrt(variance)
    variation = (std_dev / mean) * 100
    mode_res = stats.mode(data_array, keepdims=True)
    mode = mode_res.mode[0]
    mode_freq = mode_res.count[0]
    median = np.median(data_array)
    range_val = np.ptp(data_array)
    return mean, variance, std_dev, variation, mode, mode_freq, median, range_val

stats_discrete = calculate_stats(data)
mean, variance, std_dev, variation, mode, mode_freq, median, range_val = stats_discrete

print("Статистические характеристики (дискретный ряд):")
print(f"  Среднее: {mean:.2f}")
print(f"  Дисперсия: {variance:.2f}")
print(f"  Среднеквадратичное отклонение: {std_dev:.2f}")
print(f"  Коэффициент вариации: {variation:.2f}%")
print(f"  Мода: {mode} (встречается {mode_freq} раз)")
print(f"  Медиана: {median:.2f}")
print(f"  Размах: {range_val}")
print()

midpoints = (bins[:-1] + bins[1:]) / 2
weighted_mean = np.average(midpoints, weights=interval_counts)
weighted_var = np.average((midpoints - weighted_mean)**2, weights=interval_counts)
weighted_std = np.sqrt(weighted_var)
weighted_variation = (weighted_std / weighted_mean) * 100

modal_idx = np.argmax(interval_counts)
modal_interval = intervals[modal_idx]
f_modal = interval_counts[modal_idx]
f_prev = interval_counts[modal_idx - 1] if modal_idx > 0 else 0
f_next = interval_counts[modal_idx + 1] if modal_idx < len(interval_counts) - 1 else 0
x_modal = bins[modal_idx]
mode_interval = x_modal + h * (f_modal - f_prev) / (2 * f_modal - f_prev - f_next) if (2 * f_modal - f_prev - f_next) != 0 else x_modal + h / 2

n_half = n / 2
cumsum = np.cumsum(interval_counts)
med_idx = np.searchsorted(cumsum, n_half)
med_idx = min(med_idx, len(interval_counts) - 1)
S_prev = cumsum[med_idx - 1] if med_idx > 0 else 0
x_med = bins[med_idx]
f_med = interval_counts[med_idx]
median_interval = x_med + h * (n_half - S_prev) / f_med

print("Статистические характеристики (интервальный ряд):")
print(f"  Среднее: {weighted_mean:.2f}")
print(f"  Дисперсия: {weighted_var:.2f}")
print(f"  Среднеквадратичное отклонение: {weighted_std:.2f}")
print(f"  Коэффициент вариации: {weighted_variation:.2f}%")
print(f"  Мода: {mode_interval:.2f} (модальный интервал {modal_interval})")
print(f"  Медиана: {median_interval:.2f}")
R_interval = int(bins[-1] - bins[0])
print(f"  Размах: {R_interval}")
print()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(disc['Возраст'], disc['Частота'], marker='o', linestyle='-')
plt.title('Полигон частот (Дискретный ряд)')
plt.xlabel('Возраст')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
plt.bar(bins[:-1], interval_counts, width=h, align='edge', color='skyblue', edgecolor='black')
plt.xticks(bins[:-1] + h/2, intervals, rotation=45)
plt.title('Гистограмма частот (Интервальный ряд)')
plt.xlabel('Интервалы возраста')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()