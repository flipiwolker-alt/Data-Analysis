import numpy as np
import pandas as pd

with open('Москва_2021.txt', 'r') as f:
    data = np.array([int(line.strip()) for line in f if line.strip()])

n = len(data)
x_min, x_max = data.min(), data.max()
R = x_max - x_min

m = int(np.ceil(1 + 3.322 * np.log10(R)))
h = int(np.ceil(R / m))
bins = [x_min + i * h for i in range(m + 1)]

df = pd.DataFrame({'Возраст': data})
df['Группа'] = pd.cut(df['Возраст'], bins=bins, include_lowest=True, right=False)
n_in_groups = df['Группа'].notna().sum()
if n_in_groups != n:
    print(f"Внимание: {n - n_in_groups} значений вне интервалов (всего {n})")

groups = df.groupby('Группа')['Возраст']
group_stats = []
for name, grp in groups:
    vals = grp.values
    n_i = len(vals)
    x_mean = sum(vals) / n_i
    D_i = sum((x - x_mean)**2 for x in vals) / n_i
    group_stats.append({'Группа': name, 'count': n_i, 'mean': x_mean, 'var': D_i})
group_stats = pd.DataFrame(group_stats)

print("Таблица групповых характеристик:")
print(group_stats)
print("-" * 30)

x_avg_total = sum(data) / n
D_total = sum((x - x_avg_total)**2 for x in data) / n

D_vnutr = sum(row['count'] * row['var'] for _, row in group_stats.iterrows()) / n

D_mezh = sum(row['count'] * (row['mean'] - x_avg_total)**2 for _, row in group_stats.iterrows()) / n

print(f"Общая средняя (x_общ): {x_avg_total:.4f}")
print(f"Общая дисперсия (D_общ): {D_total:.4f}")
print(f"Внутригрупповая дисперсия (D_внутр): {D_vnutr:.4f}")
print(f"Межгрупповая дисперсия (D_меж): {D_mezh:.4f}")
print(f"Сумма (D_внутр + D_меж): {D_vnutr + D_mezh:.4f}")

if np.isclose(D_total, D_vnutr + D_mezh, rtol=1e-4):
    print("\nОК")
else:
    print("\nОшибка")