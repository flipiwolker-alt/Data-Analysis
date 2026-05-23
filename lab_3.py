import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

data = {
    'Год': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'y': [2404.8, 2302.2, 2206.2, 2190.6, 2388.5, 2160.1, 2058.5, 1991.5, 2024.3, 2044.2, 2004.4, 1966.8]
}
# делаем табличку
df_all = pd.DataFrame(data)
# скрываем 2022 год чтобы реализовать экстраполяцию
df = df_all[df_all['Год'] <= 2021].copy()
y0 = df['y'].iloc[0]

df['Абс_прирост_баз'] = df['y'] - y0
df['Темп_роста_баз_%'] = (df['y'] / y0) * 100
df['Темп_прироста_баз_%'] = df['Темп_роста_баз_%'] - 100

df['Абс_прирост_цеп'] = df['y'].diff()
df['Темп_роста_цеп_%'] = (df['y'] / df['y'].shift(1)) * 100
df['Темп_прироста_цеп_%'] = df['Темп_роста_цеп_%'] - 100

cols_temy = [
    'Год',
    'y',
    'Темп_роста_баз_%',
    'Темп_прироста_баз_%',
    'Темп_роста_цеп_%',
    'Темп_прироста_цеп_%',
]
df[cols_temy].to_csv(
    'temy_bazis_tsep.csv',
    index=False,
    float_format='%.4f',
    encoding='utf-8-sig',
)
_tbl = df[cols_temy].copy()
for _c in cols_temy[2:]:
    _tbl[_c] = _tbl[_c].map(lambda x: f'{x:.2f}' if pd.notna(x) else '—')
print(
    '\nБазисные темпы — к уровню первого года ряда (2011); '
    'цепные — к предыдущему году.\n'
    'Темп роста, % = (уровень / база)·100; темп прироста, % = темп роста − 100.\n'
)
print(_tbl.rename(columns={
    'Темп_роста_баз_%': 'Т_роста_баз_%',
    'Темп_прироста_баз_%': 'Т_прироста_баз_%',
    'Темп_роста_цеп_%': 'Т_роста_цеп_%',
    'Темп_прироста_цеп_%': 'Т_прироста_цеп_%',
}).to_string(index=False))
print()

n = len(df) - 1
yn = df['y'].iloc[-1]

y_avg = df['y'].mean()

delta_y_avg = (yn - y0) / n

k_avg = np.log(yn / y0) / n
t_p_avg = np.exp(k_avg) * 100

t_pr_avg = t_p_avg - 100

y_2022_abs = yn + delta_y_avg

y_2022_growth = yn * np.exp(k_avg)

df['t'] = range(1, len(df) + 1)
t_mean = df['t'].mean()

y_mean = df['y'].mean()
num_lin = (df['t'] * df['y']).sum() - len(df) * t_mean * y_mean
den_lin = (df['t'] ** 2).sum() - len(df) * (t_mean ** 2)
b_lin = num_lin / den_lin
a_lin = y_mean - b_lin * t_mean
y_2022_lin = a_lin + b_lin * 12

ln_y = np.log(df['y'])
ln_y_mean = ln_y.mean()
num_ln = (df['t'] * ln_y).sum() - len(df) * t_mean * ln_y_mean
den_ln = (df['t'] ** 2).sum() - len(df) * (t_mean ** 2)
beta_ln = num_ln / den_ln
alpha_ln = ln_y_mean - beta_ln * t_mean
A_exp = np.exp(alpha_ln)
y_2022_exp = A_exp * np.exp(beta_ln * 12)

y_fact = df_all[df_all['Год'] == 2022]['y'].values[0]

def get_error(y_pred, y_actual):
    return abs(y_pred - y_actual) / y_actual * 100 

err_abs = get_error(y_2022_abs, y_fact)
err_growth = get_error(y_2022_growth, y_fact)
err_lin = get_error(y_2022_lin, y_fact)
err_exp = get_error(y_2022_exp, y_fact)

print(f"Средний уровень за 2011-2021 гг.: {y_avg:.2f}\n")

print("Средние значения темпов (ряд 2011–2021, n =", n, "интервалов):")
print(f"  Средний абсолютный прирост: {delta_y_avg:.4f}")
print(f"  Средний темп роста: {t_p_avg:.4f} %")
print(f"  Средний темп прироста: {t_pr_avg:.4f} %")
print(
    f"  Среднее цепного темпа роста (арифм. по годам): "
    f"{df['Темп_роста_цеп_%'].mean():.4f} %"
)
print(
    f"  Среднее цепного темпа прироста (арифм. по годам): "
    f"{df['Темп_прироста_цеп_%'].mean():.4f} %\n"
)
print(f"Фактическое значение (2022): {y_fact:.2f}\n")
print(f"Прогноз (Абс. прирост): {y_2022_abs:.2f}, Ошибка: {err_abs:.2f}%")
print(f"Прогноз (Темп роста, exp): {y_2022_growth:.2f}, Ошибка: {err_growth:.2f}%")
print(f"Прогноз (прямая, МНК): y = {a_lin:.2f} {b_lin:+.4f}·t → {y_2022_lin:.2f}, Ошибка: {err_lin:.2f}%")
print(f"Прогноз (экспонента, МНК по ln y): {y_2022_exp:.2f}, Ошибка: {err_exp:.2f}%\n")

errors_dict = {
    "по среднему абсолютному приросту": err_abs,
    "по среднему темпу роста (экспонента)": err_growth,
    "методом выравнивания прямой (МНК)": err_lin,
    "методом выравнивания экспонентой (МНК по ln y)": err_exp,
}

best_method = min(errors_dict, key=errors_dict.get)
min_error = errors_dict[best_method]

print("-" * 50)
print("ВЫВОД О ТОЧНОСТИ ПРОГНОЗА (все методы):")
print(f"Сопоставив расчеты с фактическим значением за 2022 год, можно сделать вывод, \n"
      f"что наиболее точным оказался прогноз {best_method} \n"
      f"с минимальной относительной погрешностью {min_error:.2f}%.")

years_ext = np.arange(2011, 2023)
t_ext = np.arange(1, len(years_ext) + 1)
y_trend_lin = a_lin + b_lin * t_ext
y_trend_exp = A_exp * np.exp(beta_ln * t_ext)

fig1, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(df_all["Год"], df_all["y"], "o-", color="#2c3e50", linewidth=2, markersize=7, label="Фактический ряд")
ax1.plot(
    years_ext,
    y_trend_lin,
    "--",
    color="#2980b9",
    linewidth=2,
    label=rf"Прямая: $y = {a_lin:.1f} {b_lin:+.3f}t$",
)
ax1.plot(
    years_ext,
    y_trend_exp,
    "--",
    color="#e74c3c",
    linewidth=2,
    label=rf"Экспонента: $y = {A_exp:.1f}\,e^{{{beta_ln:.4f}t}}$",
)
ax1.scatter(
    [2022, 2022, 2022, 2022, 2022],
    [y_2022_abs, y_2022_growth, y_2022_lin, y_2022_exp, y_fact],
    s=[70, 70, 80, 80, 120],
    c=["#3498db", "#9b59b6", "#2980b9", "#1abc9c", "#27ae60"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax1.scatter([], [], c="#3498db", s=70, label="Прогноз: абс. прирост")
ax1.scatter([], [], c="#9b59b6", s=70, label="Прогноз: темп роста")
ax1.scatter([], [], c="#2980b9", s=80, label="Прогноз: прямая МНК")
ax1.scatter([], [], c="#1abc9c", s=80, label="Прогноз: эксп. МНК")
ax1.scatter([], [], c="#27ae60", s=120, label="Факт 2022")
anno_2022 = [
    (y_2022_exp, f"эксп. {y_2022_exp:.0f}", (10, -16)),
    (y_2022_lin, f"лин. {y_2022_lin:.0f}", (-48, -6)),
    (y_2022_abs, f"абс. {y_2022_abs:.0f}", (8, 4)),
    (y_fact, f"факт {y_fact:.0f}", (-44, 12)),
    (y_2022_growth, f"темп {y_2022_growth:.0f}", (8, 14)),
]
for y_val, txt, off in sorted(anno_2022, key=lambda z: z[0]):
    ax1.annotate(txt, (2022, y_val), textcoords="offset points", xytext=off, fontsize=8)
ax1.set_xlabel("Год")
ax1.set_ylabel("y")
ax1.set_title("Динамика показателя и прогнозы на 2022 г.")
ax1.legend(loc="best", fontsize=8)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph_1_ryad_prognoz.png", dpi=150)
plt.show()
