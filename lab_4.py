import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    '2019': [154.7, 159.6, 176.6, 176.0, 168.8, 175.8, 171.8, 172.1, 166.2, 199.0, 148.1, 155.5],
    '2020': [159.2, 168.1, 183.3, 161.9, 159.9, 178.7, 178.4, 177.9, 172.8, 199.5, 150.8, 153.7],
    '2021': [148.7, 160.7, 192.2, 178.2, 165.9, 176.4, 166.1, 168.2, 165.2, 184.3, 147.3, 151.3]
}
months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

Y = np.concatenate([data['2019'], data['2020'], data['2021']])
n = 12
h = 3
t = np.arange(1, n * h + 1)

y_mean_total = Y.mean()

Y_matrix = Y.reshape((h, n))
y_mean_month = Y_matrix.mean(axis=0)
I_s = (y_mean_month / y_mean_total) * 100.0

# СС порядка m = 3 по лекции (пример 2.1): внутри каждого года отдельно
# ŷ₂…ŷ₁₁ — средние трёх соседних месяцев;
# ŷ₁ = ŷ₂ − (ŷ₁₁ − ŷ₂)/9,  ŷ₁₂ = ŷ₁₁ + (ŷ₁₁ − ŷ₂)/9 (линейные формулы, с. 12–13)
ma_window = 3


def moving_average_lecture_m3(year_levels: np.ndarray) -> np.ndarray:
    """year_levels — 12 уровней одного года (январь … декабрь)."""
    b = np.asarray(year_levels, dtype=float)
    yhat = np.zeros(12)
    for m in range(1, 11):  # месяцы 2…11 (индексы 1…10): ŷ = (y_{m-1}+y_m+y_{m+1})/3
        yhat[m] = (b[m - 1] + b[m] + b[m + 1]) / 3.0
    yhat[0] = yhat[1] - (yhat[10] - yhat[1]) / 9.0
    yhat[11] = yhat[10] + (yhat[10] - yhat[1]) / 9.0
    return yhat


Y_matrix_rows = Y.reshape((h, n))
Y_ma = np.concatenate([moving_average_lecture_m3(Y_matrix_rows[j, :]) for j in range(h)])

s_add_matrix = (Y - Y_ma).reshape((h, n))
s_add_mean = s_add_matrix.mean(axis=0)
c_add = s_add_mean.sum() / n
S_add = s_add_mean - c_add
S_add_full = np.tile(S_add, h)

Y_deseason_add = Y - S_add_full
model_add = LinearRegression().fit(t.reshape(-1, 1), Y_deseason_add)
T_add = model_add.predict(t.reshape(-1, 1))
Y_model_add = T_add + S_add_full

s_mult_matrix = (Y / Y_ma).reshape((h, n))
s_mult_mean = s_mult_matrix.mean(axis=0)
c_mult = s_mult_mean.sum() / n
S_mult = s_mult_mean / c_mult
S_mult_full = np.tile(S_mult, h)

Y_deseason_mult = Y / S_mult_full
model_mult = LinearRegression().fit(t.reshape(-1, 1), Y_deseason_mult)
T_mult = model_mult.predict(t.reshape(-1, 1))
Y_model_mult = T_mult * S_mult_full

b0, b1 = float(model_add.intercept_), float(model_add.coef_[0])
c0, c1 = float(model_mult.intercept_), float(model_mult.coef_[0])

print("\n--- 0. Индексы сезонности (по средним месяцам: I_j = ȳ_j / ȳ · 100%) ---")
print("ȳ_j — среднее по j-му месяцу за 2019–2021 гг., ȳ — общее среднее ряда.")
for m, idx in zip(months, I_s):
    print(f"  {m}: {idx:.2f}%")

print("\n--- 1. Уравнения моделей ---")
print("t — порядковый номер месяца (1 … 36); j — месяц года (1=январь … 12=декабрь);")
print("S_j и k_j — сезонные компоненты (см. раздел 2).")
print()
print("Аддитивная модель:  Y_t = T_t + S_j,")
print(f"  тренд по десезонализированному ряду (Y - S):  T_t = {b0:.6f} + ({b1:.6f})*t")
print()
print("Мультипликативная модель:  Y_t = T_t * k_j,")
print(f"  тренд по ряду Y/k_j:  T_t = {c0:.6f} + ({c1:.6f})*t")

w_ma = 1.0 / ma_window
print("\n--- 1а. Скользящая средняя (для сезонных коэффициентов в п. 2) ---")
print(
    f"Центрированная простая СС порядка m = {ma_window}: "
    f"для внутренних t значение — среднее m соседних уровней Y."
)
print(f"Веса при равных долях: w_i = 1/m = 1/{ma_window} = {w_ma:.4f}.")
print("Края года: линейная экстраполяция по ŷ₂ и ŷ₁₁ (лекция, с. 12–13).")
print(f"\nРяд Ŷ_ma (и Y_t), тыс.:")
print(f"{'t':>4} {'Месяц':12} {'Y_t':>10} {'Ŷ_ma':>10}")
for it in range(len(Y)):
    tt = int(t[it])
    month_idx = (tt - 1) % n
    print(f"{tt:4d} {months[month_idx]:12} {Y[it]:10.4f} {Y_ma[it]:10.4f}")

print("\n--- 2. Сезонные компоненты (по СС с m = 3, как в модели выше) ---")
print("Аддитивная модель: S_j — отклонения, тыс. (сумма S_j по году = 0):")
for m, s in zip(months, S_add):
    print(f"  {m}: {s:+.4f}")
print(f"  Проверка: сумма S = {S_add.sum():.2e}")

print("\nМультипликативная модель: k_j — коэффициенты сезонности (среднее k = 1):")
for m, k in zip(months, S_mult):
    print(f"  {m}: {k:.6f}")
print(f"  Проверка: среднее k = {S_mult.mean():.6f}")

delta = Y - y_mean_total
ss_tot = np.sum(delta ** 2)
R2_add = 1 - np.sum((Y - Y_model_add) ** 2) / ss_tot
R2_mult = 1 - np.sum((Y - Y_model_mult) ** 2) / ss_tot

print(f"\n--- 3. Качество моделей (коэффициент детерминации) ---")
print(f"R^2 (Аддитивная): {R2_add:.4f}  (округлённо: {round(R2_add, 2):.2f})")
print(f"R^2 (Мультипликативная): {R2_mult:.4f}  (округлённо: {round(R2_mult, 2):.2f})")

# Стандартная ошибка тренда T(t) в точке t (десезонализированные ряды, линейная регрессия)
n_obs = len(t)
t_mean = float(t.mean())
Sxx = float(np.sum((t - t_mean) ** 2))
res_add = Y_deseason_add - T_add
res_mult = Y_deseason_mult - T_mult
mse_add = float(np.sum(res_add ** 2) / (n_obs - 2))
mse_mult = float(np.sum(res_mult ** 2) / (n_obs - 2))


def se_trend(mse: float, t_new: float) -> float:
    return float(np.sqrt(mse * (1.0 / n_obs + (t_new - t_mean) ** 2 / Sxx)))


z95 = 1.96
forecasts_rows = []
for j in range(12):
    t_new = float(n * h + 1 + j)
    T_add_new = float(model_add.predict([[t_new]])[0])
    T_mult_new = float(model_mult.predict([[t_new]])[0])
    margin_add = z95 * se_trend(mse_add, t_new)
    margin_mult = z95 * se_trend(mse_mult, t_new) * S_mult[j]
    forecasts_rows.append(
        {
            "label": f"{months[j]} 2022",
            "t": int(t_new),
            "forecast_add": T_add_new + S_add[j],
            "margin_add": margin_add,
            "forecast_mult": T_mult_new * S_mult[j],
            "margin_mult": margin_mult,
        }
    )

# Март 2022: после 36 месяцев (2019–2021) — t = 37 (янв.), 38 (фев.), 39 (март)
t_march_2022 = n * h + 3
fact_march = 179.7

T_march_add = model_add.predict([[t_march_2022]])[0]
T_march_mult = model_mult.predict([[t_march_2022]])[0]
forecast_add = T_march_add + S_add[2]
forecast_mult = T_march_mult * S_mult[2]

# Относительная погрешность к факту, как в примере 4.1 (лекция): δ = 100·|y − ŷ| / y
y_fact = fact_march
abs_err_add = abs(y_fact - forecast_add)
abs_err_mult = abs(y_fact - forecast_mult)
error_add_pct = 100.0 * abs_err_add / y_fact
error_mult_pct = 100.0 * abs_err_mult / y_fact


def print_forecasts_table(rows, fact_y: float) -> None:
    """Печать прогнозов ŷ, полуширины ± для тренда и погрешностей к факту (март)."""
    print("\n--- 4. Прогнозы (2022): значения и погрешности (кодом в консоль) ---")
    print(
        "ŷ — точечный прогноз (тыс.); ± — 95%-й полуинтервал для T(t): 1,96·SE(T); "
        "ŷ = T + S_j (адд.) или ŷ = T·k_j (мульт.)."
    )
    print(
        f"{'Период':<16} {'t':>4}  "
        f"{'ŷ (адд.)':>10} {'±':>8}  "
        f"{'ŷ (мульт.)':>12} {'±':>8}"
    )
    for row in rows:
        print(
            f"{row['label']:<16} {row['t']:4d}  "
            f"{row['forecast_add']:10.2f} {row['margin_add']:8.2f}  "
            f"{row['forecast_mult']:12.2f} {row['margin_mult']:8.2f}"
        )
    print(f"\nФакт за март 2022: {fact_y} тыс. Сравнение прогноза с фактом:")
    print(f"{'Модель':<22} {'ŷ, тыс.':>10} {'|y−ŷ|, тыс.':>12} {'δ = 100·|y−ŷ|/y, %':>18}")
    print(f"{'Аддитивная':<22} {forecast_add:10.2f} {abs_err_add:12.2f} {error_add_pct:18.2f}")
    print(f"{'Мультипликативная':<22} {forecast_mult:10.2f} {abs_err_mult:12.2f} {error_mult_pct:18.2f}")


print_forecasts_table(forecasts_rows, fact_march)

print("\n--- 5. График: исходный ряд и модельные значения ---")

plt.figure(figsize=(12, 6))
plt.plot(t, Y, label='Исходные данные', marker='o', linewidth=2, color='black')
plt.plot(t, Y_model_add, label='Аддитивная модель', linestyle='--', color='blue')
plt.plot(t, Y_model_mult, label='Мультипликативная модель', linestyle='-.', color='red')
plt.title('Моделирование сезонных колебаний числа преступлений (2019-2021)')
plt.xlabel('Месяц (t)')
plt.ylabel('Количество преступлений, тыс.')
plt.legend()
plt.grid(True)
plt.show()


