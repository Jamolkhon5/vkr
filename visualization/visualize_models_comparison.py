# visualize_models_comparison.py - Сравнение моделей с учетом и без учета капиллярных эффектов
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настройка стиля
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

# Создание директории для результатов
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Загрузка данных о профиле насыщенности
day = 50  # день для анализа
file_path = os.path.join("..", "output", f"saturation_profile_day_{day}.csv")

if not os.path.exists(file_path):
    print(f"Файл {file_path} не найден")
    exit(1)

df = pd.read_csv(file_path)

# 1. Создание графика для анализа капиллярного противотока
plt.figure(figsize=(14, 10))

# Первый подграфик - профили насыщенности
ax1 = plt.subplot(211)
ax1.plot(df["Distance_m"], df["Sw_without_cap"], linewidth=3, color='blue', label='Без капиллярных эффектов')
ax1.plot(df["Distance_m"], df["Sw_with_cap"], linewidth=3, color='red', label='С капиллярными эффектами')
ax1.set_ylabel('Водонасыщенность Sw, д.ед.', fontsize=16)
ax1.set_title(f'Профили насыщенности на {day}-й день', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=14)

# Второй подграфик - разница в насыщенности
ax2 = plt.subplot(212, sharex=ax1)
diff = df["Sw_with_cap"] - df["Sw_without_cap"]
ax2.plot(df["Distance_m"], diff, linewidth=3, color='purple')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
ax2.set_xlabel('Расстояние, м', fontsize=16)
ax2.set_ylabel('Разница водонасыщенности, д.ед.', fontsize=16)
ax2.set_title(f'Влияние капиллярных эффектов на {day}-й день', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.7)

# Определение зон капиллярного противотока
positive_diff = diff > 0.005
negative_diff = diff < -0.005
ax2.fill_between(df["Distance_m"], diff, 0, where=positive_diff, color='green', alpha=0.5, label='Зона повышенной насыщенности')
ax2.fill_between(df["Distance_m"], diff, 0, where=negative_diff, color='red', alpha=0.5, label='Зона пониженной насыщенности')
ax2.legend(fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "capillary_countercurrent_analysis.png"), dpi=300)

# 2. Анализ скорости движения фронта и времени прорыва
# Информация о фронте вытеснения (из консольного вывода)
front_data = {
    "Параметры": ["Время прорыва воды (дни)", "Скорость движения фронта (м/день)", "Ширина переходной зоны (м)"],
    "Без капиллярных эффектов": [39.5, 2.53, 2.0],
    "С капиллярными эффектами": [43.0, 2.33, 25.0]
}

front_df = pd.DataFrame(front_data)

plt.figure(figsize=(12, 8))
bar_width = 0.35
x = np.arange(len(front_df["Параметры"]))

# Вычисление процентного изменения
pct_change = [(front_df["С капиллярными эффектами"][i] - front_df["Без капиллярных эффектов"][i]) /
              front_df["Без капиллярных эффектов"][i] * 100 for i in range(len(front_df["Параметры"]))]

# Создаем текстовые метки с процентным изменением
text_labels = [f"{pct:.1f}%" for pct in pct_change]

bars1 = plt.bar(x - bar_width/2, front_df["Без капиллярных эффектов"], bar_width, label='Без капиллярных эффектов', color='blue')
bars2 = plt.bar(x + bar_width/2, front_df["С капиллярными эффектами"], bar_width, label='С капиллярными эффектами', color='red')

plt.xlabel('Параметр', fontsize=16)
plt.ylabel('Значение', fontsize=16)
plt.title('Сравнение параметров фронта вытеснения', fontsize=18)
plt.xticks(x, front_df["Параметры"], rotation=15)
plt.legend(fontsize=14)

# Добавление текстовых меток с процентным изменением
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height = max(bar1.get_height(), bar2.get_height())
    plt.text(x[i], height * 1.05, text_labels[i], ha='center', va='bottom', fontsize=12, color='black', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "front_parameters_comparison.png"), dpi=300)

# 3. Анализ коэффициента нефтеотдачи
# Данные о коэффициенте нефтеотдачи (из консольного вывода)
recovery_data = {
    "Время (дни)": [10, 30, 50, 70, 100],
    "Без капиллярных эффектов": [0.13, 0.38, 0.53, 0.56, 0.59],
    "С капиллярными эффектами": [0.13, 0.37, 0.55, 0.58, 0.60]
}

recovery_df = pd.DataFrame(recovery_data)

plt.figure(figsize=(12, 8))
plt.plot(recovery_df["Время (дни)"], recovery_df["Без капиллярных эффектов"], 'o-', linewidth=3, color='blue', label='Без капиллярных эффектов')
plt.plot(recovery_df["Время (дни)"], recovery_df["С капиллярными эффектами"], 'o-', linewidth=3, color='red', label='С капиллярными эффектами')

# Вычисление разницы в коэффициенте нефтеотдачи
for i, day in enumerate(recovery_df["Время (дни)"]):
    diff = recovery_df["С капиллярными эффектами"][i] - recovery_df["Без капиллярных эффектов"][i]
    plt.annotate(f"{diff:.3f}",
                xy=(day, (recovery_df["С капиллярными эффектами"][i] + recovery_df["Без капиллярных эффектов"][i])/2),
                xytext=(10, 0),
                textcoords="offset points",
                ha='left', va='center',
                fontsize=12,
                arrowprops=dict(arrowstyle="->", color='black'))

plt.xlabel('Время, дни', fontsize=16)
plt.ylabel('Коэффициент нефтеотдачи, д.ед.', fontsize=16)
plt.title('Динамика коэффициента нефтеотдачи', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "recovery_factor_analysis.png"), dpi=300)

print("Анализ и сравнение моделей завершены. Результаты сохранены в директории:", results_dir)