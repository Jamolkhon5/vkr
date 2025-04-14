# visualize_saturation.py - Визуализация профилей насыщенности
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation

# Настройка стиля
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

# Создание директории для результатов
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Загрузка данных из CSV файлов
days = [10, 50, 100]
dfs = {}

for day in days:
    file_path = os.path.join("..", "output", f"saturation_profile_day_{day}.csv")
    if os.path.exists(file_path):
        dfs[day] = pd.read_csv(file_path)
    else:
        print(f"Файл {file_path} не найден")

if not dfs:
    print("Не найдены данные для визуализации")
    exit(1)

# 1. Построение профилей насыщенности для разных дней
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

linestyles = ['-', '--', '-.']
colors = ['blue', 'green', 'red']

for i, day in enumerate(days):
    if day in dfs:
        df = dfs[day]
        # Верхний график - без капиллярных эффектов
        ax1.plot(df["Distance_m"], df["Sw_without_cap"], linestyle=linestyles[i], color=colors[i], linewidth=3,
                 label=f'День {day}')

        # Нижний график - с капиллярными эффектами
        ax2.plot(df["Distance_m"], df["Sw_with_cap"], linestyle=linestyles[i], color=colors[i], linewidth=3,
                 label=f'День {day}')

ax1.set_title('Профиль насыщенности без учета капиллярных эффектов', fontsize=18)
ax1.set_ylabel('Водонасыщенность Sw, д.ед.', fontsize=16)
ax1.legend(fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_title('Профиль насыщенности с учетом капиллярных эффектов', fontsize=18)
ax2.set_xlabel('Расстояние, м', fontsize=16)
ax2.set_ylabel('Водонасыщенность Sw, д.ед.', fontsize=16)
ax2.legend(fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "saturation_profiles_comparison.png"), dpi=300)

# 2. Построение разницы в насыщенности (с капиллярными эффектами - без них)
fig, axs = plt.subplots(len(days), 1, figsize=(12, 4 * len(days)), sharex=True)
if len(days) == 1:
    axs = [axs]

for i, day in enumerate(days):
    if day in dfs:
        df = dfs[day]
        diff = df["Sw_with_cap"] - df["Sw_without_cap"]
        axs[i].plot(df["Distance_m"], diff, linewidth=3, color='purple')
        axs[i].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        axs[i].set_title(f'Разница в насыщенности (с кап. - без кап.), день {day}', fontsize=16)
        axs[i].set_ylabel('Разница Sw, д.ед.', fontsize=14)
        axs[i].grid(True, linestyle='--', alpha=0.7)

axs[-1].set_xlabel('Расстояние, м', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "saturation_difference.png"), dpi=300)

# 3. Построение 3D-графика эволюции фронта вытеснения (для случая, когда есть данные для нескольких дней)
if len(dfs) >= 3:
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Подготовка сетки
    for i, day in enumerate(days):
        if day in dfs:
            df = dfs[day]
            X = df["Distance_m"]
            Y = np.ones_like(X) * day
            Z1 = df["Sw_without_cap"]
            Z2 = df["Sw_with_cap"]

            # Построение 3D-графиков
            ax1.plot(X, Y, Z1, color=colors[i], linewidth=2)
            ax2.plot(X, Y, Z2, color=colors[i], linewidth=2)

    # Настройка графиков
    ax1.set_xlabel('Расстояние, м', fontsize=14)
    ax1.set_ylabel('Время, дни', fontsize=14)
    ax1.set_zlabel('Водонасыщенность, д.ед.', fontsize=14)
    ax1.set_title('Эволюция фронта (без кап. эффектов)', fontsize=16)

    ax2.set_xlabel('Расстояние, м', fontsize=14)
    ax2.set_ylabel('Время, дни', fontsize=14)
    ax2.set_zlabel('Водонасыщенность, д.ед.', fontsize=14)
    ax2.set_title('Эволюция фронта (с кап. эффектами)', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "saturation_evolution_3d.png"), dpi=300)

print("Визуализация профилей насыщенности завершена. Результаты сохранены в директории:", results_dir)