# visualize_3d.py - Создание 3D визуализаций результатов моделирования
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Настройка стиля
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 14

# Создание директории для результатов
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Загрузка данных
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

# 1. 3D поверхность водонасыщенности
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Подготовка сетки X (расстояние) и Y (время)
all_distances = dfs[days[0]]["Distance_m"].values
all_times = np.array(days)

# Подготовка 2D-сетки
X, Y = np.meshgrid(all_distances, all_times)

# Подготовка данных о насыщенности
Z1 = np.zeros((len(days), len(all_distances)))
Z2 = np.zeros((len(days), len(all_distances)))

for i, day in enumerate(days):
    Z1[i, :] = dfs[day]["Sw_without_cap"].values
    Z2[i, :] = dfs[day]["Sw_with_cap"].values

# Построение 3D-поверхностей
surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
surf2 = ax2.plot_surface(X, Y, Z2, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)

# Настройка графиков
ax1.set_xlabel('Расстояние, м', fontsize=14)
ax1.set_ylabel('Время, дни', fontsize=14)
ax1.set_zlabel('Водонасыщенность, д.ед.', fontsize=14)
ax1.set_title('Эволюция водонасыщенности без капиллярных эффектов', fontsize=16)
ax1.view_init(elev=30, azim=220)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

ax2.set_xlabel('Расстояние, м', fontsize=14)
ax2.set_ylabel('Время, дни', fontsize=14)
ax2.set_zlabel('Водонасыщенность, д.ед.', fontsize=14)
ax2.set_title('Эволюция водонасыщенности с капиллярными эффектами', fontsize=16)
ax2.view_init(elev=30, azim=220)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "saturation_evolution_3d_surface.png"), dpi=300)

# 2. 3D-визуализация разницы в насыщенности
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Вычисление разницы в насыщенности
Z_diff = Z2 - Z1

# Построение 3D-поверхности разницы
surf = ax.plot_surface(X, Y, Z_diff, cmap=cm.coolwarm, linewidth=0, antialiased=True)

# Настройка графика
ax.set_xlabel('Расстояние, м', fontsize=14)
ax.set_ylabel('Время, дни', fontsize=14)
ax.set_zlabel('Разница водонасыщенности, д.ед.', fontsize=14)
ax.set_title('Влияние капиллярных эффектов на насыщенность', fontsize=16)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "saturation_difference_3d.png"), dpi=300)

# 3. 3D-визуализация скорости перемещения фронта
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Вычисление скорости изменения насыщенности (градиент по пространству)
velocity1 = np.zeros((len(days), len(all_distances)-1))
velocity2 = np.zeros((len(days), len(all_distances)-1))

for i, day in enumerate(days):
    for j in range(len(all_distances)-1):
        dx = all_distances[j+1] - all_distances[j]
        velocity1[i, j] = (Z1[i, j+1] - Z1[i, j]) / dx
        velocity2[i, j] = (Z2[i, j+1] - Z2[i, j]) / dx

# Подготовка сетки для скорости
X_v, Y_v = np.meshgrid(all_distances[:-1], all_times)

# Построение 3D-поверхностей скорости
surf1 = ax1.plot_surface(X_v, Y_v, velocity1, cmap=cm.plasma, linewidth=0, antialiased=True)
surf2 = ax2.plot_surface(X_v, Y_v, velocity2, cmap=cm.plasma, linewidth=0, antialiased=True)

# Настройка графиков
ax1.set_xlabel('Расстояние, м', fontsize=14)
ax1.set_ylabel('Время, дни', fontsize=14)
ax1.set_zlabel('Градиент насыщенности, 1/м', fontsize=14)
ax1.set_title('Скорость изменения насыщенности без капиллярных эффектов', fontsize=16)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

ax2.set_xlabel('Расстояние, м', fontsize=14)
ax2.set_ylabel('Время, дни', fontsize=14)
ax2.set_zlabel('Градиент насыщенности, 1/м', fontsize=14)
ax2.set_title('Скорость изменения насыщенности с капиллярными эффектами', fontsize=16)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "saturation_gradient_3d.png"), dpi=300)

print("3D-визуализации завершены. Результаты сохранены в директории:", results_dir)