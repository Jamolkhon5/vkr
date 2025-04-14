# visualize_rel_perm.py - Визуализация относительной проницаемости
import os
import json
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

# Загрузка данных из JSON файла
with open(os.path.join("..", "output", "rel_perm_table.json"), "r") as f:
    rel_perm_data = json.load(f)

# Преобразование в DataFrame
df = pd.DataFrame(rel_perm_data["data"])

# 1. Базовый график ОФП
plt.figure(figsize=(12, 8))
plt.plot(df["Sw"], df["Krw"], 'o-', linewidth=3, color='blue', label='Krw (вода)')
plt.plot(df["Sw"], df["Kro"], 'o-', linewidth=3, color='green', label='Kro (нефть)')
plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Относительная проницаемость Kr, д.ед.', fontsize=16)
plt.title('Кривые относительной проницаемости', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "rel_perm_basic.png"), dpi=300)

# 2. График с отмеченными критическими точками
plt.figure(figsize=(12, 8))
plt.plot(df["Sw"], df["Krw"], 'o-', linewidth=3, color='blue', label='Krw (вода)')
plt.plot(df["Sw"], df["Kro"], 'o-', linewidth=3, color='green', label='Kro (нефть)')

# Находим критические точки
Swc = df["Sw"][df["Krw"] > 0.001].min()  # Критическая водонасыщенность
Sor = 1 - df["Sw"][df["Kro"] > 0.001].max()  # Остаточная нефтенасыщенность
crossover = None
min_diff = float('inf')

for i in range(len(df)):
    diff = abs(df["Krw"].iloc[i] - df["Kro"].iloc[i])
    if diff < min_diff:
        min_diff = diff
        crossover = i

plt.axvline(x=Swc, color='blue', linestyle='--', alpha=0.7, label=f'Swc = {Swc:.2f}')
plt.axvline(x=1-Sor, color='green', linestyle='--', alpha=0.7, label=f'1-Sor = {1-Sor:.2f}')
if crossover is not None:
    plt.plot(df["Sw"].iloc[crossover], df["Krw"].iloc[crossover], 'ro', markersize=10, label=f'Точка пересечения Sw = {df["Sw"].iloc[crossover]:.2f}')

plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Относительная проницаемость Kr, д.ед.', fontsize=16)
plt.title('Кривые ОФП с критическими точками', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "rel_perm_critical_points.png"), dpi=300)

# 3. График с соотношением подвижностей
plt.figure(figsize=(12, 8))

# Вычисляем отношение подвижностей M = (krw/μw)/(kro/μo)
mu_w = 1.0  # мПа·с
mu_o = 5.0  # мПа·с
M = np.zeros_like(df["Sw"])

for i in range(len(df)):
    if df["Kro"].iloc[i] > 0:
        M[i] = (df["Krw"].iloc[i] / mu_w) / (df["Kro"].iloc[i] / mu_o)
    else:
        M[i] = np.nan

plt.plot(df["Sw"], M, 'o-', linewidth=3, color='red')
plt.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='M = 1 (стабильный фронт)')

plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Отношение подвижностей M', fontsize=16)
plt.title('Отношение подвижностей', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14)
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "mobility_ratio.png"), dpi=300)

print("Визуализация относительной проницаемости завершена. Результаты сохранены в директории:", results_dir)