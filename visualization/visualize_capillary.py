# visualize_capillary.py - Визуализация капиллярного давления
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
with open(os.path.join("..", "output", "capillary_table.json"), "r") as f:
    capillary_data = json.load(f)

# Преобразование в DataFrame
df = pd.DataFrame(capillary_data["data"])

# 1. Базовый график капиллярного давления
plt.figure(figsize=(12, 8))
plt.plot(df["Sw"], df["Pc_MPa"], 'o-', linewidth=3, color='blue')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Капиллярное давление Pc, МПа', fontsize=16)
plt.title('Кривая капиллярного давления', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "capillary_pressure_basic.png"), dpi=300)

# 2. График с тремя зонами (гидрофобная, переходная, гидрофильная)
plt.figure(figsize=(12, 8))
plt.plot(df["Sw"], df["Pc_MPa"], 'o-', linewidth=3, color='blue')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)

# Разделение на зоны
low_sw = df["Sw"] <= 0.3
mid_sw = (df["Sw"] > 0.3) & (df["Sw"] < 0.8)
high_sw = df["Sw"] >= 0.8

plt.fill_between(df["Sw"][low_sw], df["Pc_MPa"][low_sw], alpha=0.3, color='red', label='Гидрофобная зона')
plt.fill_between(df["Sw"][mid_sw], df["Pc_MPa"][mid_sw], alpha=0.3, color='purple', label='Переходная зона')
plt.fill_between(df["Sw"][high_sw], df["Pc_MPa"][high_sw], alpha=0.3, color='green', label='Гидрофильная зона')

plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Капиллярное давление Pc, МПа', fontsize=16)
plt.title('Кривая капиллярного давления по зонам смачиваемости', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "capillary_pressure_zones.png"), dpi=300)

# 3. График логарифмический для анализа параметров Брукса-Кори
valid_data = df[(df["Sw"] > 0.2) & (df["Sw"] < 0.8) & (df["Pc_MPa"] > 0)]

plt.figure(figsize=(12, 8))
plt.scatter(valid_data["Sw"], valid_data["Pc_MPa"], s=100, alpha=0.7, color='blue')
plt.yscale('log')
plt.xlabel('Водонасыщенность Sw, д.ед.', fontsize=16)
plt.ylabel('Капиллярное давление Pc, МПа (лог. шкала)', fontsize=16)
plt.title('Капиллярное давление в логарифмическом масштабе', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "capillary_pressure_log.png"), dpi=300)

print("Визуализация капиллярного давления завершена. Результаты сохранены в директории:", results_dir)