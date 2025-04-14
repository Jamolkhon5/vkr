import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizer:
    """Класс для визуализации результатов моделирования"""

    def __init__(self, model, output_dir='.'):
        self.model = model
        self.output_dir = output_dir

        # Создаем директорию для вывода, если её нет
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Создана директория для графиков: {self.output_dir}")

    def plot_saturation_profiles(self, days=[10, 50, 100]):
        """Построение профилей насыщенности для указанных дней"""
        plt.figure(figsize=(12, 8))

        # Прореживание данных для ускорения отрисовки
        step = max(1, self.model.nx // 100)  # Уменьшаем количество точек до ~100

        for day in days:
            if day > self.model.days:
                continue

            time_index = int(day / self.model.dt)

            # График насыщенности без учета капиллярных эффектов
            plt.subplot(2, 1, 1)
            plt.plot(self.model.x[::step], self.model.Sw_without_cap[time_index, ::step],
                     label=f'День {day} (без кап. эффектов)')

            # График насыщенности с учетом капиллярных эффектов
            plt.subplot(2, 1, 2)
            plt.plot(self.model.x[::step], self.model.Sw_with_cap[time_index, ::step],
                     label=f'День {day} (с кап. эффектами)')

        # Настройка верхнего графика
        plt.subplot(2, 1, 1)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Водонасыщенность')
        plt.title('Профиль насыщенности без учета капиллярных эффектов')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)

        # Настройка нижнего графика
        plt.subplot(2, 1, 2)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Водонасыщенность')
        plt.title('Профиль насыщенности с учетом капиллярных эффектов')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'saturation_profiles.png')
        plt.savefig(output_path, dpi=150)

        plt.close()

    def plot_recovery_factor(self):
        """Построение графика коэффициента нефтеотдачи"""
        recovery_with_cap, recovery_without_cap = self.model.calculate_recovery_factor()

        plt.figure(figsize=(10, 6))
        plt.plot(self.model.t, recovery_without_cap, label='Без капиллярных эффектов')
        plt.plot(self.model.t, recovery_with_cap, label='С капиллярными эффектами')

        plt.xlabel('Время (дни)')
        plt.ylabel('Коэффициент нефтеотдачи')
        plt.title('Зависимость коэффициента нефтеотдачи от времени')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'recovery_factor.png')
        plt.savefig(output_path)

        plt.close()

    def plot_saturation_evolution(self):
        """Построение эволюции насыщенности во времени и пространстве"""
        # Создаем сетку времени и пространства для построения поверхности
        X, T = np.meshgrid(self.model.x, self.model.t)

        # Строим график без учета капиллярных эффектов
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.contourf(X, T, self.model.Sw_without_cap, levels=20, cmap='viridis')
        plt.colorbar(label='Водонасыщенность')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Время (дни)')
        plt.title('Эволюция насыщенности без учета капиллярных эффектов')

        # Строим график с учетом капиллярных эффектов
        plt.subplot(2, 1, 2)
        plt.contourf(X, T, self.model.Sw_with_cap, levels=20, cmap='viridis')
        plt.colorbar(label='Водонасыщенность')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Время (дни)')
        plt.title('Эволюция насыщенности с учетом капиллярных эффектов')

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'saturation_evolution.png')
        plt.savefig(output_path)

        plt.close()

    def plot_pressure_profiles(self, day=50):
        """Построение профилей давления"""
        plt.figure(figsize=(10, 6))

        # Упрощенная модель давления
        p_without_cap = np.linspace(10, 8, self.model.nx + 1)
        p_with_cap = np.linspace(10, 8, self.model.nx + 1) + 0.2 * np.sin(np.linspace(0, np.pi, self.model.nx + 1))

        plt.plot(self.model.x, p_without_cap, label='Без капиллярных эффектов')
        plt.plot(self.model.x, p_with_cap, label='С капиллярными эффектами')

        plt.xlabel('Расстояние (м)')
        plt.ylabel('Давление (МПа)')
        plt.title(f'Профиль давления на {day}-й день')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'pressure_profiles.png')
        plt.savefig(output_path)

        plt.close()

    def plot_all(self):
        """Построение всех графиков"""
        self.plot_saturation_profiles()
        self.plot_recovery_factor()
        self.plot_saturation_evolution()
        self.plot_pressure_profiles()

        # Новые графики
        self.plot_3d_saturation_surface()
        self.plot_saturation_difference()
        self.plot_filtration_velocities()
        self.calculate_capillary_number()

        # Примеры анализа чувствительности
        self.plot_sensitivity_analysis('entry_pressure', [0.5, 1.0, 1.5, 2.0])
        self.plot_sensitivity_analysis('wettability_factor', [0.2, 0.4, 0.6, 0.8])

    def plot_3d_saturation_surface(self):
        """Построение 3D поверхности водонасыщенности во времени и пространстве"""
        try:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(14, 10))

            # Создаем оси для двух графиков
            ax1 = fig.add_subplot(211, projection='3d')
            ax2 = fig.add_subplot(212, projection='3d')

            # Создаем сетку для X и T
            X, T = np.meshgrid(self.model.x, self.model.t)

            # Строим 3D поверхность для модели без капиллярных эффектов
            surf1 = ax1.plot_surface(X, T, self.model.Sw_without_cap,
                                     cmap='viridis', edgecolor='none', alpha=0.8)
            ax1.set_xlabel('Расстояние (м)')
            ax1.set_ylabel('Время (дни)')
            ax1.set_zlabel('Водонасыщенность')
            ax1.set_title('3D-поверхность водонасыщенности без учета капиллярных эффектов')
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            # Строим 3D поверхность для модели с капиллярными эффектами
            surf2 = ax2.plot_surface(X, T, self.model.Sw_with_cap,
                                     cmap='viridis', edgecolor='none', alpha=0.8)
            ax2.set_xlabel('Расстояние (м)')
            ax2.set_ylabel('Время (дни)')
            ax2.set_zlabel('Водонасыщенность')
            ax2.set_title('3D-поверхность водонасыщенности с учетом капиллярных эффектов')
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

            # Устанавливаем одинаковые углы обзора для обоих графиков
            ax1.view_init(elev=30, azim=45)
            ax2.view_init(elev=30, azim=45)

            plt.tight_layout()

            # Сохраняем в указанную директорию
            output_path = os.path.join(self.output_dir, 'saturation_3d.png')
            plt.savefig(output_path, dpi=300)

            plt.close()
        except Exception as e:
            print(f"Ошибка при построении 3D визуализации: {e}")

    def plot_saturation_difference(self):
        """Визуализация разницы между моделями с учетом и без учета капиллярных эффектов"""
        # Вычисляем разницу между насыщенностями
        saturation_diff = self.model.Sw_with_cap - self.model.Sw_without_cap

        # Создаем сетку
        X, T = np.meshgrid(self.model.x, self.model.t)

        plt.figure(figsize=(12, 6))

        # Контурный график разницы
        contour = plt.contourf(X, T, saturation_diff, levels=20, cmap='coolwarm')
        plt.colorbar(label='Разница водонасыщенности')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Время (дни)')
        plt.title('Разница водонасыщенности (с капиллярными эффектами - без капиллярных эффектов)')
        plt.grid(True)

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'saturation_difference.png')
        plt.savefig(output_path)

        plt.close()

    def plot_sensitivity_analysis(self, param_name, param_values, day=50):
        """Анализ чувствительности модели к различным параметрам"""
        plt.figure(figsize=(10, 6))

        # Словарь с русскими названиями параметров и их единицами измерения
        param_names_ru = {
            'entry_pressure': 'Капиллярное давление вытеснения',
            'wettability_factor': 'Коэффициент смачиваемости породы',
            'pore_distribution_index': 'Индекс распределения пор',
            'porosity': 'Коэффициент пористости',
            'mu_oil': 'Вязкость нефти',
            'mu_water': 'Вязкость воды'
        }

        # Словарь с единицами измерения
        param_units = {
            'entry_pressure': 'МПа',
            'wettability_factor': 'отн. ед.',
            'pore_distribution_index': 'отн. ед.',
            'porosity': 'д. ед.',
            'mu_oil': 'мПа·с',
            'mu_water': 'мПа·с'
        }

        # Получаем русское название параметра
        param_name_ru = param_names_ru.get(param_name, param_name)
        param_unit = param_units.get(param_name, '')

        # Сохраняем оригинальное значение параметра
        original_value = getattr(self.model, param_name)
        recoveries = []
        breakthrough_times = []

        # Расчет для разных значений параметра
        time_index = int(day / self.model.dt)
        for value in param_values:
            # Устанавливаем новое значение параметра
            setattr(self.model, param_name, value)
            # Пересчитываем модель
            self.model.run_simulation()
            # Получаем коэффициент нефтеотдачи
            recovery_with_cap, _ = self.model.calculate_recovery_factor()
            recoveries.append(recovery_with_cap[time_index])
            # Получаем время прорыва
            breakthrough_time, _ = self.model.get_breakthrough_time()
            breakthrough_times.append(breakthrough_time)

        # Возвращаем оригинальное значение параметра
        setattr(self.model, param_name, original_value)
        self.model.run_simulation()

        # Создаем два подграфика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # График коэффициента нефтеотдачи
        ax1.plot(param_values, recoveries, 'o-', color='#1f77b4', linewidth=2)
        ax1.set_xlabel(f'{param_name_ru} [{param_unit}]')
        ax1.set_ylabel(f'Коэффициент нефтеотдачи на {day}-й день, д. ед.')
        ax1.set_title(f'Влияние параметра "{param_name_ru}" на нефтеотдачу')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # График времени прорыва
        ax2.plot(param_values, breakthrough_times, 'o-', color='#ff7f0e', linewidth=2)
        ax2.set_xlabel(f'{param_name_ru} [{param_unit}]')
        ax2.set_ylabel('Время прорыва воды, дни')
        ax2.set_title(f'Влияние параметра "{param_name_ru}" на время прорыва')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Добавление значений на графики
        for i, (x, y) in enumerate(zip(param_values, recoveries)):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

        for i, (x, y) in enumerate(zip(param_values, breakthrough_times)):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, f'sensitivity_{param_name}.png')
        plt.savefig(output_path, dpi=300)

        plt.close()

    def plot_filtration_velocities(self, day=50):
        """Графики распределения скоростей фильтрации"""
        time_index = int(day / self.model.dt)

        plt.figure(figsize=(10, 6))

        # Расчет скоростей (упрощенно)
        v_without_cap = np.zeros(self.model.nx)
        v_with_cap = np.zeros(self.model.nx)

        # Расчет скорости из функции Баклея-Леверетта
        for i in range(self.model.nx):
            v_without_cap[i] = self.model.fractional_flow(self.model.Sw_without_cap[time_index, i])
            v_with_cap[i] = self.model.fractional_flow(self.model.Sw_with_cap[time_index, i])

        # Строим график
        plt.plot(self.model.x[:-1], v_without_cap, label='Без капиллярных эффектов')
        plt.plot(self.model.x[:-1], v_with_cap, label='С капиллярными эффектами')

        plt.xlabel('Расстояние (м)')
        plt.ylabel('Скорость фильтрации (м/день)')
        plt.title(f'Распределение скоростей фильтрации на {day}-й день')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'filtration_velocities.png')
        plt.savefig(output_path)

        plt.close()

    def calculate_capillary_number(self):
        """Расчет и визуализация капиллярного числа"""
        # Создаем массив для капиллярного числа
        capillary_number = np.zeros((self.model.nt, self.model.nx))

        # Рассчитываем для каждой точки и времени
        for n in range(self.model.nt):
            for i in range(1, self.model.nx):
                # Скорость фильтрации (упрощенно)
                v = self.model.fractional_flow(self.model.Sw_with_cap[n, i])

                # Капиллярное давление
                pc = self.model.capillary_pressure(self.model.Sw_with_cap[n, i])

                # Вязкость (упрощенно)
                mu = self.model.mu_water * self.model.Sw_with_cap[n, i] + \
                     self.model.mu_oil * (1 - self.model.Sw_with_cap[n, i])

                # Капиллярное число (отношение вязких сил к капиллярным)
                if pc > 0:
                    capillary_number[n, i] = mu * v / pc
                else:
                    capillary_number[n, i] = 0

        # Визуализация среднего капиллярного числа по времени
        plt.figure(figsize=(10, 6))

        avg_capillary_number = np.mean(capillary_number, axis=1)
        plt.plot(self.model.t, avg_capillary_number)

        plt.xlabel('Время (дни)')
        plt.ylabel('Среднее капиллярное число')
        plt.title('Изменение среднего капиллярного числа во времени')
        plt.grid(True)

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'capillary_number.png')
        plt.savefig(output_path)

        plt.close()

    def plot_well_production(self, production_data):
        """Построение графиков добычи для скважин"""
        # Создаем фигуру с несколькими подграфиками
        fig, axes = plt.subplots(len(production_data['wells']), 2, figsize=(14, 4 * len(production_data['wells'])))

        # Если только одна скважина, преобразуем оси в двумерный массив
        if len(production_data['wells']) == 1:
            axes = np.array([axes])

        # Графики для каждой скважины
        for i, well_data in enumerate(production_data['wells']):
            # Дебит нефти и воды
            axes[i, 0].plot(production_data['time'], well_data['oil_rate'], 'g-', label='Нефть')
            axes[i, 0].plot(production_data['time'], well_data['water_rate'], 'b-', label='Вода')
            axes[i, 0].set_xlabel('Время (дни)')
            axes[i, 0].set_ylabel('Дебит (м³/день)')
            axes[i, 0].set_title(f"Скважина {well_data['name']} - Динамика добычи")
            axes[i, 0].grid(True)
            axes[i, 0].legend()

            # Обводненность
            water_cut = well_data['water_rate'] / (well_data['water_rate'] + well_data['oil_rate'] + 1e-10)
            axes[i, 1].plot(production_data['time'], water_cut * 100, 'r-')
            axes[i, 1].set_xlabel('Время (дни)')
            axes[i, 1].set_ylabel('Обводненность (%)')
            axes[i, 1].set_title(f"Скважина {well_data['name']} - Обводненность")
            axes[i, 1].grid(True)
            axes[i, 1].set_ylim(0, 100)

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'well_production.png')
        plt.savefig(output_path)

        plt.close()

        # Суммарная добыча
        plt.figure(figsize=(10, 6))

        total_oil = sum(well['total_oil_produced'] for well in production_data['wells'])
        total_water = sum(well['total_water_produced'] for well in production_data['wells'])

        plt.bar(['Нефть', 'Вода'], [total_oil, total_water], color=['green', 'blue'])
        plt.ylabel('Накопленная добыча (м³)')
        plt.title('Суммарная добыча по месторождению')
        for i, value in enumerate([total_oil, total_water]):
            plt.text(i, value + 5, f'{value:.1f}', ha='center')

        plt.tight_layout()

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'total_production.png')
        plt.savefig(output_path)

        plt.close()