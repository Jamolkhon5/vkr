# model.py - Базовая модель фильтрации нефти методом апвинд
import numpy as np
import matplotlib.pyplot as plt

class OilFiltrationModel:
    """
    Базовая модель одномерной фильтрации нефти в пористой среде
    с использованием метода апвинд
    """

    def __init__(self):
        # Параметры пласта
        self.length = 1000.0  # длина пласта, м (изменено с 100.0)
        self.porosity = 0.2  # пористость

        # Параметры флюидов
        self.mu_oil = 5.0  # вязкость нефти, мПа·с
        self.mu_water = 1.0  # вязкость воды, мПа·с
        self.initial_water_saturation = 0.2  # начальная водонасыщенность
        self.residual_oil_saturation = 0.2  # остаточная нефтенасыщенность

        # Параметры расчёта
        self.nx = 1000  # число узлов сетки (изменено с 100)
        self.dx = self.length / self.nx  # шаг по x
        self.days = 100  # дней симуляции
        self.dt = 0.05  # шаг по времени, дней
        self.nt = int(self.days / self.dt) + 1  # число временных шагов

        # Создаем сетки
        self.x = np.linspace(0, self.length, self.nx + 1)
        self.t = np.linspace(0, self.days, self.nt)

        # Создаем массивы для хранения результатов
        # Насыщенность с учетом и без учета капиллярных эффектов
        self.Sw_with_cap = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation
        self.Sw_without_cap = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation

        # Устанавливаем граничные условия - закачка воды на входе
        self.Sw_with_cap[:, 0] = 0.8
        self.Sw_without_cap[:, 0] = 0.8

        # Параметры для модели капиллярного давления Брукса-Кори
        self.entry_pressure = 1.0  # давление входа, МПа
        self.pore_distribution_index = 1.5  # индекс распределения пор (λ)
        self.wettability_factor = 0.3  # коэффициент смачиваемости (изменено с 0.6 на 0.3 для усиления эффектов)

    def relative_permeability_water(self, Sw):
        """Относительная проницаемость для воды"""
        Swc = self.initial_water_saturation
        Sor = self.residual_oil_saturation

        if Sw <= Swc:
            return 0.0
        elif Sw >= 1 - Sor:
            return 1.0
        else:
            Swn = (Sw - Swc) / (1 - Swc - Sor)
            return Swn ** 3  # кубическая зависимость

    def relative_permeability_oil(self, Sw):
        """Относительная проницаемость для нефти"""
        Swc = self.initial_water_saturation
        Sor = self.residual_oil_saturation

        if Sw >= 1 - Sor:
            return 0.0
        elif Sw <= Swc:
            return 1.0
        else:
            Son = (1 - Sw - Sor) / (1 - Swc - Sor)
            return Son ** 2  # квадратичная зависимость

    def fractional_flow(self, Sw):
        """Функция Баклея-Леверетта"""
        krw = self.relative_permeability_water(Sw)
        kro = self.relative_permeability_oil(Sw)

        # Добавляем малое число для избежания деления на ноль
        M = (krw / self.mu_water) / (kro / self.mu_oil + 1e-10)
        return M / (1 + M)

    def capillary_pressure(self, Sw):
        """
        Функция капиллярного давления по модели Брукса-Кори с плавным переходом
        в граничных зонах для повышения численной стабильности.
        """
        # Граничные значения насыщенности
        Swc = self.initial_water_saturation
        Sor = self.residual_oil_saturation

        # Избегаем численных проблем у границ диапазона насыщенности
        epsilon = 0.01  # параметр сглаживания вблизи границ

        if Sw <= Swc + epsilon:
            # Плавный переход к максимальному капиллярному давлению
            alpha = (Sw - Swc) / epsilon
            max_pc = self.entry_pressure * 3.0  # максимальное капиллярное давление, МПа
            return max_pc * (1.0 - alpha) + self.entry_pressure * alpha

        elif Sw >= 1 - Sor - epsilon:
            # Плавный переход к нулю капиллярного давления
            alpha = (1 - Sor - Sw) / epsilon
            return self.entry_pressure * 0.05 * alpha  # близко к нулю в конечной точке

        else:
            # Нормализованная водонасыщенность (эффективная)
            Se = (Sw - Swc) / (1 - Swc - Sor)

            # Модель Брукса-Кори
            pc = self.entry_pressure * (Se ** (-1.0 / self.pore_distribution_index))

            # Корректировка с учетом смачиваемости
            # Для гидрофобной среды (oil-wet) увеличиваем капиллярное давление
            pc = pc * (2.0 - self.wettability_factor)

            return pc

    def diffusion_coefficient(self, Sw):
        """Коэффициент капиллярной диффузии"""
        # Предотвращаем выход за граничные значения
        Sw = max(min(Sw, 0.99), 0.01)

        # Вычисление производной функции Баклея-Леверетта
        delta = 1e-4
        Sw_minus = max(Sw - delta, 0.01)
        Sw_plus = min(Sw + delta, 0.99)

        df_dS = (self.fractional_flow(Sw_plus) - self.fractional_flow(Sw_minus)) / (2 * delta)

        # Вычисление производной капиллярного давления
        dpc_dS = (self.capillary_pressure(Sw_plus) - self.capillary_pressure(Sw_minus)) / (2 * delta)

        # Увеличиваем коэффициент проницаемости для ОЧЕНЬ СИЛЬНОГО усиления эффекта
        k = 10.0  # изменено с 5.0 на 10.0

        mu = max(self.mu_water * Sw + self.mu_oil * (1 - Sw), 0.1)

        # Теоретическая формула с ОЧЕНЬ СИЛЬНЫМ усилением
        D = -k / (self.porosity * mu) * df_dS * dpc_dS

        # Увеличиваем множитель для усиления эффекта
        max_diffusion = 0.45 * self.dx ** 2 / self.dt

        # Обрабатываем отрицательные значения и применяем ограничение устойчивости с ОЧЕНЬ СИЛЬНЫМ усилением
        if D < 0:
            return min(abs(D) * 8.0, max_diffusion)  # изменено с 5.0 на 8.0
        else:
            return min(D * 8.0, max_diffusion)  # изменено с 5.0 на 8.0

    def run_simulation(self):
        """Запуск моделирования"""
        # Моделирование с учетом капиллярных эффектов
        for n in range(self.nt - 1):
            for i in range(1, self.nx):
                # Апвинд схема для конвективного члена
                f_i = self.fractional_flow(self.Sw_with_cap[n, i])
                f_im1 = self.fractional_flow(self.Sw_with_cap[n, i - 1])

                # Диффузионный член (капиллярные эффекты)
                D_i = self.diffusion_coefficient(self.Sw_with_cap[n, i])

                # Схема апвинд с учетом капиллярных эффектов
                self.Sw_with_cap[n + 1, i] = self.Sw_with_cap[n, i] - \
                                             (self.dt / self.dx) * (f_i - f_im1) + \
                                             (self.dt / self.dx ** 2) * D_i * (
                                                     self.Sw_with_cap[n, i + 1] - 2 * self.Sw_with_cap[n, i] +
                                                     self.Sw_with_cap[n, i - 1])

            # Граничное условие на правом конце
            self.Sw_with_cap[n + 1, -1] = self.Sw_with_cap[n + 1, -2]

        # Моделирование без учета капиллярных эффектов
        for n in range(self.nt - 1):
            for i in range(1, self.nx):
                # Апвинд схема для конвективного члена
                f_i = self.fractional_flow(self.Sw_without_cap[n, i])
                f_im1 = self.fractional_flow(self.Sw_without_cap[n, i - 1])

                # Схема апвинд без учета капиллярных эффектов
                self.Sw_without_cap[n + 1, i] = self.Sw_without_cap[n, i] - \
                                                (self.dt / self.dx) * (f_i - f_im1)

            # Граничное условие на правом конце
            self.Sw_without_cap[n + 1, -1] = self.Sw_without_cap[n + 1, -2]

    def calculate_recovery_factor(self):
        """Расчет коэффициента нефтеотдачи"""
        initial_oil = 1 - self.initial_water_saturation

        recovery_with_cap = np.zeros(self.nt)
        recovery_without_cap = np.zeros(self.nt)

        for n in range(self.nt):
            # Средняя нефтенасыщенность
            avg_oil_with_cap = 1 - np.mean(self.Sw_with_cap[n, :])
            avg_oil_without_cap = 1 - np.mean(self.Sw_without_cap[n, :])

            # Коэффициент нефтеотдачи
            recovery_with_cap[n] = (initial_oil - avg_oil_with_cap) / initial_oil
            recovery_without_cap[n] = (initial_oil - avg_oil_without_cap) / initial_oil

        return recovery_with_cap, recovery_without_cap

    def get_breakthrough_time(self):
        """Определение времени прорыва воды"""
        threshold = self.initial_water_saturation + 0.05

        # Время прорыва с учетом капиллярных эффектов
        breakthrough_with_cap = self.days
        for n in range(self.nt):
            if self.Sw_with_cap[n, -1] > threshold:
                breakthrough_with_cap = self.t[n]
                break

        # Время прорыва без учета капиллярных эффектов
        breakthrough_without_cap = self.days
        for n in range(self.nt):
            if self.Sw_without_cap[n, -1] > threshold:
                breakthrough_without_cap = self.t[n]
                break

        return breakthrough_with_cap, breakthrough_without_cap


class MultiWellModel(OilFiltrationModel):
    """
    Расширенная модель фильтрации с учетом нескольких скважин на одномерном пласте
    """

    def __init__(self, length=1000.0, nx=1000, days=100, wells_config=None):
        # Вызываем инициализатор базового класса
        super().__init__()

        # Обновляем параметры
        self.length = length
        self.nx = nx
        self.dx = self.length / self.nx
        self.days = days
        self.nt = int(self.days / self.dt) + 1

        # Создаем сетки заново
        self.x = np.linspace(0, self.length, self.nx + 1)
        self.t = np.linspace(0, self.days, self.nt)

        # Создаем массивы для хранения результатов
        self.Sw_with_cap = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation
        self.Sw_without_cap = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation

        # Конфигурация скважин
        if wells_config is None:
            # По умолчанию - два нагнетателя и три добывающих для длинного пласта
            self.wells_config = {
                'injectors': [
                    {'position': 0, 'rate': 2.0, 'water_cut': 1.0},  # Нагнетательная на левой границе
                    {'position': int(0.3 * self.nx), 'rate': 1.8, 'water_cut': 1.0}  # Нагнетательная на 30% длины
                ],
                'producers': [
                    {'position': int(0.6 * self.nx), 'rate': 1.3},  # Добывающая на 60% длины
                    {'position': int(0.8 * self.nx), 'rate': 1.5},  # Добывающая на 80% длины
                    {'position': self.nx, 'rate': 1.0}  # Добывающая на правой границе
                ]
            }
        else:
            self.wells_config = wells_config

        # Расход через границы пласта
        self.flow_rates = np.zeros(self.nx + 1)

        # Инициализация скважин
        self.setup_wells()

    def setup_wells(self):
        """Настройка скважин и начальных условий"""
        # Сбрасываем расходы
        self.flow_rates = np.zeros(self.nx + 1)

        # Устанавливаем расходы для нагнетательных скважин
        for injector in self.wells_config['injectors']:
            pos = injector['position']
            rate = injector['rate']
            water_cut = injector['water_cut']

            # Устанавливаем расход
            self.flow_rates[pos] += rate

            # Устанавливаем постоянную водонасыщенность на нагнетательной скважине
            self.Sw_with_cap[:, pos] = water_cut
            self.Sw_without_cap[:, pos] = water_cut

        # Устанавливаем расходы для добывающих скважин (отрицательные)
        for producer in self.wells_config['producers']:
            pos = producer['position']
            rate = producer['rate']

            # Устанавливаем расход (отрицательный для добывающих)
            self.flow_rates[pos] -= rate

    def run_simulation(self):
        """Запуск моделирования с учетом нескольких скважин"""
        # Моделирование с учетом капиллярных эффектов
        for n in range(self.nt - 1):
            # Обновляем граничные условия для скважин
            self.apply_well_conditions(n)

            for i in range(1, self.nx):
                # Проверяем наличие скважины в данной точке
                if i in [inj['position'] for inj in self.wells_config['injectors']] or \
                        i in [prod['position'] for prod in self.wells_config['producers']]:
                    continue  # Пропускаем узлы со скважинами, там заданы граничные условия

                # Апвинд схема для конвективного члена
                f_i = self.fractional_flow(self.Sw_with_cap[n, i])
                f_im1 = self.fractional_flow(self.Sw_with_cap[n, i - 1])

                # Диффузионный член (капиллярные эффекты)
                D_i = self.diffusion_coefficient(self.Sw_with_cap[n, i])

                # Учитываем направление потока между узлами
                v_i = self.flow_rates[i]
                v_im1 = self.flow_rates[i - 1]

                # Модифицированная схема апвинд с учетом скважин
                conv_term = 0.0
                if v_i > 0:  # Поток слева направо
                    conv_term += v_i * f_i
                else:  # Поток справа налево
                    conv_term += v_i * self.fractional_flow(self.Sw_with_cap[n, i + 1])

                if v_im1 > 0:  # Поток слева направо
                    conv_term -= v_im1 * f_im1
                else:  # Поток справа налево
                    conv_term -= v_im1 * f_i

                # Схема апвинд с учетом скважин и капиллярных эффектов
                self.Sw_with_cap[n + 1, i] = self.Sw_with_cap[n, i] - \
                                             (self.dt / self.dx) * conv_term + \
                                             (self.dt / self.dx ** 2) * D_i * (
                                                     self.Sw_with_cap[n, i + 1] - 2 * self.Sw_with_cap[n, i] +
                                                     self.Sw_with_cap[n, i - 1])

            # Граничные условия на правом конце
            if self.nx not in [prod['position'] for prod in self.wells_config['producers']]:
                self.Sw_with_cap[n + 1, -1] = self.Sw_with_cap[n + 1, -2]

        # Моделирование без учета капиллярных эффектов
        for n in range(self.nt - 1):
            # Обновляем граничные условия для скважин
            self.apply_well_conditions(n)

            for i in range(1, self.nx):
                # Пропускаем узлы со скважинами
                if i in [inj['position'] for inj in self.wells_config['injectors']] or \
                        i in [prod['position'] for prod in self.wells_config['producers']]:
                    continue

                # Апвинд схема для конвективного члена с учетом направления потока
                f_i = self.fractional_flow(self.Sw_without_cap[n, i])
                f_im1 = self.fractional_flow(self.Sw_without_cap[n, i - 1])

                # Учитываем направление потока между узлами
                v_i = self.flow_rates[i]
                v_im1 = self.flow_rates[i - 1]

                # Модифицированная схема апвинд с учетом скважин
                conv_term = 0.0
                if v_i > 0:  # Поток слева направо
                    conv_term += v_i * f_i
                else:  # Поток справа налево
                    conv_term += v_i * self.fractional_flow(self.Sw_without_cap[n, i + 1])

                if v_im1 > 0:  # Поток слева направо
                    conv_term -= v_im1 * f_im1
                else:  # Поток справа налево
                    conv_term -= v_im1 * f_i

                # Схема апвинд без учета капиллярных эффектов, но с учетом скважин
                self.Sw_without_cap[n + 1, i] = self.Sw_without_cap[n, i] - \
                                                (self.dt / self.dx) * conv_term

            # Граничные условия на правом конце
            if self.nx not in [prod['position'] for prod in self.wells_config['producers']]:
                self.Sw_without_cap[n + 1, -1] = self.Sw_without_cap[n + 1, -2]

    def apply_well_conditions(self, time_step):
        """Применение условий на скважинах для конкретного временного шага"""
        # Нагнетательные скважины
        for injector in self.wells_config['injectors']:
            pos = injector['position']
            water_cut = injector['water_cut']

            # Устанавливаем водонасыщенность на нагнетательной скважине
            self.Sw_with_cap[time_step + 1, pos] = water_cut
            self.Sw_without_cap[time_step + 1, pos] = water_cut

        # Для добывающих скважин специальных условий нет,
        # там просто отбирается флюид с текущей насыщенностью

    def calculate_recovery_factor(self):
        """Расчет коэффициента нефтеотдачи для модели с несколькими скважинами"""
        # Начальный объем нефти в пласте (м³) = площадь * пористость * (1 - начальная водонасыщенность)
        # Для одномерной модели "площадь" = длина * единичная ширина * единичная высота = длина
        initial_oil_volume = self.length * self.porosity * (1 - self.initial_water_saturation)

        # Массивы для хранения накопленной добычи нефти и коэффициентов нефтеотдачи
        oil_produced_with_cap = np.zeros(self.nt)
        oil_produced_without_cap = np.zeros(self.nt)
        recovery_with_cap = np.zeros(self.nt)
        recovery_without_cap = np.zeros(self.nt)

        # Расчет добычи нефти для всех временных шагов
        for t in range(1, self.nt):
            # Добыча на предыдущем шаге
            prev_oil_with_cap = oil_produced_with_cap[t - 1]
            prev_oil_without_cap = oil_produced_without_cap[t - 1]

            # Временной шаг
            dt = self.t[t] - self.t[t - 1]

            # Суммирование добычи со всех добывающих скважин
            for producer in self.wells_config['producers']:
                pos = producer['position']
                rate = producer['rate']

                # Нефтенасыщенность на добывающей скважине
                oil_saturation_with_cap = 1 - self.Sw_with_cap[t, pos]
                oil_saturation_without_cap = 1 - self.Sw_without_cap[t, pos]

                # Дебит нефти (м³/день)
                oil_rate_with_cap = rate * oil_saturation_with_cap
                oil_rate_without_cap = rate * oil_saturation_without_cap

                # Накопленная добыча нефти (м³) с учетом временного шага
                oil_produced_with_cap[t] = prev_oil_with_cap + oil_rate_with_cap * dt
                oil_produced_without_cap[t] = prev_oil_without_cap + oil_rate_without_cap * dt

            # Коэффициент нефтеотдачи (от 0 до 1)
            recovery_with_cap[t] = min(1.0, oil_produced_with_cap[t] / initial_oil_volume)
            recovery_without_cap[t] = min(1.0, oil_produced_without_cap[t] / initial_oil_volume)

        return recovery_with_cap, recovery_without_cap

    def get_well_production_data(self):
        """Получение данных о добыче из скважин"""
        # Создаем словарь для хранения данных о добыче
        production_data = {
            'time': self.t,
            'wells': []
        }

        # Данные для добывающих скважин
        for i, producer in enumerate(self.wells_config['producers']):
            pos = producer['position']
            rate = producer['rate']

            # Инициализируем массивы для данных
            water_rate = np.zeros(self.nt)
            oil_rate = np.zeros(self.nt)

            # Расчет дебитов для каждого временного шага
            for n in range(self.nt):
                # Обводненность в точке добывающей скважины
                water_cut_with_cap = self.Sw_with_cap[n, pos]

                # Дебит воды и нефти (с учетом капиллярных эффектов)
                water_rate[n] = rate * water_cut_with_cap
                oil_rate[n] = rate * (1 - water_cut_with_cap)

            # Добавляем данные скважины
            well_data = {
                'name': f'Producer_{i + 1}',
                'position': pos,
                'water_rate': water_rate,
                'oil_rate': oil_rate,
                'total_water_produced': np.trapz(water_rate, self.t),
                'total_oil_produced': np.trapz(oil_rate, self.t)
            }
            production_data['wells'].append(well_data)

        return production_data

    def plot_wells_location(self):
        """Отображение расположения скважин на пласте"""
        plt.figure(figsize=(15, 4))  # Увеличенный размер для лучшей видимости длинного пласта
        plt.plot([0, self.length], [0, 0], 'k-', linewidth=2)  # Пласт

        # Рисуем нагнетательные скважины (синие треугольники вверх)
        for injector in self.wells_config['injectors']:
            pos = injector['position']
            x_pos = pos * self.dx  # Преобразуем индекс в метры
            plt.plot(x_pos, 0, 'b^', markersize=10)
            plt.text(x_pos, 0.2, f'Инж. {injector["rate"]:.1f} м³/день',
                     ha='center', va='bottom', color='blue')

        # Рисуем добывающие скважины (красные треугольники вниз)
        for producer in self.wells_config['producers']:
            pos = producer['position']
            x_pos = pos * self.dx  # Преобразуем индекс в метры
            plt.plot(x_pos, 0, 'rv', markersize=10)
            plt.text(x_pos, -0.2, f'Доб. {producer["rate"]:.1f} м³/день',
                     ha='center', va='top', color='red')

        plt.xlim(-50, self.length + 50)  # Расширенные границы для лучшей видимости
        plt.ylim(-1, 1)
        plt.xlabel('Расстояние (м)')
        plt.title('Расположение скважин на пласте')
        plt.yticks([])
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig('wells_location.png')
        plt.close()