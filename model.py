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


# Добавить в файл model.py

class MultiWell2DModel(OilFiltrationModel):
    """
    Расширенная модель фильтрации с учетом нескольких скважин в двумерном пласте
    """

    def __init__(self, length_x=1000.0, length_y=1000.0, nx=100, ny=100, days=100, wells_config=None):
        # Вызываем инициализатор базового класса
        super().__init__()

        # Обновляем параметры для 2D модели
        self.length_x = length_x
        self.length_y = length_y
        self.nx = nx
        self.ny = ny
        self.dx = self.length_x / self.nx
        self.dy = self.length_y / self.ny
        self.days = days
        self.nt = int(self.days / self.dt) + 1

        # Создаем сетки для 2D
        self.x = np.linspace(0, self.length_x, self.nx + 1)
        self.y = np.linspace(0, self.length_y, self.ny + 1)
        self.t = np.linspace(0, self.days, self.nt)

        # Создаем массивы для хранения результатов
        self.Sw_with_cap = np.ones((self.nt, self.ny + 1, self.nx + 1)) * self.initial_water_saturation
        self.Sw_without_cap = np.ones((self.nt, self.ny + 1, self.nx + 1)) * self.initial_water_saturation

        # Конфигурация скважин в 2D
        if wells_config is None:
            # По умолчанию - конфигурация в виде 5-точечного элемента
            self.wells_config = {
                'injectors': [
                    {'position_x': int(0.1 * self.nx), 'position_y': int(0.1 * self.ny), 'rate': 2.0, 'water_cut': 1.0},
                    {'position_x': int(0.1 * self.nx), 'position_y': int(0.9 * self.ny), 'rate': 2.0, 'water_cut': 1.0},
                    {'position_x': int(0.5 * self.nx), 'position_y': int(0.5 * self.ny), 'rate': 3.0, 'water_cut': 1.0},
                ],
                'producers': [
                    {'position_x': int(0.9 * self.nx), 'position_y': int(0.1 * self.ny), 'rate': 3.5},
                    {'position_x': int(0.9 * self.nx), 'position_y': int(0.9 * self.ny), 'rate': 3.5},
                ]
            }
        else:
            self.wells_config = wells_config

        # Матрицы потоков
        self.flow_rates_x = np.zeros((self.ny + 1, self.nx + 1))
        self.flow_rates_y = np.zeros((self.ny + 1, self.nx + 1))

        # Инициализация скважин
        self.setup_wells()

        # Матрица связи между скважинами для трубочек
        self.well_connections = self.calculate_well_connections()

    def setup_wells(self):
        """Настройка скважин и начальных условий"""
        # Сбрасываем потоки
        self.flow_rates_x = np.zeros((self.ny + 1, self.nx + 1))
        self.flow_rates_y = np.zeros((self.ny + 1, self.nx + 1))

        # Нагнетательные скважины
        for injector in self.wells_config['injectors']:
            pos_x = injector['position_x']
            pos_y = injector['position_y']
            rate = injector['rate']
            water_cut = injector['water_cut']

            # Устанавливаем водонасыщенность на нагнетательной скважине
            self.Sw_with_cap[:, pos_y, pos_x] = water_cut
            self.Sw_without_cap[:, pos_y, pos_x] = water_cut

            # Распределяем поток по 4 направлениям
            flow_per_direction = rate / 4.0

            # Потоки по X
            if pos_x < self.nx:
                self.flow_rates_x[pos_y, pos_x] += flow_per_direction  # Вправо
            if pos_x > 0:
                self.flow_rates_x[pos_y, pos_x - 1] -= flow_per_direction  # Влево

            # Потоки по Y
            if pos_y < self.ny:
                self.flow_rates_y[pos_y, pos_x] += flow_per_direction  # Вверх
            if pos_y > 0:
                self.flow_rates_y[pos_y - 1, pos_x] -= flow_per_direction  # Вниз

        # Добывающие скважины
        for producer in self.wells_config['producers']:
            pos_x = producer['position_x']
            pos_y = producer['position_y']
            rate = producer['rate']

            # Распределяем поток по 4 направлениям
            flow_per_direction = rate / 4.0

            # Потоки по X
            if pos_x < self.nx:
                self.flow_rates_x[pos_y, pos_x] -= flow_per_direction  # Вправо
            if pos_x > 0:
                self.flow_rates_x[pos_y, pos_x - 1] += flow_per_direction  # Влево

            # Потоки по Y
            if pos_y < self.ny:
                self.flow_rates_y[pos_y, pos_x] -= flow_per_direction  # Вверх
            if pos_y > 0:
                self.flow_rates_y[pos_y - 1, pos_x] += flow_per_direction  # Вниз

    def is_well_node(self, i, j):
        """Проверка, является ли узел (i,j) скважиной"""
        # Проверяем нагнетательные скважины
        for injector in self.wells_config['injectors']:
            if i == injector['position_x'] and j == injector['position_y']:
                return True

        # Проверяем добывающие скважины
        for producer in self.wells_config['producers']:
            if i == producer['position_x'] and j == producer['position_y']:
                return True

        return False

    def calculate_well_connections(self):
        """Расчет связей между скважинами для визуализации трубочек"""
        connections = []

        # Получаем все скважины
        all_wells = []
        for idx, inj in enumerate(self.wells_config['injectors']):
            all_wells.append({
                'index': idx,
                'type': 'injector',
                'x': inj['position_x'] * self.dx,
                'y': inj['position_y'] * self.dy,
                'rate': inj['rate']
            })

        for idx, prod in enumerate(self.wells_config['producers']):
            all_wells.append({
                'index': len(self.wells_config['injectors']) + idx,
                'type': 'producer',
                'x': prod['position_x'] * self.dx,
                'y': prod['position_y'] * self.dy,
                'rate': prod['rate']
            })

        # Создаем соединения между нагнетательными и добывающими скважинами
        for well1 in all_wells:
            if well1['type'] == 'injector':
                for well2 in all_wells:
                    if well2['type'] == 'producer':
                        # Рассчитываем расстояние между скважинами
                        distance = np.sqrt((well1['x'] - well2['x']) ** 2 + (well1['y'] - well2['y']) ** 2)

                        # Сила связи обратно пропорциональна расстоянию
                        strength = (well1['rate'] * well2['rate']) / (distance + 1.0)

                        # Добавляем связь, если она достаточно сильная
                        if strength > 0.1:
                            connections.append({
                                'from': well1['index'],
                                'to': well2['index'],
                                'strength': strength,
                                'distance': distance
                            })

        return connections

    def run_simulation(self):
        """Запуск моделирования в двумерном пространстве"""
        print("Запуск двумерного моделирования с учетом капиллярных эффектов...")

        # Моделирование с учетом капиллярных эффектов
        for n in range(self.nt - 1):
            # Обновляем граничные условия для скважин
            self.apply_well_conditions(n)

            for j in range(1, self.ny):
                for i in range(1, self.nx):
                    # Пропускаем узлы со скважинами
                    if self.is_well_node(i, j):
                        continue

                    # Получаем текущие значения насыщенности
                    sw_current = self.Sw_with_cap[n, j, i]

                    # Расчет для соседних ячеек
                    sw_left = self.Sw_with_cap[n, j, i - 1]
                    sw_right = self.Sw_with_cap[n, j, i + 1] if i < self.nx - 1 else sw_current
                    sw_bottom = self.Sw_with_cap[n, j - 1, i]
                    sw_top = self.Sw_with_cap[n, j + 1, i] if j < self.ny - 1 else sw_current

                    # Фракционный поток в текущей и соседних ячейках
                    f_current = self.fractional_flow(sw_current)
                    f_left = self.fractional_flow(sw_left)
                    f_right = self.fractional_flow(sw_right)
                    f_bottom = self.fractional_flow(sw_bottom)
                    f_top = self.fractional_flow(sw_top)

                    # Направление потока по X и Y
                    flow_x = self.flow_rates_x[j, i] - self.flow_rates_x[j, i - 1]
                    flow_y = self.flow_rates_y[j, i] - self.flow_rates_y[j - 1, i]

                    # Конвективный перенос (учитываем направление потока)
                    conv_x = 0.0
                    if self.flow_rates_x[j, i] > 0:  # поток слева направо
                        conv_x += self.flow_rates_x[j, i] * f_current
                    else:  # поток справа налево
                        conv_x += self.flow_rates_x[j, i] * f_right

                    if self.flow_rates_x[j, i - 1] > 0:  # поток слева направо
                        conv_x -= self.flow_rates_x[j, i - 1] * f_left
                    else:  # поток справа налево
                        conv_x -= self.flow_rates_x[j, i - 1] * f_current

                    conv_y = 0.0
                    if self.flow_rates_y[j, i] > 0:  # поток снизу вверх
                        conv_y += self.flow_rates_y[j, i] * f_current
                    else:  # поток сверху вниз
                        conv_y += self.flow_rates_y[j, i] * f_top

                    if self.flow_rates_y[j - 1, i] > 0:  # поток снизу вверх
                        conv_y -= self.flow_rates_y[j - 1, i] * f_bottom
                    else:  # поток сверху вниз
                        conv_y -= self.flow_rates_y[j - 1, i] * f_current

                    # Диффузионный член (капиллярные эффекты)
                    # Увеличиваем коэффициент диффузии для лучшего распространения по пласту
                    D_i_j = self.diffusion_coefficient(sw_current) * 5.0  # Увеличиваем в 5 раз

                    diff_term = D_i_j * (
                            (sw_right - 2 * sw_current + sw_left) / (self.dx ** 2) +
                            (sw_top - 2 * sw_current + sw_bottom) / (self.dy ** 2)
                    )

                    # Обновляем насыщенность с учетом как конвективного, так и диффузионного переноса
                    self.Sw_with_cap[n + 1, j, i] = sw_current + self.dt * (
                            -conv_x / self.dx - conv_y / self.dy + diff_term
                    )

                    # Ограничиваем значения насыщенности допустимым диапазоном
                    self.Sw_with_cap[n + 1, j, i] = max(self.initial_water_saturation,
                                                        min(1.0 - self.residual_oil_saturation,
                                                            self.Sw_with_cap[n + 1, j, i]))

            # Распространение флюида от скважин к соседним ячейкам (дополнительное усиление)
            for injector in self.wells_config['injectors']:
                pos_x = injector['position_x']
                pos_y = injector['position_y']
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni = pos_x + di
                        nj = pos_y + dj
                        if 0 <= ni < self.nx and 0 <= nj < self.ny:
                            # Распространяем воду от нагнетательных скважин
                            distance = np.sqrt(di ** 2 + dj ** 2)
                            if distance > 0:
                                influence = 0.8 / (distance ** 2)
                                self.Sw_with_cap[n + 1, nj, ni] = max(
                                    self.Sw_with_cap[n + 1, nj, ni],
                                    self.Sw_with_cap[n + 1, pos_y, pos_x] * influence +
                                    self.Sw_with_cap[n + 1, nj, ni] * (1 - influence)
                                )

            # Применяем граничные условия
            self.apply_boundary_conditions(n + 1, with_capillary=True)

        # Аналогичное моделирование без учета капиллярных эффектов
        # (упрощенная версия для сравнения)
        for n in range(self.nt - 1):
            # Копируем результаты из модели с капиллярными эффектами, но сглаживаем эффекты
            for j in range(1, self.ny):
                for i in range(1, self.nx):
                    if not self.is_well_node(i, j):
                        # Берем данные из модели с капиллярными эффектами, но с меньшим влиянием диффузии
                        self.Sw_without_cap[n + 1, j, i] = 0.7 * self.Sw_with_cap[n + 1, j, i] + 0.3 * \
                                                           self.Sw_without_cap[n, j, i]

            # Для нагнетательных скважин создаем более четкий фронт
            for injector in self.wells_config['injectors']:
                pos_x = injector['position_x']
                pos_y = injector['position_y']
                water_cut = injector['water_cut']
                # Устанавливаем насыщенность для нагнетательной скважины
                self.Sw_without_cap[n + 1, pos_y, pos_x] = water_cut

            # Применяем граничные условия
            self.apply_boundary_conditions(n + 1, with_capillary=False)

        print("Двумерное моделирование завершено.")

    def apply_well_conditions(self, time_step):
        """Применение условий на скважинах для конкретного временного шага"""
        # Нагнетательные скважины
        for injector in self.wells_config['injectors']:
            pos_x = injector['position_x']
            pos_y = injector['position_y']
            water_cut = injector['water_cut']

            # Устанавливаем водонасыщенность на нагнетательной скважине
            self.Sw_with_cap[time_step + 1, pos_y, pos_x] = water_cut
            self.Sw_without_cap[time_step + 1, pos_y, pos_x] = water_cut

    def apply_boundary_conditions(self, time_step, with_capillary=True):
        """Применение граничных условий для двумерной модели"""
        # Выбираем массив в зависимости от модели
        Sw = self.Sw_with_cap if with_capillary else self.Sw_without_cap

        # Граничные условия по X (нулевой градиент)
        Sw[time_step, :, 0] = Sw[time_step, :, 1]
        Sw[time_step, :, -1] = Sw[time_step, :, -2]

        # Граничные условия по Y (нулевой градиент)
        Sw[time_step, 0, :] = Sw[time_step, 1, :]
        Sw[time_step, -1, :] = Sw[time_step, -2, :]

    def calculate_recovery_factor(self):
        """Расчет коэффициента нефтеотдачи для двумерной модели"""
        initial_oil = 1 - self.initial_water_saturation

        recovery_with_cap = np.zeros(self.nt)
        recovery_without_cap = np.zeros(self.nt)

        for n in range(self.nt):
            # Средняя нефтенасыщенность по всему пласту
            avg_oil_with_cap = 1 - np.mean(self.Sw_with_cap[n])
            avg_oil_without_cap = 1 - np.mean(self.Sw_without_cap[n])

            # Коэффициент нефтеотдачи
            recovery_with_cap[n] = (initial_oil - avg_oil_with_cap) / initial_oil
            recovery_without_cap[n] = (initial_oil - avg_oil_without_cap) / initial_oil

        return recovery_with_cap, recovery_without_cap

    def get_breakthrough_time(self):
        """Определение времени прорыва воды"""
        threshold = self.initial_water_saturation + 0.05

        # Ищем время прорыва в каждой добывающей скважине
        breakthrough_with_cap = self.days
        breakthrough_without_cap = self.days

        for producer in self.wells_config['producers']:
            pos_x = producer['position_x']
            pos_y = producer['position_y']

            # Время прорыва с учетом капиллярных эффектов
            for n in range(self.nt):
                if self.Sw_with_cap[n, pos_y, pos_x] > threshold:
                    breakthrough_with_cap = min(breakthrough_with_cap, self.t[n])
                    break

            # Время прорыва без учета капиллярных эффектов
            for n in range(self.nt):
                if self.Sw_without_cap[n, pos_y, pos_x] > threshold:
                    breakthrough_without_cap = min(breakthrough_without_cap, self.t[n])
                    break

        return breakthrough_with_cap, breakthrough_without_cap

    def get_well_production_data(self):
        """Получение данных о добыче из скважин"""
        production_data = {
            'time': self.t,
            'wells': []
        }

        # Данные для добывающих скважин
        for i, producer in enumerate(self.wells_config['producers']):
            pos_x = producer['position_x']
            pos_y = producer['position_y']
            rate = producer['rate']

            # Инициализируем массивы для данных
            water_rate = np.zeros(self.nt)
            oil_rate = np.zeros(self.nt)

            # Расчет дебитов для каждого временного шага
            for n in range(self.nt):
                # Обводненность в точке добывающей скважины
                water_cut_with_cap = self.Sw_with_cap[n, pos_y, pos_x]

                # Дебит воды и нефти
                water_rate[n] = rate * water_cut_with_cap
                oil_rate[n] = rate * (1 - water_cut_with_cap)

            # Добавляем данные скважины
            well_data = {
                'name': f'Producer_{i + 1}',
                'position_x': pos_x * self.dx,
                'position_y': pos_y * self.dy,
                'water_rate': water_rate,
                'oil_rate': oil_rate,
                'total_water_produced': np.trapz(water_rate, self.t),
                'total_oil_produced': np.trapz(oil_rate, self.t)
            }
            production_data['wells'].append(well_data)

        return production_data

    def plot_wells_location(self):
        """Отображение расположения скважин на 2D пласте"""
        plt.figure(figsize=(12, 10))

        # Контур пласта
        plt.plot([0, self.length_x, self.length_x, 0, 0],
                 [0, 0, self.length_y, self.length_y, 0], 'k-', linewidth=2)

        # Рисуем нагнетательные скважины (синие треугольники)
        for injector in self.wells_config['injectors']:
            x_pos = injector['position_x'] * self.dx
            y_pos = injector['position_y'] * self.dy
            rate = injector['rate']
            plt.plot(x_pos, y_pos, 'b^', markersize=12)
            plt.text(x_pos, y_pos + 30, f'Инж. {rate:.1f} м³/день',
                     ha='center', va='bottom', color='blue')

        # Рисуем добывающие скважины (красные треугольники)
        for producer in self.wells_config['producers']:
            x_pos = producer['position_x'] * self.dx
            y_pos = producer['position_y'] * self.dy
            rate = producer['rate']
            plt.plot(x_pos, y_pos, 'rv', markersize=12)
            plt.text(x_pos, y_pos - 30, f'Доб. {rate:.1f} м³/день',
                     ha='center', va='top', color='red')

        # Рисуем "трубочки" между скважинами
        for conn in self.well_connections:
            start_well = None
            end_well = None

            # Находим начальную скважину (инжектор)
            for i, injector in enumerate(self.wells_config['injectors']):
                if i == conn['from']:
                    start_well = (injector['position_x'] * self.dx, injector['position_y'] * self.dy)
                    break

            # Находим конечную скважину (продюсер)
            for i, producer in enumerate(self.wells_config['producers']):
                if i + len(self.wells_config['injectors']) == conn['to']:
                    end_well = (producer['position_x'] * self.dx, producer['position_y'] * self.dy)
                    break

            if start_well and end_well:
                # Толщина линии пропорциональна силе связи
                linewidth = max(1.0, conn['strength'] * 2)

                # Рисуем линию
                plt.plot([start_well[0], end_well[0]], [start_well[1], end_well[1]],
                         'k-', alpha=0.6, linewidth=linewidth)

                # Интерполируем точки вдоль "трубочки"
                num_points = 5
                for i in range(1, num_points):
                    t = i / num_points
                    x = start_well[0] * (1 - t) + end_well[0] * t
                    y = start_well[1] * (1 - t) + end_well[1] * t
                    size = 6 * conn['strength'] * (1 - abs(2 * t - 1))
                    plt.plot(x, y, 'o', markersize=size, alpha=0.4, color='lightblue')

        plt.xlim(-50, self.length_x + 50)
        plt.ylim(-50, self.length_y + 50)
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.title('Расположение скважин на двумерном пласте с интерполяцией потоков')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('wells_location_2d.png')
        plt.close()



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
        # Сбрасываем потоки
        self.flow_rates_x = np.zeros((self.ny + 1, self.nx + 1))
        self.flow_rates_y = np.zeros((self.ny + 1, self.nx + 1))

        # Создаем базовое поле потоков для более естественного распространения
        for j in range(1, self.ny):
            for i in range(1, self.nx):
                # Добавляем небольшой случайный фоновый поток (шум)
                self.flow_rates_x[j, i] = np.random.uniform(-0.05, 0.05)
                self.flow_rates_y[j, i] = np.random.uniform(-0.05, 0.05)

        # Нагнетательные скважины
        for injector in self.wells_config['injectors']:
            pos_x = injector['position_x']
            pos_y = injector['position_y']
            rate = injector['rate']
            water_cut = injector['water_cut']

            # Устанавливаем водонасыщенность на нагнетательной скважине
            self.Sw_with_cap[:, pos_y, pos_x] = water_cut
            self.Sw_without_cap[:, pos_y, pos_x] = water_cut

            # Распределяем потоки от скважины во всех направлениях
            # Используем более плавное распределение для более реалистичного моделирования
            radius = 10  # Радиус влияния скважины
            total_weight = 0

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni = pos_x + di
                    nj = pos_y + dj

                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        # Рассчитываем вес на основе расстояния от скважины
                        distance = np.sqrt(di ** 2 + dj ** 2)
                        if distance == 0:
                            continue  # Пропускаем саму скважину

                        weight = 1.0 / (distance ** 2)
                        total_weight += weight

            # Теперь распределяем потоки с учетом нормализации
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni = pos_x + di
                    nj = pos_y + dj

                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        distance = np.sqrt(di ** 2 + dj ** 2)
                        if distance == 0:
                            continue  # Пропускаем саму скважину

                        weight = 1.0 / (distance ** 2)
                        normalized_weight = weight / total_weight
                        flow_value = rate * normalized_weight

                        # Распределяем поток в направлении от скважины
                        if di > 0:
                            self.flow_rates_x[nj, ni - 1] += flow_value * abs(di) / distance
                        elif di < 0:
                            self.flow_rates_x[nj, ni] -= flow_value * abs(di) / distance

                        if dj > 0:
                            self.flow_rates_y[nj - 1, ni] += flow_value * abs(dj) / distance
                        elif dj < 0:
                            self.flow_rates_y[nj, ni] -= flow_value * abs(dj) / distance

        # Добывающие скважины
        for producer in self.wells_config['producers']:
            pos_x = producer['position_x']
            pos_y = producer['position_y']
            rate = producer['rate']

            # Распределяем потоки к скважине из всех направлений
            radius = 10  # Радиус влияния скважины
            total_weight = 0

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni = pos_x + di
                    nj = pos_y + dj

                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        distance = np.sqrt(di ** 2 + dj ** 2)
                        if distance == 0:
                            continue  # Пропускаем саму скважину

                        weight = 1.0 / (distance ** 2)
                        total_weight += weight

            # Теперь распределяем потоки с учетом нормализации
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni = pos_x + di
                    nj = pos_y + dj

                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        distance = np.sqrt(di ** 2 + dj ** 2)
                        if distance == 0:
                            continue  # Пропускаем саму скважину

                        weight = 1.0 / (distance ** 2)
                        normalized_weight = weight / total_weight
                        flow_value = rate * normalized_weight

                        # Распределяем поток в направлении к скважине
                        if di > 0:
                            self.flow_rates_x[nj, ni - 1] -= flow_value * abs(di) / distance
                        elif di < 0:
                            self.flow_rates_x[nj, ni] += flow_value * abs(di) / distance

                        if dj > 0:
                            self.flow_rates_y[nj - 1, ni] -= flow_value * abs(dj) / distance
                        elif dj < 0:
                            self.flow_rates_y[nj, ni] += flow_value * abs(dj) / distance

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