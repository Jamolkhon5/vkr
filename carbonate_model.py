# carbonate_model.py - Специализированная модель для карбонатных коллекторов
import numpy as np
from model import OilFiltrationModel
from data_loader import DataLoader


class CarbonateModel(OilFiltrationModel):
    """
    Расширенная модель для карбонатных коллекторов с учетом
    двойной пористости и детального моделирования капиллярных эффектов
    """

    def __init__(self, rock_type="Limestone"):
        super().__init__()

        # Загрузка данных из файлов
        self.data_loader = DataLoader()
        self.data_loader.load_all_data()

        # Установка параметров из загруженных данных
        self.set_parameters_from_data(rock_type)

        # Дополнительные параметры для карбонатных коллекторов
        self.fracture_porosity = 0.01  # Пористость трещин
        self.matrix_porosity = self.porosity - self.fracture_porosity  # Пористость матрицы
        self.fracture_permeability = 100.0  # Проницаемость трещин, мД
        self.matrix_permeability = 0.1  # Проницаемость матрицы, мД

        # Параметры для моделирования двойной пористости
        # Увеличиваем форм-фактор для лучшего обмена между трещинами и матрицей в длинном пласте
        self.shape_factor = 0.3  # Форм-фактор для обмена между трещинами и матрицей (изменено с 0.1)

        # Массивы для хранения результатов для матрицы и трещин
        self.Sw_matrix = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation
        self.Sw_fracture = np.ones((self.nt, self.nx + 1)) * self.initial_water_saturation

        # Устанавливаем граничные условия
        self.Sw_matrix[:, 0] = 0.8
        self.Sw_fracture[:, 0] = 0.8

    def set_parameters_from_data(self, rock_type):
        """Установка параметров модели из реальных данных"""
        # Получение усредненных петрофизических параметров
        avg_params = self.data_loader.get_average_parameters(rock_type)
        if avg_params:
            self.porosity = avg_params['porosity']

            # Проницаемость переводим из мД в Дарси
            k_md = avg_params['permeability']
            self.permeability = k_md * 9.869233e-4  # мД в Дарси

            # Параметр смачиваемости
            self.wettability_factor = avg_params['wettability_index']

        # Получение параметров модели Брукса-Кори
        bc_params = self.data_loader.get_brooks_corey_params(rock_type)
        if bc_params and len(bc_params) > 0:
            # Берем первый доступный образец
            sample_id = list(bc_params.keys())[0]
            params = bc_params[sample_id]

            self.entry_pressure = params['entry_pressure']
            self.pore_distribution_index = params['pore_distribution_index']
            self.initial_water_saturation = params['initial_water_saturation']

        # Получение PVT-свойств флюидов
        pvt_props = self.data_loader.get_pvt_properties()
        if pvt_props:
            self.mu_oil = pvt_props['oil_viscosity']
            self.mu_water = pvt_props['water_viscosity']

    def run_dual_porosity_simulation(self):
        """Запуск моделирования с учетом двойной пористости"""
        print("Запуск моделирования карбонатного коллектора с двойной пористостью...")

        # Моделирование течения в трещинах (быстрое течение)
        for n in range(self.nt - 1):
            for i in range(1, self.nx):
                # Апвинд схема для конвективного члена в трещинах
                f_i = self.fractional_flow(self.Sw_fracture[n, i])
                f_im1 = self.fractional_flow(self.Sw_fracture[n, i - 1])

                # Схема апвинд для трещин (без капиллярных эффектов в трещинах)
                self.Sw_fracture[n + 1, i] = self.Sw_fracture[n, i] - \
                                             (self.dt / self.dx) * (f_i - f_im1) + \
                                             self.dt * self.transfer_term(n, i)

            # Граничное условие на правом конце
            self.Sw_fracture[n + 1, -1] = self.Sw_fracture[n + 1, -2]

        # Моделирование течения в матрице (медленное течение с капиллярными эффектами)
        for n in range(self.nt - 1):
            for i in range(1, self.nx):
                # Капиллярное давление в матрице
                pc_gradient = self.matrix_capillary_gradient(n, i)

                # Обновление насыщенности в матрице
                self.Sw_matrix[n + 1, i] = self.Sw_matrix[n, i] + \
                                           self.dt * pc_gradient - \
                                           self.dt * self.transfer_term(n, i)

            # Граничное условие на правом конце
            self.Sw_matrix[n + 1, -1] = self.Sw_matrix[n + 1, -2]

        # Вычисление итоговой насыщенности как взвешенного среднего
        matrix_volume = self.matrix_porosity / self.porosity
        fracture_volume = self.fracture_porosity / self.porosity

        self.Sw_with_cap = matrix_volume * self.Sw_matrix + fracture_volume * self.Sw_fracture

        # Запускаем обычное моделирование для сравнения (без капиллярных эффектов)
        super().run_simulation()

        print("Моделирование карбонатного коллектора завершено.")

    def transfer_term(self, n, i):
        """Расчет обмена флюидами между трещинами и матрицей"""
        # Разница в капиллярном давлении
        pc_matrix = self.capillary_pressure(self.Sw_matrix[n, i])
        pc_fracture = 0  # В трещинах капиллярное давление принимаем равным нулю

        # Скорость обмена пропорциональна разнице давлений и коэффициенту обмена
        exchange_rate = self.shape_factor * (pc_fracture - pc_matrix)

        return exchange_rate

    def matrix_capillary_gradient(self, n, i):
        """Расчет градиента капиллярного давления в матрице"""
        # Рассчитываем градиент капиллярного давления
        if i == 0:
            # Граничное условие слева
            pc_grad = (self.capillary_pressure(self.Sw_matrix[n, i + 1]) -
                       self.capillary_pressure(self.Sw_matrix[n, i])) / self.dx
        elif i == self.nx:
            # Граничное условие справа
            pc_grad = (self.capillary_pressure(self.Sw_matrix[n, i]) -
                       self.capillary_pressure(self.Sw_matrix[n, i - 1])) / self.dx
        else:
            # Внутренние точки
            pc_grad = (self.capillary_pressure(self.Sw_matrix[n, i + 1]) -
                       self.capillary_pressure(self.Sw_matrix[n, i - 1])) / (2 * self.dx)

        # Коэффициент мобильности
        mobility = self.matrix_permeability / (self.mu_water * self.matrix_porosity)

        return mobility * pc_grad

    def calculate_recovery_factor(self):
        """Расчет коэффициента нефтеотдачи с учетом двойной пористости"""
        initial_oil = 1 - self.initial_water_saturation

        recovery_with_cap = np.zeros(self.nt)
        recovery_without_cap = np.zeros(self.nt)

        # Вычисляем взвешенное среднее для матрицы и трещин
        matrix_volume = self.matrix_porosity / self.porosity
        fracture_volume = self.fracture_porosity / self.porosity

        for n in range(self.nt):
            # Средняя нефтенасыщенность для двойной пористости
            avg_oil_matrix = 1 - np.mean(self.Sw_matrix[n, :])
            avg_oil_fracture = 1 - np.mean(self.Sw_fracture[n, :])
            avg_oil_with_cap = matrix_volume * avg_oil_matrix + fracture_volume * avg_oil_fracture

            # Средняя нефтенасыщенность без учета капиллярных эффектов
            avg_oil_without_cap = 1 - np.mean(self.Sw_without_cap[n, :])

            # Коэффициент нефтеотдачи
            recovery_with_cap[n] = (initial_oil - avg_oil_with_cap) / initial_oil
            recovery_without_cap[n] = (initial_oil - avg_oil_without_cap) / initial_oil

        return recovery_with_cap, recovery_without_cap

    def get_breakthrough_time(self):
        """Определение времени прорыва воды с учетом двойной пористости"""
        threshold = self.initial_water_saturation + 0.05

        # Используем взвешенное среднее для матрицы и трещин
        matrix_volume = self.matrix_porosity / self.porosity
        fracture_volume = self.fracture_porosity / self.porosity

        # Время прорыва с учетом капиллярных эффектов
        breakthrough_with_cap = self.days
        for n in range(self.nt):
            # Вычисляем среднюю насыщенность на выходе из пласта
            avg_Sw_exit = matrix_volume * self.Sw_matrix[n, -1] + fracture_volume * self.Sw_fracture[n, -1]
            if avg_Sw_exit > threshold:
                breakthrough_with_cap = self.t[n]
                break

        # Время прорыва без учета капиллярных эффектов
        breakthrough_without_cap = self.days
        for n in range(self.nt):
            if self.Sw_without_cap[n, -1] > threshold:
                breakthrough_without_cap = self.t[n]
                break

        return breakthrough_with_cap, breakthrough_without_cap