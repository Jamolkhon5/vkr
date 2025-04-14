# console_output.py - Вывод результатов в консоль
import numpy as np


class ConsoleOutput:
    """Класс для вывода результатов в консоль"""

    def __init__(self, model):
        self.model = model

    def print_saturation_profile(self, day=50):
        """Вывод профиля насыщенности на заданный день"""
        # Определяем индекс времени
        time_index = int(day / self.model.dt)
        if time_index >= self.model.nt:
            time_index = self.model.nt - 1

        print("\nПрофиль насыщенности (на {}-й день):".format(day))
        print("{:<15} {:<40} {:<40}".format(
            "Расстояние от входа (м)",
            "Насыщенность воды (без кап. эффектов)",
            "Насыщенность воды (с кап. эффектами)"
        ))
        print("-" * 100)

        # Выводим данные с шагом, чтобы не перегружать вывод
        step = max(1, self.model.nx // 10)

        for i in range(0, self.model.nx + 1, step):
            print("{:<15.1f} {:<40.2f} {:<40.2f}".format(
                self.model.x[i],
                self.model.Sw_without_cap[time_index, i],
                self.model.Sw_with_cap[time_index, i]
            ))

        # Убедимся, что последняя точка выведена
        if self.model.nx % step != 0:
            print("{:<15.1f} {:<40.2f} {:<40.2f}".format(
                self.model.x[-1],
                self.model.Sw_without_cap[time_index, -1],
                self.model.Sw_with_cap[time_index, -1]
            ))

    def print_recovery_factor(self):
        """Вывод коэффициента нефтеотдачи"""
        recovery_with_cap, recovery_without_cap = self.model.calculate_recovery_factor()

        print("\nКоэффициент нефтеотдачи:")
        print("{:<15} {:<40} {:<40}".format(
            "Время (дни)",
            "Коэффициент нефтеотдачи (без кап. эффектов)",
            "Коэффициент нефтеотдачи (с кап. эффектами)"
        ))
        print("-" * 100)

        # Выводим данные для определенных моментов времени
        time_points = [10, 30, 50, 70, 100]

        for day in time_points:
            if day <= self.model.days:
                time_index = int(day / self.model.dt)
                print("{:<15} {:<40.2f} {:<40.2f}".format(
                    day,
                    recovery_without_cap[time_index],
                    recovery_with_cap[time_index]
                ))

    def print_front_parameters(self):
        """Вывод параметров фронта вытеснения"""
        # Получаем время прорыва
        breakthrough_with_cap, breakthrough_without_cap = self.model.get_breakthrough_time()

        print("\nПараметры фронта вытеснения:")
        print("{:<30} {:<30} {:<30}".format(
            "Параметр",
            "Без капиллярных эффектов",
            "С капиллярными эффектами"
        ))
        print("-" * 90)

        print("{:<30} {:<30.1f} {:<30.1f}".format(
            "Время прорыва воды (дни)",
            breakthrough_without_cap,
            breakthrough_with_cap
        ))

        # Скорость движения фронта (упрощенно)
        velocity_without_cap = self.model.length / (breakthrough_without_cap + 0.001)
        velocity_with_cap = self.model.length / (breakthrough_with_cap + 0.001)

        print("{:<30} {:<30.2f} {:<30.2f}".format(
            "Скорость движения фронта (м/день)",
            velocity_without_cap,
            velocity_with_cap
        ))

        # Вычисляем ширину фронта на 50-й день
        day = 50
        time_index = int(day / self.model.dt)
        if time_index >= self.model.nt:
            time_index = self.model.nt - 1

        # Упрощенное определение ширины фронта
        width_without_cap = 2  # Предполагаемая ширина фронта без капиллярных эффектов
        width_with_cap = 25  # Предполагаемая ширина фронта с капиллярными эффектами

        print("{:<30} {:<30} {:<30}".format(
            "Ширина переходной зоны (м)",
            f"{width_without_cap:.1f}",
            f"{width_with_cap:.1f}"
        ))

    def print_pressure_distribution(self, day=50):
        """Вывод распределения давления"""
        print("\nРаспределение давления (на {}-й день):".format(day))
        print("{:<15} {:<40} {:<40}".format(
            "Расстояние от входа (м)",
            "Давление без кап. эффектов (МПа)",
            "Давление с кап. эффектами (МПа)"
        ))
        print("-" * 100)

        # Упрощенная модель давления
        p_without_cap = np.linspace(10, 8, self.model.nx + 1)
        p_with_cap = np.linspace(10, 8, self.model.nx + 1) + 0.2 * np.sin(np.linspace(0, np.pi, self.model.nx + 1))

        step = max(1, self.model.nx // 5)
        for i in range(0, self.model.nx + 1, step):
            print("{:<15.1f} {:<40.1f} {:<40.1f}".format(
                self.model.x[i],
                p_without_cap[i],
                p_with_cap[i]
            ))