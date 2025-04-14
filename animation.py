# animation.py - Анимация фильтрации нефти
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge, Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import os
import sys


class OilFiltrationAnimation:
    """Класс для создания анимации фильтрации нефти"""

    def __init__(self, model, fps=15, dpi=100, save_path='animation'):
        self.model = model
        self.fps = fps  # Кадров в секунду
        self.dpi = dpi  # Разрешение анимации
        self.save_path = save_path

        # Создаем директорию для сохранения анимаций, если её нет
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Создаем кастомную цветовую карту (нефть - коричневая, вода - синяя)
        self.cmap = LinearSegmentedColormap.from_list(
            'oil_water',
            [(0.6, 0.3, 0, 1.0),  # коричневый (нефть)
             (0.0, 0.0, 0.7, 1.0)]  # синий (вода)
        )

        # Проверяем доступность writer'ов для анимации
        self.available_writers = animation.writers.list()
        if 'ffmpeg' in self.available_writers:
            self.default_writer = 'ffmpeg'
            self.default_ext = '.mp4'
        elif 'imagemagick' in self.available_writers:
            self.default_writer = 'imagemagick'
            self.default_ext = '.gif'
        else:
            # Если специальных writer'ов нет, будем сохранять отдельные кадры
            self.default_writer = None
            self.default_ext = '.png'

        print(f"Используемый инструмент для анимации: {self.default_writer or 'Отдельные кадры'}")

    def animate_saturation(self, frames=100, with_capillary=True):
        """
        Создание анимации изменения водонасыщенности

        Parameters:
        -----------
        frames : int
            Количество кадров для анимации
        with_capillary : bool
            Использовать данные с учетом капиллярных эффектов (True)
            или без них (False)
        """
        print(
            f"Создание анимации водонасыщенности {'с учетом' if with_capillary else 'без учета'} капиллярных эффектов...")

        # Выбираем данные в зависимости от флага
        if with_capillary:
            saturation_data = self.model.Sw_with_cap
            title_prefix = "С учетом капиллярных эффектов"
            base_filename = "saturation_with_capillary"
        else:
            saturation_data = self.model.Sw_without_cap
            title_prefix = "Без учета капиллярных эффектов"
            base_filename = "saturation_without_capillary"

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 6))

        # Подготавливаем первый кадр
        im = ax.imshow(
            saturation_data[0].reshape(1, -1),
            aspect='auto',
            extent=[0, self.model.length, -50, 50],
            cmap=self.cmap,
            vmin=0,
            vmax=1,
            interpolation='bicubic'
        )

        # Добавляем колонтитул
        colorbar = plt.colorbar(im, ax=ax)
        colorbar.set_label('Водонасыщенность')

        # Проверяем, имеет ли модель скважины
        has_wells = hasattr(self.model, 'wells_config')

        # Рисуем пласт и границы
        ax.add_patch(Rectangle((0, -50), self.model.length, 100, fill=False, edgecolor='k', linewidth=2))

        # Добавляем скважины, если они есть
        wells_artists = []
        if has_wells:
            # Нагнетательные скважины (синие треугольники)
            for injector in self.model.wells_config['injectors']:
                pos = injector['position']
                rate = injector['rate']
                x_pos = pos * self.model.dx
                wedge = Wedge((x_pos, 0), 15, 0, 360, width=2, color='blue', alpha=0.7)
                ax.add_patch(wedge)
                wells_artists.append(wedge)
                ax.text(x_pos, 30, f'Инж. {rate:.1f} м³/д', ha='center', va='center', fontsize=8, color='blue')

            # Добывающие скважины (красные треугольники)
            for producer in self.model.wells_config['producers']:
                pos = producer['position']
                rate = producer['rate']
                x_pos = pos * self.model.dx
                circle = Circle((x_pos, 0), 15, color='red', alpha=0.7)
                ax.add_patch(circle)
                wells_artists.append(circle)
                ax.text(x_pos, -30, f'Доб. {rate:.1f} м³/д', ha='center', va='center', fontsize=8, color='red')

        # Настройка осей
        ax.set_xlim(0, self.model.length)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('Расстояние (м)')
        ax.set_title(f'Анимация фильтрации нефти\n{title_prefix}')

        # Отключаем метки на оси Y
        ax.set_yticks([])

        # Добавляем сетку
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Текст с информацией о времени
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

        # Шаг времени для анимации
        time_step = max(1, self.model.nt // frames)

        # Функция обновления для анимации
        def update(frame):
            time_idx = frame * time_step
            if time_idx >= self.model.nt:
                time_idx = self.model.nt - 1

            # Обновляем данные
            im.set_array(saturation_data[time_idx].reshape(1, -1))

            # Обновляем текст времени
            days = self.model.t[time_idx]
            time_text.set_text(f'Время: {days:.1f} дней')

            return [im, time_text] + wells_artists

        if self.default_writer:
            # Создаем анимацию
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=frames,
                interval=1000 / self.fps,
                blit=True
            )

            # Сохраняем анимацию
            output_path = os.path.join(self.save_path, base_filename + self.default_ext)
            try:
                anim.save(output_path, writer=self.default_writer, fps=self.fps, dpi=self.dpi)
                print(f"Анимация сохранена в {output_path}")
            except Exception as e:
                print(f"Ошибка при сохранении анимации: {e}")
                # Если ошибка при сохранении, сохраняем кадры
                self._save_frames(update, frames, base_filename)
        else:
            # Если нет доступных writer'ов, сохраняем кадры
            self._save_frames(update, frames, base_filename)

        plt.close(fig)
        return os.path.join(self.save_path, base_filename + (self.default_ext if self.default_writer else ""))

    def _save_frames(self, update_func, frames, base_filename):
        """
        Сохраняет отдельные кадры анимации вместо видео
        """
        print(f"Сохранение отдельных кадров (всего {frames})...")
        frames_dir = os.path.join(self.save_path, base_filename + "_frames")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        for frame in range(frames):
            # Обновляем фигуру
            update_func(frame)

            # Сохраняем текущий кадр
            frame_path = os.path.join(frames_dir, f"{base_filename}_frame_{frame:04d}.png")
            plt.savefig(frame_path, dpi=self.dpi)

            if frame % 10 == 0:
                print(f"Сохранен кадр {frame}/{frames}")

        print(f"Кадры сохранены в директории {frames_dir}")

    def animate_2d_field(self, frames=100, resolution=50):
        """
        Создание реалистичной 2D анимации поля фильтрации нефти
        с использованием интерполяции для создания двумерного представления

        Parameters:
        -----------
        frames : int
            Количество кадров для анимации
        resolution : int
            Разрешение сетки по оси Y (количество точек)
        """
        print("Создание реалистичной 2D анимации поля фильтрации нефти...")

        # Создаем фигуру большего размера для лучшего качества
        fig, ax = plt.subplots(figsize=(16, 9), dpi=self.dpi)

        # Данные водонасыщенности с учетом капиллярных эффектов
        saturation_data = self.model.Sw_with_cap

        # Создаем двумерную сетку для визуализации в виде поля
        nx = self.model.nx + 1
        ny = resolution

        # Определяем ширину пласта относительно длины
        field_width = self.model.length / 4  # Ширина пласта = 1/4 длины

        # Создаем координатные сетки
        x = np.linspace(0, self.model.length, nx)
        y = np.linspace(-field_width / 2, field_width / 2, ny)
        X, Y = np.meshgrid(x, y)

        # Создаем начальную сетку насыщенности
        Z = np.ones((ny, nx)) * self.model.initial_water_saturation

        # Вычисляем коэффициент затухания для краев пласта
        def attenuation_factor(y_pos):
            # Нормализованное расстояние от центра (0 в центре, 1 на краях)
            normalized_dist = abs(y_pos) / (field_width / 2)
            # Коэффициент затухания (1 в центре, 0.2 на краях)
            return 1.0 - 0.8 * normalized_dist

        # Инициализируем центральную линию данными из модели
        mid_y = ny // 2
        Z[mid_y] = saturation_data[0]

        # Распространяем начальные значения от центра к краям
        for j in range(ny):
            attenuate = attenuation_factor(y[j])
            Z[j] = self.model.initial_water_saturation + attenuate * (
                        saturation_data[0] - self.model.initial_water_saturation)

        # Подготавливаем первый кадр
        im = ax.imshow(
            Z,
            extent=[0, self.model.length, -field_width / 2, field_width / 2],
            cmap=self.cmap,
            vmin=0,
            vmax=1,
            interpolation='bicubic',
            aspect='auto'
        )

        # Добавляем колонтитул
        colorbar = plt.colorbar(im, ax=ax)
        colorbar.set_label('Водонасыщенность')

        # Проверяем, имеет ли модель скважины
        has_wells = hasattr(self.model, 'wells_config')

        # Визуализация скважин, если они есть
        well_markers = []
        if has_wells:
            # Определяем размер маркеров скважин
            well_radius = self.model.length / 40

            # Нагнетательные скважины (синие)
            for injector in self.model.wells_config['injectors']:
                pos = injector['position']
                rate = injector['rate']
                x_pos = pos * self.model.dx

                # Рисуем скважину как круг
                circle = plt.Circle((x_pos, 0), well_radius, color='blue', alpha=0.6, zorder=10)
                ax.add_patch(circle)
                well_markers.append(circle)

                # Добавляем текст с информацией о скважине
                text = ax.text(x_pos, -field_width / 8, f'Инж. {rate:.1f} м³/д',
                               ha='center', va='center', fontsize=9, color='blue',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
                well_markers.append(text)

            # Добывающие скважины (красные)
            for producer in self.model.wells_config['producers']:
                pos = producer['position']
                rate = producer['rate']
                x_pos = pos * self.model.dx

                # Рисуем скважину как круг
                circle = plt.Circle((x_pos, 0), well_radius, color='red', alpha=0.6, zorder=10)
                ax.add_patch(circle)
                well_markers.append(circle)

                # Добавляем текст с информацией о скважине
                text = ax.text(x_pos, field_width / 8, f'Доб. {rate:.1f} м³/д',
                               ha='center', va='center', fontsize=9, color='red',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
                well_markers.append(text)

        # Добавляем границы пласта
        ax.add_patch(Rectangle((0, -field_width / 2), self.model.length, field_width,
                               fill=False, edgecolor='black', linewidth=2, zorder=15))

        # Визуализация неоднородностей пласта
        def add_heterogeneity():
            # Добавляем некоторые линзы и неоднородности для реализма
            # Это чисто визуальный эффект для анимации, не влияющий на расчеты

            # Несколько линз разного размера и формы
            lens_patches = []

            # Параметры линз
            lens_centers = [
                (self.model.length * 0.2, field_width * 0.1),
                (self.model.length * 0.5, -field_width * 0.15),
                (self.model.length * 0.8, field_width * 0.05)
            ]
            lens_sizes = [
                (self.model.length * 0.15, field_width * 0.1),
                (self.model.length * 0.1, field_width * 0.15),
                (self.model.length * 0.12, field_width * 0.08)
            ]
            lens_angles = [30, -15, 10]

            for (cx, cy), (w, h), angle in zip(lens_centers, lens_sizes, lens_angles):
                ellipse = Ellipse(
                    (cx, cy), w, h, angle=angle,
                    fill=True, edgecolor='gray', facecolor='none',
                    linestyle='--', linewidth=1, alpha=0.5, zorder=5
                )
                ax.add_patch(ellipse)
                lens_patches.append(ellipse)

            # Несколько линий разломов
            fault_lines = []

            # Параметры разломов
            fault_start_points = [
                (self.model.length * 0.3, -field_width * 0.4),
                (self.model.length * 0.7, field_width * 0.35)
            ]
            fault_end_points = [
                (self.model.length * 0.45, field_width * 0.4),
                (self.model.length * 0.6, -field_width * 0.2)
            ]

            for (x1, y1), (x2, y2) in zip(fault_start_points, fault_end_points):
                line = Line2D([x1, x2], [y1, y2],
                              color='black', linestyle='--', linewidth=1, alpha=0.6, zorder=5)
                ax.add_line(line)
                fault_lines.append(line)

            return lens_patches + fault_lines

        # Добавляем неоднородности
        heterogeneity_elements = add_heterogeneity()

        # Настройка осей
        ax.set_xlim(0, self.model.length)
        ax.set_ylim(-field_width / 2, field_width / 2)
        ax.set_xlabel('Расстояние (м)')
        ax.set_ylabel('Поперечное расстояние (м)')
        ax.set_title('Анимация фильтрации нефти - 2D моделирование пласта')

        # Добавляем сетку
        ax.grid(True, linestyle='--', alpha=0.3)

        # Текст с информацией о времени
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Информация о модели
        model_info = f"Модель: {type(self.model).__name__}\n"
        model_info += f"Пористость: {self.model.porosity:.2f}, "

        if hasattr(self.model, 'matrix_porosity'):
            model_info += f"Матричная пористость: {self.model.matrix_porosity:.2f}, "
            model_info += f"Пористость трещин: {self.model.fracture_porosity:.2f}"
        else:
            model_info += f"Начальная водонасыщенность: {self.model.initial_water_saturation:.2f}"

        # Добавляем информацию о модели
        model_text = ax.text(0.5, 0.02, model_info, transform=ax.transAxes,
                             fontsize=9, ha='center', va='bottom',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Шаг времени для анимации
        time_step = max(1, self.model.nt // frames)

        # Функция обновления для анимации
        def update(frame):
            time_idx = frame * time_step
            if time_idx >= self.model.nt:
                time_idx = self.model.nt - 1

            # Обновляем данные центральной линии
            Z[mid_y] = saturation_data[time_idx]

            # Распространяем значения от центра к краям
            for j in range(ny):
                attenuate = attenuation_factor(y[j])
                Z[j] = self.model.initial_water_saturation + attenuate * (
                            saturation_data[time_idx] - self.model.initial_water_saturation)

            # Добавляем эффекты скважин
            if has_wells:
                # Нагнетательные скважины
                for injector in self.model.wells_config['injectors']:
                    pos = injector['position']
                    water_cut = injector.get('water_cut', 1.0)
                    x_pos = pos * self.model.dx

                    # Радиус влияния скважины увеличивается со временем
                    base_radius = self.model.length / 10
                    time_factor = min(1.0, self.model.t[time_idx] / (self.model.days / 2))
                    influence_radius = base_radius * (0.5 + 1.5 * time_factor)

                    # Создаем эффект распространения воды от скважины
                    for i in range(nx):
                        for j in range(ny):
                            # Рассчитываем расстояние от точки до скважины
                            dist = np.sqrt((X[j, i] - x_pos) ** 2 + Y[j, i] ** 2)

                            # Если точка в зоне влияния скважины
                            if dist < influence_radius:
                                # Коэффициент влияния (1 у скважины, 0 на границе зоны влияния)
                                influence = max(0, 1 - dist / influence_radius)

                                # Увеличиваем водонасыщенность
                                Z[j, i] = max(Z[j, i],
                                              self.model.initial_water_saturation +
                                              influence ** 1.5 * (water_cut - self.model.initial_water_saturation))

            # Обновляем изображение
            im.set_array(Z)

            # Обновляем текст времени
            days = self.model.t[time_idx]
            time_text.set_text(f'Время: {days:.1f} дней')

            return [im, time_text, model_text] + well_markers + heterogeneity_elements

        # Базовое имя файла
        base_filename = "realistic_2d_field"

        if self.default_writer:
            # Создаем анимацию
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=frames,
                interval=1000 / self.fps,
                blit=True
            )

            # Сохраняем анимацию
            output_path = os.path.join(self.save_path, base_filename + self.default_ext)
            try:
                anim.save(output_path, writer=self.default_writer, fps=self.fps, dpi=self.dpi)
                print(f"Анимация сохранена в {output_path}")
            except Exception as e:
                print(f"Ошибка при сохранении анимации: {e}")
                # Если ошибка при сохранении, сохраняем кадры
                self._save_frames(update, frames, base_filename)
        else:
            # Если нет доступных writer'ов, сохраняем кадры
            self._save_frames(update, frames, base_filename)

        plt.close(fig)
        return os.path.join(self.save_path, base_filename + (self.default_ext if self.default_writer else ""))

    def animate_multi_well(self, frames=50):
        """
        Создание анимации для модели с несколькими скважинами
        """
        # Проверяем, является ли модель мультискважинной
        if not hasattr(self.model, 'wells_config'):
            print("Анимация нескольких скважин доступна только для модели с несколькими скважинами.")
            return None

        print("Создание анимации для модели с несколькими скважинами...")

        # Упрощенная версия для надежности - используем базовую анимацию
        # Здесь мы отказываемся от более сложной имитации 2D и используем более простую визуализацию
        base_filename = "multi_well_animation"

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(14, 6))

        # Подготавливаем первый кадр
        saturation_data = self.model.Sw_with_cap
        im = ax.imshow(
            saturation_data[0].reshape(1, -1),
            aspect='auto',
            extent=[0, self.model.length, -50, 50],
            cmap=self.cmap,
            vmin=0,
            vmax=1,
            interpolation='bicubic'
        )

        # Добавляем колонтитул
        colorbar = plt.colorbar(im, ax=ax)
        colorbar.set_label('Водонасыщенность')

        # Добавляем скважины
        well_markers = []

        # Нагнетательные скважины (синие маркеры)
        for injector in self.model.wells_config['injectors']:
            pos = injector['position']
            rate = injector['rate']
            x_pos = pos * self.model.dx
            # Рисуем скважину как окружность
            circle = plt.Circle((x_pos, 0), 20, color='blue', zorder=10)
            ax.add_patch(circle)
            well_markers.append(circle)
            # Добавляем текст
            text = ax.text(x_pos, 0, f'Инж\n{rate:.1f}', ha='center', va='center', fontsize=8, color='white')
            well_markers.append(text)

        # Добывающие скважины (красные маркеры)
        for producer in self.model.wells_config['producers']:
            pos = producer['position']
            rate = producer['rate']
            x_pos = pos * self.model.dx
            # Рисуем скважину как окружность
            circle = plt.Circle((x_pos, 0), 20, color='red', zorder=10)
            ax.add_patch(circle)
            well_markers.append(circle)
            # Добавляем текст
            text = ax.text(x_pos, 0, f'Доб\n{rate:.1f}', ha='center', va='center', fontsize=8, color='white')
            well_markers.append(text)

        # Настройка осей
        ax.set_xlim(0, self.model.length)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('Расстояние (м)')
        ax.set_title('Анимация фильтрации нефти в пласте с несколькими скважинами')
        ax.set_yticks([])

        # Добавляем сетку
        ax.grid(True, linestyle='--', alpha=0.3)

        # Текст с информацией о времени
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Шаг времени для анимации
        time_step = max(1, self.model.nt // frames)

        # Функция обновления для анимации
        def update(frame):
            time_idx = frame * time_step
            if time_idx >= self.model.nt:
                time_idx = self.model.nt - 1

            # Обновляем данные
            im.set_array(saturation_data[time_idx].reshape(1, -1))

            # Обновляем текст времени
            days = self.model.t[time_idx]
            time_text.set_text(f'Время: {days:.1f} дней')

            return [im, time_text] + well_markers

        if self.default_writer:
            # Создаем анимацию
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=frames,
                interval=1000 / self.fps,
                blit=True
            )

            # Сохраняем анимацию
            output_path = os.path.join(self.save_path, base_filename + self.default_ext)
            try:
                anim.save(output_path, writer=self.default_writer, fps=self.fps, dpi=self.dpi)
                print(f"Анимация сохранена в {output_path}")
            except Exception as e:
                print(f"Ошибка при сохранении анимации: {e}")
                # Если ошибка при сохранении, сохраняем кадры
                self._save_frames(update, frames, base_filename)
        else:
            # Если нет доступных writer'ов, сохраняем кадры
            self._save_frames(update, frames, base_filename)

        plt.close(fig)
        return os.path.join(self.save_path, base_filename + (self.default_ext if self.default_writer else ""))

    def create_all_animations(self):
        """Создание всех типов анимаций"""
        # Список для хранения путей к созданным анимациям
        created_animations = []

        try:
            # Стандартная анимация с учётом капиллярных эффектов
            with_cap_path = self.animate_saturation(with_capillary=True)
            created_animations.append(with_cap_path)
        except Exception as e:
            print(f"Ошибка при создании анимации с учётом капиллярных эффектов: {e}")

        try:
            # Стандартная анимация без учёта капиллярных эффектов
            without_cap_path = self.animate_saturation(with_capillary=False)
            created_animations.append(without_cap_path)
        except Exception as e:
            print(f"Ошибка при создании анимации без учёта капиллярных эффектов: {e}")

        try:
            # Реалистичная 2D анимация
            realistic_path = self.animate_2d_field(frames=50, resolution=50)
            created_animations.append(realistic_path)
        except Exception as e:
            print(f"Ошибка при создании реалистичной 2D анимации: {e}")

        try:
            # Анимация для модели с несколькими скважинами
            if hasattr(self.model, 'wells_config'):
                multi_well_path = self.animate_multi_well()
                created_animations.append(multi_well_path)
        except Exception as e:
            print(f"Ошибка при создании анимации для модели с несколькими скважинами: {e}")

        return created_animations


def load_default_model():
    """Загрузка модели по умолчанию для демонстрации"""
    try:
        # Сначала пробуем импортировать многоскважинную модель
        from model import MultiWellModel

        print("Создание демонстрационной модели с несколькими скважинами...")
        # Создаем конфигурацию скважин
        wells_config = {
            'injectors': [
                {'position': 0, 'rate': 2.0, 'water_cut': 1.0},
                {'position': 300, 'rate': 1.5, 'water_cut': 1.0}
            ],
            'producers': [
                {'position': 600, 'rate': 1.2},
                {'position': 800, 'rate': 1.3},
                {'position': 1000, 'rate': 1.0}
            ]
        }

        # Создаем модель
        model = MultiWellModel(length=1000.0, nx=1000, days=100, wells_config=wells_config)

        # Запускаем моделирование
        print("Запуск моделирования...")
        model.run_simulation()
        print("Моделирование завершено.")

        return model
    except ImportError:
        # Если не удалось импортировать многоскважинную модель, пробуем базовую
        try:
            from model import OilFiltrationModel

            print("Создание демонстрационной базовой модели...")
            model = OilFiltrationModel()

            # Запускаем моделирование
            print("Запуск моделирования...")
            model.run_simulation()
            print("Моделирование завершено.")

            return model
        except ImportError:
            print("Не удалось импортировать ни одну модель. Убедитесь, что файл model.py доступен.")
            return None


if __name__ == "__main__":
    print("Запуск программы анимации фильтрации нефти")

    # Загружаем демонстрационную модель
    model = load_default_model()

    if model is not None:
        # Создаем аниматор
        animator = OilFiltrationAnimation(model, fps=10, dpi=100, save_path='animation')

        # Создаем анимации
        animations = animator.create_all_animations()

        print(f"\nСоздано анимаций: {len(animations)}")
        for anim_path in animations:
            if anim_path:
                print(f"- {anim_path}")
    else:
        print("Не удалось создать модель для анимации.")