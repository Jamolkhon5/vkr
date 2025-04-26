# scientific_visualization.py - Визуализация научных аспектов моделирования
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
import networkx as nx


class ScientificVisualizer:
    """
    Класс для создания научных визуализаций, графиков и таблиц,
    иллюстрирующих теоретические аспекты моделирования фильтрации нефти
    """

    def __init__(self, output_dir='scientific_figures', basic_model=None, carbonate_model=None):
        self.output_dir = output_dir
        self.basic_model = basic_model
        self.carbonate_model = carbonate_model

        # Создаем директорию для вывода, если её нет
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Создана директория для научных визуализаций: {self.output_dir}")

    def create_buckley_leverett_plot(self):
        """Создание графика функции Баклея-Леверетта f(S)"""
        plt.figure(figsize=(10, 6))

        # Диапазон насыщенности
        sw = np.linspace(0.0, 1.0, 1000)

        # Вычисление значений функции Баклея-Леверетта
        f_values = np.zeros_like(sw)
        for i, s in enumerate(sw):
            f_values[i] = self.basic_model.fractional_flow(s)

        # Построение графика
        plt.plot(sw, f_values, 'b-', linewidth=2)

        # Отметим характерные точки
        swc = self.basic_model.initial_water_saturation
        sor = self.basic_model.residual_oil_saturation

        # Находим точку перегиба (максимальная производная)
        df = np.diff(f_values) / np.diff(sw)
        inflection_idx = np.argmax(df) + 1
        sw_inflection = sw[inflection_idx]
        f_inflection = f_values[inflection_idx]

        # Отмечаем точки
        plt.plot(swc, 0, 'ro', markersize=8, label=f'Swc = {swc}')
        plt.plot(1 - sor, 1, 'go', markersize=8, label=f'1-Sor = {1 - sor}')
        plt.plot(sw_inflection, f_inflection, 'mo', markersize=8, label=f'Точка перегиба')

        # Добавляем касательную в точке перегиба
        tangent_x = np.array([0, 1])
        tangent_y = f_inflection + df[inflection_idx - 1] * (tangent_x - sw_inflection)
        plt.plot(tangent_x, tangent_y, 'r--', alpha=0.7, label='Касательная')

        # Оформление графика
        plt.xlabel('Водонасыщенность, S')
        plt.ylabel('Функция Баклея-Леверетта, f(S)')
        plt.title('Рисунок 2.1: Функция Баклея-Леверетта')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Добавляем пояснительный текст
        plt.text(0.1, 0.8, 'Функция Баклея-Леверетта:\n' +
                 r'$f(S) = \frac{k_{rw}/\mu_w}{k_{rw}/\mu_w + k_{ro}/\mu_o}$',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # Сохранение графика
        output_path = os.path.join(self.output_dir, 'buckley_leverett.png')
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"График функции Баклея-Леверетта сохранен: {output_path}")
        return output_path

    def create_capillary_pressure_plot(self):
        """Создание графика капиллярного давления p_c(S) с разными параметрами"""
        plt.figure(figsize=(12, 8))

        # Диапазон насыщенности
        sw = np.linspace(0.1, 0.9, 1000)

        # Базовая модель
        pc_base = np.zeros_like(sw)
        for i, s in enumerate(sw):
            pc_base[i] = self.basic_model.capillary_pressure(s)

        # Строим график для базовой модели
        plt.subplot(2, 2, 1)
        plt.plot(sw, pc_base, 'b-', linewidth=2, label='Базовая модель')
        plt.xlabel('Водонасыщенность, S')
        plt.ylabel('Капиллярное давление, МПа')
        plt.title('Базовые параметры')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Варьируем давление входа
        plt.subplot(2, 2, 2)
        original_entry_pressure = self.basic_model.entry_pressure

        entry_pressures = [0.5, 1.0, 1.5, 2.0]
        for ep in entry_pressures:
            self.basic_model.entry_pressure = ep
            pc_values = np.zeros_like(sw)
            for i, s in enumerate(sw):
                pc_values[i] = self.basic_model.capillary_pressure(s)
            plt.plot(sw, pc_values, linewidth=2, label=f'Pe = {ep} МПа')

        # Восстанавливаем исходное значение
        self.basic_model.entry_pressure = original_entry_pressure

        plt.xlabel('Водонасыщенность, S')
        plt.ylabel('Капиллярное давление, МПа')
        plt.title('Влияние давления входа')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Варьируем индекс распределения пор
        plt.subplot(2, 2, 3)
        original_pore_index = self.basic_model.pore_distribution_index

        pore_indices = [0.5, 1.0, 1.5, 2.0]
        for pi in pore_indices:
            self.basic_model.pore_distribution_index = pi
            pc_values = np.zeros_like(sw)
            for i, s in enumerate(sw):
                pc_values[i] = self.basic_model.capillary_pressure(s)
            plt.plot(sw, pc_values, linewidth=2, label=f'λ = {pi}')

        # Восстанавливаем исходное значение
        self.basic_model.pore_distribution_index = original_pore_index

        plt.xlabel('Водонасыщенность, S')
        plt.ylabel('Капиллярное давление, МПа')
        plt.title('Влияние индекса распределения пор')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Варьируем коэффициент смачиваемости
        plt.subplot(2, 2, 4)
        original_wettability = self.basic_model.wettability_factor

        wettabilities = [0.1, 0.3, 0.6, 0.9]
        for w in wettabilities:
            self.basic_model.wettability_factor = w
            pc_values = np.zeros_like(sw)
            for i, s in enumerate(sw):
                pc_values[i] = self.basic_model.capillary_pressure(s)
            plt.plot(sw, pc_values, linewidth=2, label=f'W = {w}')

        # Восстанавливаем исходное значение
        self.basic_model.wettability_factor = original_wettability

        plt.xlabel('Водонасыщенность, S')
        plt.ylabel('Капиллярное давление, МПа')
        plt.title('Влияние коэффициента смачиваемости')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Общий заголовок
        plt.suptitle('Рисунок 2.2: Капиллярное давление при различных параметрах', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Сохранение графика
        output_path = os.path.join(self.output_dir, 'capillary_pressure.png')
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"График капиллярного давления сохранен: {output_path}")
        return output_path

    def create_upwind_scheme_illustration(self):
        """Создание схемы метода апвинд с иллюстрацией направления потока"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Настройка осей
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Рисуем вертикальные пунктирные линии по всей высоте графика (узлы сетки)
        for i in range(1, 10):
            ax.axvline(i, color='gray', linestyle='--', alpha=0.5, ymin=0, ymax=1)

        # Заголовок рисунка
        ax.text(5, 9.2, 'Рисунок 2.3: Иллюстрация метода апвинд для решения задачи фильтрации',
                ha='center', va='center', fontsize=16, fontweight='bold')

        # Уравнение метода апвинд с выделением членов
        equation = r'$S_i^{n+1} = S_i^n - \frac{\Delta t}{\Delta x}[f(S_i^n) - f(S_{i-1}^n)] + D_i \cdot \frac{\Delta t}{\Delta x^2}[S_{i+1}^n - 2S_i^n + S_{i-1}^n]$'
        ax.text(5, 7.5, equation, ha='center', va='center', fontsize=16)

        # Выделение конвективного члена (красный)
        rect_conv = plt.Rectangle((3.2, 7.35), 2.6, 0.35, fill=True, alpha=0.2, color='red', zorder=1)
        ax.add_patch(rect_conv)

        # Выделение диффузионного члена (синий)
        rect_diff = plt.Rectangle((6.3, 7.35), 3.0, 0.35, fill=True, alpha=0.2, color='blue', zorder=1)
        ax.add_patch(rect_diff)

        # Название метода
        ax.text(5, 6.2, 'Метод апвинд (первый порядок):', ha='center', va='center', fontsize=14)

        # Конвективный член - название и пояснение
        ax.text(5, 5.4, 'Конвективный член', ha='center', va='center', fontsize=13, color='red')
        ax.text(5, 5.0, '(учитывает направление)', ha='center', va='center', fontsize=13, color='red')

        # Диффузионный член - название и пояснение
        ax.text(5, 4.3, 'Диффузионный член', ha='center', va='center', fontsize=13, color='blue')
        ax.text(5, 3.9, '(капиллярные эффекты)', ha='center', va='center', fontsize=13, color='blue')

        # Заголовок для направления потока
        ax.text(5, 3.0, 'Направление потока', ha='center', va='center', fontsize=14)

        # Рисуем стрелки потока (вправо)
        arrow_y = 2.3  # Позиция стрелок по вертикали
        for i in range(1, 9):
            arrow = plt.arrow(i + 0.1, arrow_y, 0.8, 0,
                              head_width=0.2, head_length=0.1,
                              fc='red', ec='red', width=0.05)
            ax.add_patch(arrow)

        # Пояснения к методу апвинд
        ax.text(5, 1.7, 'Для схемы апвинд:', ha='center', va='center', fontsize=14)
        ax.text(5, 1.3, '1. Определяем направление потока', ha='center', va='center', fontsize=12)
        ax.text(5, 0.9, '2. Берем значения с направления "против потока"', ha='center', va='center', fontsize=12)

        # Метки узлов сетки (немного смещены влево от линий)
        node_labels = ['i-9', 'i-8', 'i-7', 'i-6', 'i', 'i+1', 'i+2', 'i+3', 'i+4']
        node_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        for pos, label in zip(node_positions, node_labels):
            # Сдвигаем метки узлов существенно левее от вертикальных линий
            ax.text(pos - 0.4, 0.4, label, ha='center', va='center', fontsize=12)

        # Отмечаем текущий узел (i) синим кругом, также сдвинутым влево
        current_node = plt.Circle((5 - 0.4, 0.4), 0.3, color='#4169E1', alpha=0.9)
        ax.add_patch(current_node)
        ax.text(5 - 0.4, 0.4, 'i', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

        # Сохраняем в указанную директорию
        output_path = os.path.join(self.output_dir, 'upwind_scheme.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Иллюстрация метода апвинд сохранена: {output_path}")
        return output_path

    def create_algorithm_flowchart(self):
        """Создание блок-схемы алгоритма численного решения"""
        # Создаем ориентированный граф
        G = nx.DiGraph()

        # Добавляем узлы (шаги алгоритма)
        nodes = [
            "Начало",
            "Инициализация\nпараметров модели",
            "Задание начальных условий\nS(x,0) = S₀(x)",
            "Начало временного цикла\nt = 0, t < T, t += Δt",
            "Расчет функции\nБаклея-Леверетта f(S)",
            "Расчет капиллярного\nдавления p_c(S)",
            "Расчет коэффициента\nдиффузии D(S)",
            "Решение методом апвинд",
            "Применение граничных\nусловий",
            "Проверка устойчивости\nрешения",
            "Конец временного цикла",
            "Анализ результатов",
            "Визуализация",
            "Конец"
        ]

        # Добавляем узлы в граф
        for i, node in enumerate(nodes):
            G.add_node(i, label=node)

        # Добавляем связи между узлами (направление алгоритма)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            (7, 8), (8, 9), (9, 10), (10, 3), (10, 11), (11, 12), (12, 13)
        ]

        # Добавляем обратную связь от проверки устойчивости
        G.add_edge(9, 7, decision=True)

        # Добавляем связи в граф
        for edge in edges:
            G.add_edge(edge[0], edge[1])

        # Создаем макет графа
        pos = {
            0: (0, 10),
            1: (0, 9),
            2: (0, 8),
            3: (0, 7),
            4: (0, 6),
            5: (0, 5),
            6: (0, 4),
            7: (0, 3),
            8: (0, 2),
            9: (0, 1),
            10: (2, 1),
            11: (2, 2),
            12: (2, 3),
            13: (2, 4)
        }

        # Создаем фигуру для блок-схемы
        plt.figure(figsize=(10, 14))

        # Рисуем узлы разных типов
        start_end_nodes = [0, 13]  # Начало и конец
        process_nodes = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12]  # Процессы
        decision_nodes = [3, 9]  # Точки принятия решений

        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, nodelist=start_end_nodes, node_shape='o',
                               node_size=3000, node_color='lightgreen')
        nx.draw_networkx_nodes(G, pos, nodelist=process_nodes, node_shape='s',
                               node_size=3000, node_color='lightblue')
        nx.draw_networkx_nodes(G, pos, nodelist=decision_nodes, node_shape='d',
                               node_size=3000, node_color='lightsalmon')

        # Рисуем связи
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)

        # Добавляем пунктирную линию для условного перехода
        plt.plot([0, 1.4, 1.4, 0], [1, 1, 3, 3], 'k--', alpha=0.7)
        plt.text(0.7, 2, 'Если условие\nне выполнено', ha='center', fontsize=10)

        # Добавляем текст к обратной связи
        plt.plot([0, -1.4, -1.4, 0], [7, 7, 1, 1], 'k--', alpha=0.7)
        plt.text(-0.7, 4, 'Следующий\nвременной шаг', ha='center', fontsize=10, rotation=90)

        # Добавляем метки узлов
        labels = {i: node for i, node in enumerate(nodes)}
        label_pos = {k: (v[0], v[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=10, font_family="sans-serif")

        # Убираем оси
        plt.axis('off')

        # Добавляем заголовок
        plt.title('Рисунок 2.4: Блок-схема алгоритма численного решения', fontsize=16, pad=20)

        # Сохранение блок-схемы
        output_path = os.path.join(self.output_dir, 'algorithm_flowchart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Блок-схема алгоритма сохранена: {output_path}")
        return output_path

    def create_model_parameters_table(self):
        """Создание таблицы параметров моделей (базовая и карбонатная)"""
        plt.figure(figsize=(12, 8))
        plt.axis('off')

        # Заголовок таблицы
        plt.text(0.5, 0.95, 'Таблица 2.1: Параметры моделей', fontsize=16, ha='center')

        # Данные для таблицы
        columns = ['Параметр', 'Обозначение', 'Базовая модель', 'Карбонатная модель', 'Единицы измерения']

        # Собираем параметры базовой модели
        basic_params = {
            'Длина пласта': (self.basic_model.length, 'L', 'м'),
            'Пористость': (self.basic_model.porosity, 'φ', 'д. ед.'),
            'Вязкость нефти': (self.basic_model.mu_oil, 'μₒ', 'мПа·с'),
            'Вязкость воды': (self.basic_model.mu_water, 'μw', 'мПа·с'),
            'Начальная водонасыщенность': (self.basic_model.initial_water_saturation, 'Swc', 'д. ед.'),
            'Остаточная нефтенасыщенность': (self.basic_model.residual_oil_saturation, 'Sor', 'д. ед.'),
            'Давление входа': (self.basic_model.entry_pressure, 'Pe', 'МПа'),
            'Индекс распределения пор': (self.basic_model.pore_distribution_index, 'λ', 'отн. ед.'),
            'Коэффициент смачиваемости': (self.basic_model.wettability_factor, 'W', 'отн. ед.')
        }

        # Собираем параметры карбонатной модели, если она доступна
        carbonate_params = {}
        if self.carbonate_model:
            carbonate_params = {
                'Длина пласта': (self.carbonate_model.length, 'L', 'м'),
                'Пористость': (self.carbonate_model.porosity, 'φ', 'д. ед.'),
                'Вязкость нефти': (self.carbonate_model.mu_oil, 'μₒ', 'мПа·с'),
                'Вязкость воды': (self.carbonate_model.mu_water, 'μw', 'мПа·с'),
                'Начальная водонасыщенность': (self.carbonate_model.initial_water_saturation, 'Swc', 'д. ед.'),
                'Остаточная нефтенасыщенность': (self.carbonate_model.residual_oil_saturation, 'Sor', 'д. ед.'),
                'Давление входа': (self.carbonate_model.entry_pressure, 'Pe', 'МПа'),
                'Индекс распределения пор': (self.carbonate_model.pore_distribution_index, 'λ', 'отн. ед.'),
                'Коэффициент смачиваемости': (self.carbonate_model.wettability_factor, 'W', 'отн. ед.'),
                'Пористость матрицы': (self.carbonate_model.matrix_porosity, 'φₘ', 'д. ед.'),
                'Пористость трещин': (self.carbonate_model.fracture_porosity, 'φf', 'д. ед.'),
                'Проницаемость матрицы': (self.carbonate_model.matrix_permeability, 'kₘ', 'мД'),
                'Проницаемость трещин': (self.carbonate_model.fracture_permeability, 'kf', 'мД'),
                'Форм-фактор': (self.carbonate_model.shape_factor, 'σ', 'отн. ед.')
            }

        # Объединяем параметры
        all_params = set(basic_params.keys()).union(set(carbonate_params.keys()))

        # Строки таблицы
        rows = []
        for param in all_params:
            if param in basic_params:
                basic_value, symbol, unit = basic_params[param]
                basic_str = f"{basic_value:.4f}" if isinstance(basic_value, float) else str(basic_value)
            else:
                basic_str = "-"
                symbol = carbonate_params[param][1]
                unit = carbonate_params[param][2]

            if param in carbonate_params:
                carbonate_value = carbonate_params[param][0]
                carbonate_str = f"{carbonate_value:.4f}" if isinstance(carbonate_value, float) else str(carbonate_value)
            else:
                carbonate_str = "-"

            rows.append([param, symbol, basic_str, carbonate_str, unit])

        # Сортируем строки
        rows.sort(key=lambda x: x[0])

        # Рисуем таблицу
        table_data = [columns] + rows

        # Создаем таблицу
        table = plt.table(cellText=table_data,
                          colLabels=None,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.05, 0.8, 0.85])

        # Настройка стиля таблицы
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        # Устанавливаем стиль для заголовка
        for j, cell in enumerate(table._cells[(0, j)] for j in range(len(columns))):
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

        # Сохранение таблицы
        output_path = os.path.join(self.output_dir, 'model_parameters_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Таблица параметров моделей сохранена: {output_path}")
        return output_path

    def create_numerical_schemes_table(self):
        """Создание таблицы сравнения различных численных схем для задачи фильтрации"""
        plt.figure(figsize=(14, 10))
        plt.axis('off')

        # Заголовок таблицы
        plt.text(0.5, 0.95, 'Таблица 2.2: Сравнение численных схем для задачи фильтрации', fontsize=16, ha='center')

        # Данные для таблицы
        columns = ['Численная схема', 'Порядок аппроксимации', 'Устойчивость', 'Преимущества', 'Недостатки',
                   'Применимость']

        rows = [
            [
                'Метод апвинд\n(первого порядка)',
                'O(Δt, Δx)',
                'Условная\nCFL ≤ 1',
                '• Монотонность\n• Простота реализации\n• Физичность решения',
                '• Низкий порядок точности\n• Численная диффузия',
                'Наилучший выбор для\nзадач с разрывными\nрешениями'
            ],
            [
                'Схема Лакса-Вендроффа',
                'O(Δt, Δx²)',
                'Условная\nCFL ≤ 1',
                '• Второй порядок точности\n• Меньшая численная диффузия',
                '• Нефизичные осцилляции\n• Сложность реализации',
                'Для гладких решений\nбез разрывов'
            ],
            [
                'Схема с центральными\nразностями',
                'O(Δt, Δx²)',
                'Безусловно\nнеустойчива для\nгиперболических\nуравнений',
                '• Высокий порядок точности\n• Симметричность',
                '• Неустойчивость\n• Нефизичные осцилляции',
                'Неприменима для\nзадач фильтрации\nв чистом виде'
            ],
            [
                'TVD-схемы\n(с ограничителями\nпотоков)',
                'O(Δt, Δx²)',
                'Условная\nCFL ≤ 1',
                '• Высокий порядок точности\n• Монотонность\n• Отсутствие осцилляций',
                '• Сложность реализации\n• Высокие вычислительные\nзатраты',
                'Оптимальны для\nзадач с разрывами\nи высокими\nградиентами'
            ],
            [
                'Неявные схемы',
                'O(Δt, Δx)',
                'Безусловная',
                '• Отсутствие ограничений\nна шаг по времени\n• Стабильность',
                '• Необходимость решения\nСЛАУ на каждом шаге\n• Вычислительная\nсложность',
                'Для задач, требующих\nбольших шагов\nпо времени'
            ]
        ]

        # Рисуем таблицу
        table_data = [columns] + rows

        # Создаем таблицу
        table = plt.table(cellText=table_data,
                          colLabels=None,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.05, 0.05, 0.9, 0.85])

        # Настройка стиля таблицы
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Устанавливаем стиль для заголовка
        for j, cell in enumerate(table._cells[(0, j)] for j in range(len(columns))):
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

        # Выделяем метод апвинд
        for j, cell in enumerate(table._cells[(1, j)] for j in range(len(columns))):
            cell.set_facecolor('#E2EFDA')

        # Сохранение таблицы
        output_path = os.path.join(self.output_dir, 'numerical_schemes_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Таблица численных схем сохранена: {output_path}")
        return output_path

    def create_all_visualizations(self):
        """Создание всех научных визуализаций"""
        # Создаем графики
        buckley_leverett_path = self.create_buckley_leverett_plot()
        capillary_pressure_path = self.create_capillary_pressure_plot()
        upwind_scheme_path = self.create_upwind_scheme_illustration()
        algorithm_flowchart_path = self.create_algorithm_flowchart()

        # Создаем таблицы
        model_parameters_table_path = self.create_model_parameters_table()
        numerical_schemes_table_path = self.create_numerical_schemes_table()

        # Возвращаем пути ко всем созданным визуализациям
        return {
            'buckley_leverett': buckley_leverett_path,
            'capillary_pressure': capillary_pressure_path,
            'upwind_scheme': upwind_scheme_path,
            'algorithm_flowchart': algorithm_flowchart_path,
            'model_parameters_table': model_parameters_table_path,
            'numerical_schemes_table': numerical_schemes_table_path
        }