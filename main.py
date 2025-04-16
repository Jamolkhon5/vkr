# main.py - Главный файл для запуска моделирования
import os
import sys
import argparse
import traceback
import time
from datetime import datetime

from matplotlib import pyplot as plt

from result_analysis import analyze_model_results, compare_models, create_summarydir, copy_best_graphs, create_comparison_graphs
# Импорт моделей
# В начало файла main.py, после импорта моделей
try:
    from model import OilFiltrationModel, MultiWellModel, MultiWell2DModel
    print("Базовые модели успешно импортированы")
except ImportError as e:
    print(f"ОШИБКА: Не удалось импортировать базовые модели: {e}")
    sys.exit(1)

# Импорт расширенной модели для карбонатов
try:
    from carbonate_model import CarbonateModel

    print("Модель карбонатного коллектора успешно импортирована")
except ImportError as e:
    print(f"ОШИБКА: Не удалось импортировать модель карбонатного коллектора: {e}")
    print("Будет использована только базовая модель")
    CarbonateModel = None

# Импорт вспомогательных модулей
try:
    from console_output import ConsoleOutput
    from visualizer import Visualizer

    print("Модули вывода и визуализации успешно импортированы")
except ImportError as e:
    print(f"ОШИБКА: Не удалось импортировать модули вывода: {e}")
    sys.exit(1)

# Импорт модулей для работы с данными и экспорта
try:
    from data_loader import DataLoader
    from integration_module import SimulatorInterface

    print("Модули загрузки данных и интеграции успешно импортированы")
except ImportError as e:
    print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать модули для работы с данными: {e}")
    print("Функции загрузки данных и экспорта будут недоступны")
    DataLoader = None
    SimulatorInterface = None


def check_data_directory(data_dir):
    """Проверка наличия директории с данными и файлов в ней"""
    if not os.path.exists(data_dir):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Директория {data_dir} не найдена!")
        try:
            os.makedirs(data_dir)
            print(f"Создана директория {data_dir}")
        except Exception as e:
            print(f"Не удалось создать директорию: {e}")
        return False

    data_files = [
        'rock_properties.csv',
        'capillary_pressure.csv',
        'relative_perm.csv',
        'pvt_data.csv',
        'production_data.csv'
    ]

    missing_files = []
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Следующие файлы не найдены в директории {data_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print(f"Все необходимые файлы данных найдены в директории {data_dir}")
    return True


def setup_output_directory(output_dir):
    """Создание директории для выходных файлов"""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Создана директория для вывода: {output_dir}")
        except Exception as e:
            print(f"Не удалось создать директорию для вывода: {e}")
            return False
    return True


def print_model_info(model):
    """Вывод информации о параметрах модели"""
    print("\nПараметры модели:")
    print(f"Длина пласта: {model.length} м")
    print(f"Пористость: {model.porosity}")
    print(f"Вязкость нефти: {model.mu_oil} мПа·с")
    print(f"Вязкость воды: {model.mu_water} мПа·с")
    print(f"Начальная водонасыщенность: {model.initial_water_saturation}")
    print(f"Остаточная нефтенасыщенность: {model.residual_oil_saturation}")
    print(f"Количество узлов сетки: {model.nx}")
    print(f"Временной шаг: {model.dt} дней")
    print(f"Общее время моделирования: {model.days} дней")

    # Вывод дополнительных параметров для карбонатной модели
    if hasattr(model, 'matrix_porosity'):
        print("\nДополнительные параметры карбонатной модели:")
        print(f"Пористость матрицы: {model.matrix_porosity}")
        print(f"Пористость трещин: {model.fracture_porosity}")
        print(f"Проницаемость матрицы: {model.matrix_permeability} мД")
        print(f"Проницаемость трещин: {model.fracture_permeability} мД")
        print(f"Форм-фактор обмена: {model.shape_factor}")

    # Вывод информации о скважинах для модели с несколькими скважинами
    if hasattr(model, 'wells_config'):
        print("\nИнформация о скважинах:")
        print(f"Количество нагнетательных скважин: {len(model.wells_config['injectors'])}")
        print(f"Количество добывающих скважин: {len(model.wells_config['producers'])}")

        # Проверяем, является ли модель 2D-моделью
        is_2d_model = hasattr(model, 'length_x')

        print("\nНагнетательные скважины:")
        for i, injector in enumerate(model.wells_config['injectors']):
            if is_2d_model:
                # Для 2D-модели используем position_x и position_y
                pos_x = injector['position_x']
                pos_y = injector['position_y']
                x_pos = pos_x * model.dx
                y_pos = pos_y * model.dy
                print(f"  Скважина {i + 1}: позиция = ({x_pos:.1f} м, {y_pos:.1f} м), дебит = {injector['rate']:.2f} м³/день")
            else:
                # Для 1D-модели используем position
                pos = injector['position']
                x_pos = pos * model.dx
                print(f"  Скважина {i + 1}: позиция = {x_pos:.1f} м, дебит = {injector['rate']:.2f} м³/день")

        print("\nДобывающие скважины:")
        for i, producer in enumerate(model.wells_config['producers']):
            if is_2d_model:
                # Для 2D-модели используем position_x и position_y
                pos_x = producer['position_x']
                pos_y = producer['position_y']
                x_pos = pos_x * model.dx
                y_pos = pos_y * model.dy
                print(f"  Скважина {i + 1}: позиция = ({x_pos:.1f} м, {y_pos:.1f} м), дебит = {producer['rate']:.2f} м³/день")
            else:
                # Для 1D-модели используем position
                pos = producer['position']
                x_pos = pos * model.dx
                print(f"  Скважина {i + 1}: позиция = {x_pos:.1f} м, дебит = {producer['rate']:.2f} м³/день")

    # Вывод параметров капиллярной модели
    print("\nПараметры капиллярной модели:")
    print(f"Давление входа: {model.entry_pressure} МПа")
    print(f"Индекс распределения пор: {model.pore_distribution_index}")
    print(f"Коэффициент смачиваемости: {model.wettability_factor}")

    # Если это 2D-модель, выводим дополнительные параметры
    if hasattr(model, 'length_x'):
        print("\nДополнительные параметры 2D-модели:")
        print(f"Длина пласта по X: {model.length_x} м")
        print(f"Длина пласта по Y: {model.length_y} м")
        print(f"Узлов сетки по X: {model.nx}")
        print(f"Узлов сетки по Y: {model.ny}")
        print(f"Шаг сетки по X: {model.dx} м")
        print(f"Шаг сетки по Y: {model.dy} м")


def run_simulation(model, model_type):
    """Запуск моделирования с учетом типа модели"""
    start_time = time.time()
    print("\nЗапуск моделирования...")

    try:
        if model_type == 'carbonate':
            if hasattr(model, 'run_dual_porosity_simulation'):
                print("Запуск модели с двойной пористостью (карбонатный коллектор)...")
                model.run_dual_porosity_simulation()
            else:
                print("ОШИБКА: Метод run_dual_porosity_simulation не найден в модели карбонатного коллектора!")
                print("Запуск стандартного моделирования...")
                model.run_simulation()
        else:
            print(f"Запуск {model_type} модели...")
            model.run_simulation()

        end_time = time.time()
        duration = end_time - start_time
        print(f"Моделирование успешно завершено за {duration:.2f} секунд!")
        return True
    except Exception as e:
        print(f"ОШИБКА при моделировании: {e}")
        print("Подробная информация об ошибке:")
        traceback.print_exc()
        return False


def export_results(model, export_type, output_dir):
    """Экспорт результатов в выбранном формате"""
    if SimulatorInterface is None:
        print("Экспорт невозможен: модуль SimulatorInterface недоступен")
        return False

    if export_type == 'none':
        return True

    try:
        print(f"\nЭкспорт результатов в формате {export_type}...")
        interface = SimulatorInterface(model)
        interface.output_dir = output_dir

        if export_type == 'rn_kim':
            interface.export_to_rn_kim()
            interface.generate_rn_kim_config()
            print(f"Результаты успешно экспортированы в формате РН-КИМ в директорию {output_dir}")
        elif export_type == 'eclipse':
            interface.export_to_eclipse()
            print(f"Результаты успешно экспортированы в формате Eclipse в директорию {output_dir}")

        return True
    except Exception as e:
        print(f"ОШИБКА при экспорте результатов: {e}")
        traceback.print_exc()
        return False


def create_wells_config(num_wells, nx=100):
    """Создание конфигурации для скважин"""
    # Минимум 2 скважины (одна нагнетательная, одна добывающая)
    num_wells = max(2, num_wells)

    # Определяем количество нагнетательных и добывающих скважин
    if num_wells == 2:
        # Одна нагнетательная, одна добывающая
        num_injectors = 1
        num_producers = 1
    else:
        # Примерно 40% нагнетательных, 60% добывающих
        num_injectors = max(1, int(0.4 * num_wells))
        num_producers = num_wells - num_injectors

    # Создаем конфигурацию скважин
    wells_config = {
        'injectors': [],
        'producers': []
    }

    # Добавляем нагнетательные скважины (равномерно распределенные в первой половине пласта)
    for i in range(num_injectors):
        position = int((i * nx / (2 * num_injectors - 1)) if num_injectors > 1 else 0)
        injector = {
            'position': position,
            'rate': 1.0,  # м³/день
            'water_cut': 1.0  # 100% воды
        }
        wells_config['injectors'].append(injector)

    # Добавляем добывающие скважины (равномерно распределенные во второй половине пласта)
    for i in range(num_producers):
        position = int(nx / 2 + i * nx / (2 * num_producers))
        if i == num_producers - 1:  # Последняя скважина в конце пласта
            position = nx
        producer = {
            'position': position,
            'rate': 1.0  # м³/день
        }
        wells_config['producers'].append(producer)

    # Балансируем суммарные дебиты (закачка должна равняться добыче)
    total_injection = sum(inj['rate'] for inj in wells_config['injectors'])
    total_production = sum(prod['rate'] for prod in wells_config['producers'])

    # Корректируем дебиты добывающих скважин
    ratio = total_injection / total_production
    for producer in wells_config['producers']:
        producer['rate'] *= ratio

    return wells_config


def load_wells_config(data_dir, nx=1000, length=1000.0):
    """Загрузка конфигурации скважин из файла wells_data.csv"""
    file_path = os.path.join(data_dir, 'wells_data.csv')

    # Проверяем наличие файла
    if not os.path.exists(file_path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Файл {file_path} не найден. Будет использована генерируемая конфигурация.")
        return None

    try:
        # Пытаемся импортировать pandas
        try:
            import pandas as pd
        except ImportError:
            print("ПРЕДУПРЕЖДЕНИЕ: Модуль pandas не доступен. Будет использована генерируемая конфигурация.")
            return None

        # Загружаем данные из CSV
        wells_data = pd.read_csv(file_path)

        # Создаем конфигурацию скважин
        wells_config = {
            'injectors': [],
            'producers': []
        }

        # Преобразуем позицию в метрах в индекс сетки
        dx = length / nx

        # Добавляем нагнетательные скважины
        injectors = wells_data[wells_data['Well_Type'] == 'injector']
        for _, well in injectors.iterrows():
            position = int(well['X_Position_m'] / dx)
            position = max(0, min(position, nx))  # Ограничиваем позицию в пределах сетки

            injector = {
                'position': position,
                'rate': well['Flow_Rate_m3_day'],
                'water_cut': well['Water_Cut']
            }
            wells_config['injectors'].append(injector)

        # Добавляем добывающие скважины
        producers = wells_data[wells_data['Well_Type'] == 'producer']
        for _, well in producers.iterrows():
            position = int(well['X_Position_m'] / dx)
            position = max(0, min(position, nx))  # Ограничиваем позицию в пределах сетки

            producer = {
                'position': position,
                'rate': well['Flow_Rate_m3_day']
            }
            wells_config['producers'].append(producer)

        # Проверяем, что есть хотя бы одна нагнетательная и одна добывающая скважина
        if not wells_config['injectors'] or not wells_config['producers']:
            print("ПРЕДУПРЕЖДЕНИЕ: В файле данных недостаточно скважин. Будет использована генерируемая конфигурация.")
            return None

        print(
            f"Загружена конфигурация скважин из файла: {len(wells_config['injectors'])} нагнетательных, {len(wells_config['producers'])} добывающих")
        return wells_config

    except Exception as e:
        print(f"ОШИБКА при загрузке конфигурации скважин: {e}")
        return None


def create_animations(model, model_type):
    """Создание анимаций фильтрации нефти для заданной модели"""
    print("\nСоздание анимаций фильтрации нефти...")

    try:
        # Импортируем модуль анимации
        from animation import OilFiltrationAnimation

        # Создаем экземпляр класса анимации с моделью
        animator = OilFiltrationAnimation(model, fps=15, dpi=150)

        # Создаем все анимации для модели
        animations = animator.create_all_animations()

        print(f"Анимации созданы успешно: {len(animations)} файлов")

        return True
    except Exception as e:
        print(f"ОШИБКА при создании анимаций: {e}")
        import traceback
        traceback.print_exc()
        return False


# Функция для создания директории для графиков по типу модели
def create_model_output_directory(base_dir, model_type, rock_type=None):
    """
    Создает директорию для вывода графиков определенной модели

    Parameters:
    -----------
    base_dir : str
        Базовая директория для вывода данных
    model_type : str
        Тип модели ('basic', 'carbonate', 'multi_well')
    rock_type : str, optional
        Тип породы (для карбонатной модели)

    Returns:
    --------
    str
        Путь к созданной директории
    """
    # Создаем базовую директорию, если её нет
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Создана базовая директория: {base_dir}")

    # Определяем имя директории в зависимости от типа модели
    if model_type == 'basic':
        dir_name = "базовая_модель"
    elif model_type == 'carbonate':
        dir_name = f"карбонатная_модель_{rock_type}"
    elif model_type == 'multi_well':
        dir_name = "модель_с_несколькими_скважинами"
    else:
        dir_name = f"модель_{model_type}"

    # Формируем полный путь
    output_dir = os.path.join(base_dir, dir_name)

    # Создаем директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для графиков: {output_dir}")

    return output_dir


# Импорт из модуля result_analysis
from result_analysis import analyze_model_results, compare_models, create_summarydir, copy_best_graphs, \
    create_comparison_graphs


def main():
    """Основная функция"""
    # Фиксируем время начала работы
    start_time = datetime.now()
    print(f"Запуск программы: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Моделирование фильтрации нефти с учетом капиллярных эффектов')
    # В функции main() добавьте аргумент для 2D-модели:
    parser.add_argument('--model', choices=['basic', 'carbonate', 'multi_well', 'multi_well_2d', 'all'], default='all',
                        help='Тип модели (basic - базовая, carbonate - для карбонатных коллекторов, ' +
                             'multi_well - с несколькими скважинами в 1D, ' +
                             'multi_well_2d - с несколькими скважинами в 2D, ' +
                             'all - все модели)')
    parser.add_argument('--wells', type=int, default=3,
                        help='Количество скважин (для multi_well модели, не менее 2)')
    parser.add_argument('--rock_type', type=str, default='Limestone',
                        help='Тип породы (Limestone, Dolomite, Vuggy limestone)')
    parser.add_argument('--export', choices=['none', 'rn_kim', 'eclipse'], default='none',
                        help='Экспорт данных (none - без экспорта, rn_kim - для РН-КИМ, eclipse - для Eclipse)')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Путь к директории с файлами данных')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Путь к директории для выходных файлов')
    parser.add_argument('--graphs_dir', type=str, default='graphs',
                        help='Путь к директории для сохранения графиков')
    parser.add_argument('--debug', action='store_true',
                        help='Режим отладки с дополнительной информацией')

    args = parser.parse_args()

    # Вывод информации о запуске
    print("\n" + "=" * 80)
    print("Запуск моделирования фильтрации нефти в пористой среде")
    print("с использованием метода апвинд (1D и 2D)")
    print("=" * 80)

    # Вывод параметров запуска
    print("\nПараметры запуска:")
    print(f"Тип модели: {args.model}")
    if args.model in ['multi_well', 'multi_well_2d', 'all']:
        print(f"Количество скважин: {args.wells}")
    print(f"Тип породы: {args.rock_type}")
    print(f"Экспорт данных: {args.export}")
    print(f"Директория данных: {args.data_dir}")
    print(f"Директория вывода: {args.output_dir}")
    print(f"Директория графиков: {args.graphs_dir}")
    print(f"Режим отладки: {'Включен' if args.debug else 'Выключен'}")

    # Проверка наличия данных
    data_available = check_data_directory(args.data_dir)
    if not data_available and (args.model == 'carbonate' or args.model == 'all'):
        print("ПРЕДУПРЕЖДЕНИЕ: Отсутствуют файлы данных, необходимые для карбонатной модели")
        print("Будет использована базовая модель без учета специфики карбонатного коллектора")

    # Подготовка директории вывода
    setup_output_directory(args.output_dir)

    # Подготовка директории для графиков
    if not os.path.exists(args.graphs_dir):
        os.makedirs(args.graphs_dir)
        print(f"Создана директория для графиков: {args.graphs_dir}")

    # Список моделей для запуска - ВОТ ЭТУ ЧАСТЬ НУЖНО ИЗМЕНИТЬ
    models_to_run = []
    if args.model == 'all':
        # Включаем все модели, в том числе 2D
        models_to_run = ['basic', 'carbonate', 'multi_well', 'multi_well_2d']
    else:
        models_to_run = [args.model]

    # Список для сохранения моделей для сравнительного анализа
    models_to_compare = []
    model_dirs = {}

    # Последовательный запуск всех выбранных моделей
    for model_type in models_to_run:
        print(f"\n{'=' * 40} Запуск модели: {model_type} {'=' * 40}")

        # Создание модели в зависимости от выбранного типа
        try:
            if model_type == 'carbonate' and CarbonateModel is not None:
                print(f"\nСоздание модели для карбонатных коллекторов (тип породы: {args.rock_type})...")
                model = CarbonateModel(rock_type=args.rock_type)
            elif model_type == 'multi_well':
                print(f"\nСоздание модели с несколькими скважинами (1D)...")
                # Пытаемся загрузить конфигурацию скважин из файла
                wells_config = load_wells_config(args.data_dir, nx=1000, length=1000.0)

                # Если загрузка не удалась, создаем конфигурацию программно
                if wells_config is None:
                    print(f"Создание программной конфигурации для {args.wells} скважин...")
                    wells_config = create_wells_config(args.wells, nx=1000)
                else:
                    print("Используется конфигурация скважин из файла.")

                model = MultiWellModel(length=1000.0, nx=1000, days=100, wells_config=wells_config)
            # В функции main(), в цикле по моделям:
            elif model_type == 'multi_well_2d':
                print("\nСоздание 2D-модели с несколькими скважинами...")
                # Создаем 2D-модель с более подробной сеткой и названием директории
                model = MultiWell2DModel(length_x=1000.0, length_y=1000.0, nx=100, ny=100, days=100)

            else:
                if model_type == 'carbonate' and CarbonateModel is None:
                    print("ПРЕДУПРЕЖДЕНИЕ: Модуль CarbonateModel недоступен, используется базовая модель")
                else:
                    print("\nСоздание базовой модели...")
                model = OilFiltrationModel()
        except Exception as e:
            print(f"ОШИБКА при создании модели {model_type}: {e}")
            traceback.print_exc()
            continue

        # Вывод информации о модели
        print_model_info(model)

        # Создаем отдельную директорию для графиков этой модели
        # Для 2D модели создаем отдельную директорию
        if model_type == 'multi_well_2d':
            dir_name = "модель_с_несколькими_скважинами_2d"
            output_dir = os.path.join(args.graphs_dir, dir_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Создана директория для 2D графиков: {output_dir}")
            model_graphs_dir = output_dir
        else:
            model_graphs_dir = create_model_output_directory(args.graphs_dir, model_type, args.rock_type)

        model_dirs[model_type] = model_graphs_dir

        # Отображение расположения скважин для multi_well модели
        if model_type == 'multi_well' or model_type == 'multi_well_2d':
            # Модифицируем метод plot_wells_location, чтобы сохранять график в нужную директорию
            # Либо временно меняем текущую директорию
            original_savefig = plt.savefig

            def custom_savefig(filename, *args, **kwargs):
                """Перенаправляет сохранение графика в указанную директорию"""
                # Если filename содержит путь, используем его как есть
                if os.path.dirname(filename):
                    return original_savefig(filename, *args, **kwargs)

                # Иначе добавляем путь к директории модели
                path = os.path.join(model_graphs_dir, filename)
                return original_savefig(path, *args, **kwargs)

            # Подменяем функцию сохранения
            plt.savefig = custom_savefig

            # Вызываем метод plot_wells_location
            model.plot_wells_location()

            # Для 2D модели дополнительно вызываем специальные методы для 2D визуализации
            if model_type == 'multi_well_2d':
                # Создаем визуализатор
                visualizer = Visualizer(model, output_dir=model_graphs_dir)
                # Вызываем методы 2D визуализации напрямую
                visualizer.plot_wells_location_2d()
                visualizer.plot_saturation_2d(days=[10, 30, 50, 70, 100])
                visualizer.plot_flow_directions(day=50)

            # Восстанавливаем оригинальную функцию
            plt.savefig = original_savefig

        # Запуск моделирования
        print("\n" + "-" * 80)
        simulation_success = run_simulation(model, model_type)
        if not simulation_success:
            print(f"Моделирование {model_type} не удалось, переход к следующей модели")
            continue
        print("-" * 80)

        # Вывод результатов в консоль
        try:
            console = ConsoleOutput(model)
            console.print_saturation_profile(day=50)
            console.print_recovery_factor()
            console.print_front_parameters()
            console.print_pressure_distribution(day=50)
        except Exception as e:
            print(f"ОШИБКА при выводе результатов для модели {model_type}: {e}")
            traceback.print_exc()

        # Визуализация результатов
        try:
            print(f"\nСоздание графиков в директории {model_graphs_dir}...")
            visualizer = Visualizer(model, output_dir=model_graphs_dir)
            visualizer.plot_all()

            # Дополнительные графики для multi_well модели
            if model_type in ['multi_well', 'multi_well_2d']:
                if hasattr(model, 'get_well_production_data'):
                    production_data = model.get_well_production_data()
                    visualizer.plot_well_production(production_data)

            print(f"Графики сохранены в директории {model_graphs_dir}")
        except Exception as e:
            print(f"ОШИБКА при создании графиков для модели {model_type}: {e}")
            traceback.print_exc()

        # Создаем отчет по результатам модели
        try:
            print(f"\nСоздание отчета по результатам модели {model_type}...")
            analyze_model_results(model, model_type, args.rock_type, model_graphs_dir)
        except Exception as e:
            print(f"ОШИБКА при анализе результатов модели {model_type}: {e}")
            traceback.print_exc()

        # Добавляем модель в список для сравнения
        model_data = {
            'model': model,
            'type': model_type
        }

        if model_type == 'carbonate':
            model_data['rock_type'] = args.rock_type

        models_to_compare.append(model_data)

        # Экспорт результатов, если требуется
        if args.export != 'none':
            # Создаем отдельную директорию для данных экспорта модели
            model_export_dir = os.path.join(args.output_dir,
                                            f"{model_type}_{args.rock_type if model_type == 'carbonate' else ''}")
            if not os.path.exists(model_export_dir):
                os.makedirs(model_export_dir)

            export_results(model, args.export, model_export_dir)

    # Если рассчитано более одной модели, делаем сравнительный анализ
    if len(models_to_compare) > 1:
        print("\nСоздание сравнительного анализа моделей...")

        # Создаем директорию для сводных результатов
        summary_dir = create_summarydir(os.path.join(args.graphs_dir, "summary"))

        # Создаем сравнительные графики
        comparison_dir = os.path.join(summary_dir, "comparison")
        create_comparison_graphs(models_to_compare, comparison_dir)

        # Копируем лучшие графики в сводную директорию
        copy_best_graphs(model_dirs, summary_dir)

        # Создаем сравнительный отчет
        compare_models(models_to_compare, summary_dir)

        print(f"Сравнительный анализ сохранен в директории: {summary_dir}")

    # Вывод итоговой информации
    print("\nРезультаты моделирования:")
    for model_type in models_to_run:
        if model_type == 'multi_well_2d':
            model_graphs_dir = os.path.join(args.graphs_dir, "модель_с_несколькими_скважинами_2d")
        else:
            model_graphs_dir = create_model_output_directory(args.graphs_dir, model_type, args.rock_type)
        print(f"- {model_type}: графики в директории {model_graphs_dir}")

    if len(models_to_compare) > 1:
        print(f"- Сравнительный анализ: в директории {os.path.join(args.graphs_dir, 'summary')}")

    # Завершение программы
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"Программа завершена: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Общее время выполнения: {duration:.2f} секунд")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Необработанная ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)