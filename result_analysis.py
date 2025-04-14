import os
import shutil
from datetime import datetime


# Функции для анализа и сравнения результатов различных моделей

def analyze_model_results(model, model_type, rock_type=None, output_dir=None):
    """
    Проводит расширенный анализ результатов моделирования и сохраняет отчет

    Parameters:
    -----------
    model : OilFiltrationModel
        Модель, результаты которой анализируются
    model_type : str
        Тип модели ('basic', 'carbonate', 'multi_well')
    rock_type : str, optional
        Тип породы (для карбонатной модели)
    output_dir : str, optional
        Директория для сохранения отчета
    """
    if output_dir is None:
        output_dir = "."

    # Создаем директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Формируем имя файла отчета
    model_name = model_type
    if rock_type and model_type == 'carbonate':
        model_name += f"_{rock_type}"

    report_file = os.path.join(output_dir, f"отчет_{model_name}.txt")

    # Получаем основные результаты
    recovery_with_cap, recovery_without_cap = model.calculate_recovery_factor()
    breakthrough_with_cap, breakthrough_without_cap = model.get_breakthrough_time()

    # Создаем отчет
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ОТЧЕТ ПО РЕЗУЛЬТАТАМ МОДЕЛИРОВАНИЯ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Тип модели: {model_type}\n")
        if rock_type and model_type == 'carbonate':
            f.write(f"Тип породы: {rock_type}\n")

        f.write("\nПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write(f"Длина пласта: {model.length} м\n")
        f.write(f"Пористость: {model.porosity}\n")
        f.write(f"Вязкость нефти: {model.mu_oil} мПа·с\n")
        f.write(f"Вязкость воды: {model.mu_water} мПа·с\n")
        f.write(f"Начальная водонасыщенность: {model.initial_water_saturation}\n")
        f.write(f"Остаточная нефтенасыщенность: {model.residual_oil_saturation}\n")

        if hasattr(model, 'matrix_porosity'):
            f.write("\nДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ КАРБОНАТНОЙ МОДЕЛИ:\n")
            f.write(f"Пористость матрицы: {model.matrix_porosity}\n")
            f.write(f"Пористость трещин: {model.fracture_porosity}\n")
            f.write(f"Проницаемость матрицы: {model.matrix_permeability} мД\n")
            f.write(f"Проницаемость трещин: {model.fracture_permeability} мД\n")
            f.write(f"Форм-фактор обмена: {model.shape_factor}\n")

        if hasattr(model, 'wells_config'):
            f.write("\nИНФОРМАЦИЯ О СКВАЖИНАХ:\n")
            f.write(f"Количество нагнетательных скважин: {len(model.wells_config['injectors'])}\n")
            f.write(f"Количество добывающих скважин: {len(model.wells_config['producers'])}\n")

            f.write("\nНагнетательные скважины:\n")
            for i, injector in enumerate(model.wells_config['injectors']):
                pos = injector['position']
                rate = injector['rate']
                x_pos = pos * model.dx
                f.write(f"  Скважина {i + 1}: позиция = {x_pos:.1f} м, дебит = {rate:.2f} м³/день\n")

            f.write("\nДобывающие скважины:\n")
            for i, producer in enumerate(model.wells_config['producers']):
                pos = producer['position']
                rate = producer['rate']
                x_pos = pos * model.dx
                f.write(f"  Скважина {i + 1}: позиция = {x_pos:.1f} м, дебит = {rate:.2f} м³/день\n")

        f.write("\nРЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:\n")

        # Результаты по времени прорыва воды
        f.write(f"Время прорыва воды (без кап. эффектов): {breakthrough_without_cap:.1f} дней\n")
        f.write(f"Время прорыва воды (с кап. эффектами): {breakthrough_with_cap:.1f} дней\n")
        f.write(f"Разница во времени прорыва: {(breakthrough_with_cap - breakthrough_without_cap):.1f} дней\n\n")

        # Результаты по коэффициенту нефтеотдачи
        time_points = [10, 30, 50, 70, 100]
        f.write("КОЭФФИЦИЕНТ НЕФТЕОТДАЧИ:\n")
        f.write("{:<15} {:<25} {:<25} {:<25}\n".format("Время (дни)", "Без кап. эффектов", "С кап. эффектами",
                                                       "Разница (%)"))
        f.write("-" * 90 + "\n")

        for day in time_points:
            if day <= model.days:
                time_index = int(day / model.dt)
                recovery_diff_percent = (recovery_with_cap[time_index] - recovery_without_cap[time_index]) / \
                                        recovery_without_cap[time_index] * 100 if recovery_without_cap[
                                                                                      time_index] != 0 else 0
                f.write("{:<15} {:<25.4f} {:<25.4f} {:<25.2f}\n".format(
                    day,
                    recovery_without_cap[time_index],
                    recovery_with_cap[time_index],
                    recovery_diff_percent
                ))

        f.write("\nЗАКЛЮЧЕНИЕ:\n")

        # Анализ влияния капиллярных эффектов
        last_idx = min(int(model.days / model.dt), len(recovery_with_cap) - 1)
        final_recovery_diff = recovery_with_cap[last_idx] - recovery_without_cap[last_idx]

        if final_recovery_diff > 0.05:
            effect = "значительное положительное"
        elif final_recovery_diff > 0.01:
            effect = "умеренное положительное"
        elif final_recovery_diff > -0.01:
            effect = "незначительное"
        elif final_recovery_diff > -0.05:
            effect = "умеренное отрицательное"
        else:
            effect = "значительное отрицательное"

        f.write(f"1. Капиллярные эффекты оказывают {effect} влияние на коэффициент нефтеотдачи.\n")

        # Анализ времени прорыва
        breakthrough_diff = breakthrough_with_cap - breakthrough_without_cap
        if breakthrough_diff > 5:
            bt_effect = "значительно замедляют"
        elif breakthrough_diff > 1:
            bt_effect = "умеренно замедляют"
        elif breakthrough_diff > -1:
            bt_effect = "практически не влияют на"
        elif breakthrough_diff > -5:
            bt_effect = "умеренно ускоряют"
        else:
            bt_effect = "значительно ускоряют"

        f.write(f"2. Капиллярные эффекты {bt_effect} прорыв воды.\n")

        # Рекомендации
        f.write("\nРЕКОМЕНДАЦИИ:\n")

        if model_type == 'basic':
            f.write("1. Для более точного моделирования рекомендуется учитывать капиллярные эффекты.\n")
            f.write("2. Рассмотреть возможность применения более сложных моделей (карбонатная, многоскважинная).\n")
        elif model_type == 'carbonate':
            f.write("1. Учет двойной пористости критически важен для карбонатных коллекторов.\n")
            f.write("2. Рекомендуется проводить лабораторные исследования для уточнения параметров матрицы и трещин.\n")
            f.write(
                f"3. Для данного типа породы ({rock_type}) особенно важно учитывать влияние капиллярных эффектов.\n")
        elif model_type == 'multi_well':
            f.write("1. Оптимизировать расположение скважин для максимизации охвата пласта.\n")
            f.write("2. Рассмотреть возможность изменения режимов работы скважин для улучшения показателей добычи.\n")

        f.write("\nДата формирования отчета: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"Отчет сохранен в файле: {report_file}")
    return report_file


def compare_models(models_data, output_dir=None):
    """
    Сравнивает результаты нескольких моделей и создает сводный отчет

    Parameters:
    -----------
    models_data : list of dict
        Список словарей с данными моделей, каждый словарь содержит:
        - 'model': экземпляр модели
        - 'type': тип модели
        - 'rock_type': тип породы (для карбонатных)
    output_dir : str, optional
        Директория для сохранения отчета
    """
    if output_dir is None:
        output_dir = "."

    # Создаем директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Формируем имя файла отчета
    report_file = os.path.join(output_dir, "сравнение_моделей.txt")

    # Создаем сравнительный отчет
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ МОДЕЛИРОВАНИЯ\n")
        f.write("=" * 100 + "\n\n")

        f.write("СРАВНИВАЕМЫЕ МОДЕЛИ:\n")
        for i, model_data in enumerate(models_data):
            model_type = model_data['type']
            rock_type = model_data.get('rock_type', '')
            model = model_data['model']

            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"

            f.write(f"{i + 1}. {model_name}\n")

        f.write("\nСРАВНЕНИЕ ПАРАМЕТРОВ МОДЕЛЕЙ:\n")
        f.write("-" * 100 + "\n")
        f.write("{:<30} ".format("Параметр"))

        for model_data in models_data:
            model_type = model_data['type']
            rock_type = model_data.get('rock_type', '')
            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"
            f.write("{:<20} ".format(model_name))
        f.write("\n" + "-" * 100 + "\n")

        # Общие параметры
        params = [
            ('Длина пласта, м', 'length'),
            ('Пористость', 'porosity'),
            ('Вязкость нефти, мПа·с', 'mu_oil'),
            ('Вязкость воды, мПа·с', 'mu_water'),
            ('Начальная водонасыщ.', 'initial_water_saturation'),
            ('Остаточная нефтенасыщ.', 'residual_oil_saturation'),
            ('Число узлов сетки', 'nx'),
            ('Давление входа, МПа', 'entry_pressure'),
            ('Индекс распред. пор', 'pore_distribution_index')
        ]

        for param_name, param_attr in params:
            f.write("{:<30} ".format(param_name))
            for model_data in models_data:
                model = model_data['model']
                if hasattr(model, param_attr):
                    value = getattr(model, param_attr)
                    if isinstance(value, float):
                        f.write("{:<20.4f} ".format(value))
                    else:
                        f.write("{:<20} ".format(str(value)))
                else:
                    f.write("{:<20} ".format("-"))
            f.write("\n")

        # Специальные параметры для карбонатной модели
        f.write("\nПАРАМЕТРЫ КАРБОНАТНОЙ МОДЕЛИ:\n")
        f.write("-" * 100 + "\n")
        carb_params = [
            ('Пористость матрицы', 'matrix_porosity'),
            ('Пористость трещин', 'fracture_porosity'),
            ('Проницаемость матрицы, мД', 'matrix_permeability'),
            ('Проницаемость трещин, мД', 'fracture_permeability'),
            ('Форм-фактор обмена', 'shape_factor')
        ]

        for param_name, param_attr in carb_params:
            f.write("{:<30} ".format(param_name))
            for model_data in models_data:
                model = model_data['model']
                if hasattr(model, param_attr):
                    value = getattr(model, param_attr)
                    if isinstance(value, float):
                        f.write("{:<20.4f} ".format(value))
                    else:
                        f.write("{:<20} ".format(str(value)))
                else:
                    f.write("{:<20} ".format("-"))
            f.write("\n")

        # Сравнение результатов
        f.write("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ:\n")
        f.write("-" * 100 + "\n")

        # Время прорыва
        f.write("ВРЕМЯ ПРОРЫВА ВОДЫ (дни):\n")
        f.write("{:<30} ".format("Модель"))
        f.write("{:<20} {:<20} {:<20}\n".format("Без кап. эффектов", "С кап. эффектами", "Разница"))
        f.write("-" * 100 + "\n")

        for model_data in models_data:
            model_type = model_data['type']
            rock_type = model_data.get('rock_type', '')
            model = model_data['model']

            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"

            breakthrough_with_cap, breakthrough_without_cap = model.get_breakthrough_time()
            breakthrough_diff = breakthrough_with_cap - breakthrough_without_cap

            f.write("{:<30} ".format(model_name))
            f.write("{:<20.1f} {:<20.1f} {:<20.1f}\n".format(
                breakthrough_without_cap,
                breakthrough_with_cap,
                breakthrough_diff
            ))

        # Коэффициент нефтеотдачи
        f.write("\nКОЭФФИЦИЕНТ НЕФТЕОТДАЧИ НА 50-Й ДЕНЬ:\n")
        f.write("{:<30} ".format("Модель"))
        f.write("{:<20} {:<20} {:<20}\n".format("Без кап. эффектов", "С кап. эффектами", "Разница (%)"))
        f.write("-" * 100 + "\n")

        day = 50
        for model_data in models_data:
            model_type = model_data['type']
            rock_type = model_data.get('rock_type', '')
            model = model_data['model']

            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"

            recovery_with_cap, recovery_without_cap = model.calculate_recovery_factor()
            time_index = min(int(day / model.dt), len(recovery_with_cap) - 1)

            recovery_diff_percent = (recovery_with_cap[time_index] - recovery_without_cap[time_index]) / \
                                    recovery_without_cap[time_index] * 100 if recovery_without_cap[
                                                                                  time_index] != 0 else 0

            f.write("{:<30} ".format(model_name))
            f.write("{:<20.4f} {:<20.4f} {:<20.2f}\n".format(
                recovery_without_cap[time_index],
                recovery_with_cap[time_index],
                recovery_diff_percent
            ))

        # Заключение и рекомендации
        f.write("\nЗАКЛЮЧЕНИЕ:\n")

        # Определяем модель с лучшим коэффициентом нефтеотдачи
        best_recovery_model = None
        best_recovery = -1

        for model_data in models_data:
            model = model_data['model']
            recovery_with_cap, _ = model.calculate_recovery_factor()
            time_index = min(int(model.days / model.dt), len(recovery_with_cap) - 1)

            if recovery_with_cap[time_index] > best_recovery:
                best_recovery = recovery_with_cap[time_index]
                best_recovery_model = model_data

        if best_recovery_model:
            model_type = best_recovery_model['type']
            rock_type = best_recovery_model.get('rock_type', '')

            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"

            f.write(f"1. Наилучший коэффициент нефтеотдачи демонстрирует модель: {model_name} ({best_recovery:.4f}).\n")

        # Определяем модель с наибольшим влиянием капиллярных эффектов
        max_cap_effect_model = None
        max_cap_effect = -1

        for model_data in models_data:
            model = model_data['model']
            recovery_with_cap, recovery_without_cap = model.calculate_recovery_factor()
            time_index = min(int(model.days / model.dt), len(recovery_with_cap) - 1)

            cap_effect = abs(recovery_with_cap[time_index] - recovery_without_cap[time_index])

            if cap_effect > max_cap_effect:
                max_cap_effect = cap_effect
                max_cap_effect_model = model_data

        if max_cap_effect_model:
            model_type = max_cap_effect_model['type']
            rock_type = max_cap_effect_model.get('rock_type', '')

            model_name = model_type
            if rock_type and model_type == 'carbonate':
                model_name += f" ({rock_type})"

            f.write(
                f"2. Наибольшее влияние капиллярных эффектов наблюдается в модели: {model_name} (разница: {max_cap_effect:.4f}).\n")

        f.write("\nРЕКОМЕНДАЦИИ:\n")

        if any(model_data['type'] == 'carbonate' for model_data in models_data):
            f.write("1. Для карбонатных коллекторов необходимо обязательно учитывать эффекты двойной пористости.\n")

        f.write(
            "2. Для всех типов коллекторов рекомендуется учитывать капиллярные эффекты для повышения точности прогнозов.\n")

        if any(model_data['type'] == 'multi_well' for model_data in models_data):
            f.write(
                "3. В моделях с несколькими скважинами следует оптимизировать их расположение с учетом капиллярных эффектов.\n")

        f.write("\nДата формирования отчета: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"Сравнительный отчет сохранен в файле: {report_file}")
    return report_file


def create_summarydir(root_dir="results_summary"):
    """
    Создает директорию для сводных результатов и копирует туда лучшие графики

    Parameters:
    -----------
    root_dir : str
        Путь к директории для сводных результатов

    Returns:
    --------
    str
        Путь к созданной директории
    """
    # Создаем директорию, если её нет
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"Создана директория для сводных результатов: {root_dir}")

    # Создаем вложенные директории для разных типов результатов
    subdirs = ['recovery', 'saturation', 'pressure', 'sensitivity', 'well_production']

    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    return root_dir

def copy_best_graphs(model_dirs, summary_dir):
    """
    Копирует наиболее важные графики из директорий моделей в сводную директорию

    Parameters:
    -----------
    model_dirs : dict
        Словарь с путями к директориям моделей {model_type: path}
    summary_dir : str
        Путь к директории для сводных результатов
    """
    # Проверяем, что сводная директория существует
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # Создаем директорию для сравнительных графиков
    comparison_dir = os.path.join(summary_dir, "comparison")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    # Список ключевых графиков и их целевые директории
    key_graphs = [
        {'source': 'recovery_factor.png', 'target': os.path.join(summary_dir, 'recovery')},
        {'source': 'saturation_profiles.png', 'target': os.path.join(summary_dir, 'saturation')},
        {'source': 'saturation_evolution.png', 'target': os.path.join(summary_dir, 'saturation')},
        {'source': 'pressure_profiles.png', 'target': os.path.join(summary_dir, 'pressure')},
        {'source': 'saturation_difference.png', 'target': os.path.join(summary_dir, 'saturation')},
        {'source': 'saturation_3d.png', 'target': os.path.join(summary_dir, 'saturation')},
        {'source': 'sensitivity_entry_pressure.png', 'target': os.path.join(summary_dir, 'sensitivity')},
        {'source': 'sensitivity_wettability_factor.png', 'target': os.path.join(summary_dir, 'sensitivity')},
        {'source': 'well_production.png', 'target': os.path.join(summary_dir, 'well_production')},
        {'source': 'total_production.png', 'target': os.path.join(summary_dir, 'well_production')},
        {'source': 'filtration_velocities.png', 'target': os.path.join(summary_dir, 'saturation')},
        {'source': 'capillary_number.png', 'target': os.path.join(summary_dir, 'saturation')}
    ]

    # Копируем графики из каждой директории модели
    for model_type, model_dir in model_dirs.items():
        # Проверяем, что директория существует
        if not os.path.exists(model_dir):
            print(f"Директория модели не найдена: {model_dir}")
            continue

        # Копируем каждый ключевой график, если он существует
        for graph in key_graphs:
            source_path = os.path.join(model_dir, graph['source'])

            if os.path.exists(source_path):
                # Создаем целевую директорию, если её нет
                if not os.path.exists(graph['target']):
                    os.makedirs(graph['target'])

                # Формируем имя для файла с префиксом типа модели
                target_filename = f"{model_type}_{graph['source']}"
                target_path = os.path.join(graph['target'], target_filename)

                # Копируем файл
                shutil.copy2(source_path, target_path)
                print(f"Скопирован график: {source_path} -> {target_path}")

    print(f"Лучшие графики скопированы в директорию: {summary_dir}")


def create_comparison_graphs(models_data, output_dir):
    """
    Создает сравнительные графики для разных моделей

    Parameters:
    -----------
    models_data : list of dict
        Список словарей с данными моделей, каждый словарь содержит:
        - 'model': экземпляр модели
        - 'type': тип модели
        - 'rock_type': тип породы (для карбонатных)
    output_dir : str
        Путь к директории для сохранения графиков
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Проверяем, что директория существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Сравнение коэффициентов нефтеотдачи
    plt.figure(figsize=(12, 8))

    # Верхний график - без капиллярных эффектов
    plt.subplot(2, 1, 1)
    for model_data in models_data:
        model = model_data['model']
        model_type = model_data['type']
        rock_type = model_data.get('rock_type', '')

        model_name = model_type
        if rock_type and model_type == 'carbonate':
            model_name += f" ({rock_type})"

        recovery_with_cap, recovery_without_cap = model.calculate_recovery_factor()
        plt.plot(model.t, recovery_without_cap, label=model_name)

    plt.xlabel('Время (дни)')
    plt.ylabel('Коэффициент нефтеотдачи')
    plt.title('Сравнение коэффициентов нефтеотдачи без учета капиллярных эффектов')
    plt.grid(True)
    plt.legend()

    # Нижний график - с капиллярными эффектами
    plt.subplot(2, 1, 2)
    for model_data in models_data:
        model = model_data['model']
        model_type = model_data['type']
        rock_type = model_data.get('rock_type', '')

        model_name = model_type
        if rock_type and model_type == 'carbonate':
            model_name += f" ({rock_type})"

        recovery_with_cap, recovery_without_cap = model.calculate_recovery_factor()
        plt.plot(model.t, recovery_with_cap, label=model_name)

    plt.xlabel('Время (дни)')
    plt.ylabel('Коэффициент нефтеотдачи')
    plt.title('Сравнение коэффициентов нефтеотдачи с учетом капиллярных эффектов')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_comparison.png'))
    plt.close()

    # 2. Сравнение профилей насыщенности на 50-й день
    plt.figure(figsize=(12, 8))

    # Верхний график - без капиллярных эффектов
    plt.subplot(2, 1, 1)
    for model_data in models_data:
        model = model_data['model']
        model_type = model_data['type']
        rock_type = model_data.get('rock_type', '')

        model_name = model_type
        if rock_type and model_type == 'carbonate':
            model_name += f" ({rock_type})"

        day = 50
        time_index = min(int(day / model.dt), model.nt - 1)

        # Прореживаем данные для более четкого графика
        step = max(1, model.nx // 100)
        plt.plot(model.x[::step], model.Sw_without_cap[time_index, ::step], label=model_name)

    plt.xlabel('Расстояние (м)')
    plt.ylabel('Водонасыщенность')
    plt.title(f'Сравнение профилей насыщенности без учета капиллярных эффектов (день {day})')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)

    # Нижний график - с капиллярными эффектами
    plt.subplot(2, 1, 2)
    for model_data in models_data:
        model = model_data['model']
        model_type = model_data['type']
        rock_type = model_data.get('rock_type', '')

        model_name = model_type
        if rock_type and model_type == 'carbonate':
            model_name += f" ({rock_type})"

        day = 50
        time_index = min(int(day / model.dt), model.nt - 1)

        # Прореживаем данные для более четкого графика
        step = max(1, model.nx // 100)
        plt.plot(model.x[::step], model.Sw_with_cap[time_index, ::step], label=model_name)

    plt.xlabel('Расстояние (м)')
    plt.ylabel('Водонасыщенность')
    plt.title(f'Сравнение профилей насыщенности с учетом капиллярных эффектов (день {day})')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'saturation_profiles_comparison.png'))
    plt.close()

    # 3. Сравнение времени прорыва
    plt.figure(figsize=(10, 6))

    model_names = []
    bt_without_cap = []
    bt_with_cap = []

    for model_data in models_data:
        model = model_data['model']
        model_type = model_data['type']
        rock_type = model_data.get('rock_type', '')

        model_name = model_type
        if rock_type and model_type == 'carbonate':
            model_name += f"\n({rock_type})"

        breakthrough_with_cap, breakthrough_without_cap = model.get_breakthrough_time()

        model_names.append(model_name)
        bt_without_cap.append(breakthrough_without_cap)
        bt_with_cap.append(breakthrough_with_cap)

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, bt_without_cap, width, label='Без капиллярных эффектов')
    plt.bar(x + width / 2, bt_with_cap, width, label='С капиллярными эффектами')

    plt.xlabel('Модель')
    plt.ylabel('Время прорыва воды (дни)')
    plt.title('Сравнение времени прорыва воды для разных моделей')
    plt.xticks(x, model_names)
    plt.legend()

    plt.grid(True, axis='y')

    # Добавляем значения на столбцы
    for i, v in enumerate(bt_without_cap):
        plt.text(i - width / 2, v + 0.5, f'{v:.1f}', ha='center')

    for i, v in enumerate(bt_with_cap):
        plt.text(i + width / 2, v + 0.5, f'{v:.1f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'breakthrough_comparison.png'))
    plt.close()

    print(f"Сравнительные графики созданы в директории: {output_dir}")