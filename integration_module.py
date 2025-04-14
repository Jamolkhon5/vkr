# integration_module.py - Модуль для интеграции с гидродинамическими симуляторами
import json
import numpy as np
import pandas as pd
import os


class SimulatorInterface:
    """Класс для интеграции с гидродинамическими симуляторами"""

    def __init__(self, model):
        self.model = model
        self.output_dir = "output"

        # Создаем папку для выходных файлов, если ее нет
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def export_to_rn_kim(self):
        """Экспорт данных в формате РН-КИМ"""
        print("Экспорт данных в формате, совместимом с РН-КИМ...")

        # Экспорт модели относительной проницаемости
        self.export_rel_perm_table()

        # Экспорт модели капиллярного давления
        self.export_capillary_table()

        # Экспорт результатов моделирования
        self.export_saturation_profiles()

        print(f"Экспорт завершен. Файлы сохранены в директории {self.output_dir}")

    def export_rel_perm_table(self):
        """Экспорт таблицы относительной проницаемости"""
        # Создаем таблицу насыщенности
        Sw = np.linspace(0.0, 1.0, 20)
        Krw = np.zeros_like(Sw)
        Kro = np.zeros_like(Sw)

        # Заполняем значения
        for i, sw in enumerate(Sw):
            Krw[i] = self.model.relative_permeability_water(sw)
            Kro[i] = self.model.relative_permeability_oil(sw)

        # Создаем DataFrame
        df = pd.DataFrame({
            'Sw': Sw,
            'Krw': Krw,
            'Kro': Kro
        })

        # Сохраняем в CSV
        file_path = os.path.join(self.output_dir, 'rel_perm_table.csv')
        df.to_csv(file_path, index=False)

        # Создаем также файл в формате JSON для использования в РН-КИМ
        rel_perm_json = {
            "metadata": {
                "version": "1.0",
                "model_type": "table",
                "description": "Relative permeability table for carbonate reservoir"
            },
            "data": df.to_dict(orient='records')
        }

        json_path = os.path.join(self.output_dir, 'rel_perm_table.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rel_perm_json, f, indent=4)

    def export_capillary_table(self):
        """Экспорт таблицы капиллярного давления"""
        # Создаем таблицу насыщенности
        Sw = np.linspace(0.0, 1.0, 20)
        Pc = np.zeros_like(Sw)

        # Заполняем значения
        for i, sw in enumerate(Sw):
            Pc[i] = self.model.capillary_pressure(sw)

        # Создаем DataFrame
        df = pd.DataFrame({
            'Sw': Sw,
            'Pc_MPa': Pc
        })

        # Сохраняем в CSV
        file_path = os.path.join(self.output_dir, 'capillary_table.csv')
        df.to_csv(file_path, index=False)

        # Создаем также файл в формате JSON для использования в РН-КИМ
        cap_json = {
            "metadata": {
                "version": "1.0",
                "model_type": "table",
                "description": "Capillary pressure table for carbonate reservoir"
            },
            "data": df.to_dict(orient='records')
        }

        json_path = os.path.join(self.output_dir, 'capillary_table.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cap_json, f, indent=4)

    def export_saturation_profiles(self):
        """Экспорт профилей насыщенности"""
        # Выбираем конкретные моменты времени для экспорта
        time_points = [10, 50, 100]

        for day in time_points:
            if day > self.model.days:
                continue

            time_index = int(day / self.model.dt)

            # Создаем DataFrame с профилем насыщенности
            df = pd.DataFrame({
                'Distance_m': self.model.x,
                'Sw_without_cap': self.model.Sw_without_cap[time_index, :],
                'Sw_with_cap': self.model.Sw_with_cap[time_index, :]
            })

            # Сохраняем в CSV
            file_path = os.path.join(self.output_dir, f'saturation_profile_day_{day}.csv')
            df.to_csv(file_path, index=False)

    def generate_rn_kim_config(self):
        """Генерация конфигурационного файла для РН-КИМ"""
        config = {
            "model_name": "Carbonate_Capillary_Model",
            "description": "Модель фильтрации с улучшенным учетом капиллярных эффектов для карбонатных коллекторов",
            "grid": {
                "nx": self.model.nx,
                "ny": 1,
                "nz": 1,
                "dx": self.model.dx,
                "dy": 10.0,  # Условное значение
                "dz": 10.0  # Условное значение
            },
            "rock_properties": {
                "porosity": self.model.porosity,
                "permeability": {
                    "kx": 1.0,  # В Дарси
                    "ky": 1.0,
                    "kz": 0.1
                }
            },
            "fluid_properties": {
                "viscosity": {
                    "oil": self.model.mu_oil,
                    "water": self.model.mu_water
                }
            },
            "rel_perm_model": {
                "type": "table",
                "file": "rel_perm_table.json"
            },
            "capillary_model": {
                "type": "table",
                "file": "capillary_table.json"
            },
            "dual_porosity": {
                "enabled": True,
                "matrix_porosity": self.model.matrix_porosity,
                "fracture_porosity": self.model.fracture_porosity,
                "shape_factor": self.model.shape_factor
            }
        }

        # Сохраняем конфигурационный файл
        config_path = os.path.join(self.output_dir, 'rn_kim_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"Конфигурационный файл для РН-КИМ сохранен в {config_path}")

    def export_to_eclipse(self):
        """Экспорт данных в формате Eclipse"""
        print("Экспорт данных в формате, совместимом с Eclipse...")

        # Создаем INCLUDE-файлы для Eclipse
        self.generate_eclipse_include_files()

        print(f"Экспорт для Eclipse завершен. Файлы сохранены в директории {self.output_dir}")

    def generate_eclipse_include_files(self):
        """Генерация INCLUDE-файлов для Eclipse"""
        # Файл PROPS с относительными проницаемостями и капиллярным давлением
        props_content = "-- Таблица относительных проницаемостей и капиллярного давления\n"
        props_content += "SWOF\n"

        # Генерируем таблицу
        Sw = np.linspace(self.model.initial_water_saturation, 1.0, 20)
        for sw in Sw:
            krw = self.model.relative_permeability_water(sw)
            kro = self.model.relative_permeability_oil(sw)
            pc = self.model.capillary_pressure(sw) * 1.0  # Переводим МПа в барс (атм)

            props_content += f"{sw:.4f} {krw:.4f} {kro:.4f} {pc:.4f}\n"

        props_content += "/\n"

        # Сохраняем файл PROPS
        props_path = os.path.join(self.output_dir, 'PROPS.inc')
        with open(props_path, 'w') as f:
            f.write(props_content)

        # Файл SOLUTION с начальными условиями
        solution_content = "-- Начальные условия\n"
        solution_content += "SWAT\n"

        # Добавляем начальную водонасыщенность для всех ячеек
        for i in range(self.model.nx):
            solution_content += f"{self.model.initial_water_saturation:.4f}\n"

        solution_content += "/\n"

        # Сохраняем файл SOLUTION
        solution_path = os.path.join(self.output_dir, 'SOLUTION.inc')
        with open(solution_path, 'w') as f:
            f.write(solution_content)