# data_loader.py - Загрузка и обработка данных из CSV-файлов
import pandas as pd
import numpy as np
import os


class DataLoader:
    """Класс для загрузки и обработки данных из CSV-файлов"""

    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        self.rock_data = None
        self.capillary_data = None
        self.perm_data = None
        self.pvt_data = None
        self.production_data = None

    def load_all_data(self):
        """Загрузка всех доступных данных"""
        print("Загрузка данных...")
        self.load_rock_properties()
        self.load_capillary_pressure()
        self.load_relative_permeability()
        self.load_pvt_data()
        self.load_production_data()
        print("Данные успешно загружены.")

    def load_rock_properties(self):
        """Загрузка данных о свойствах породы"""
        file_path = os.path.join(self.data_dir, 'rock_properties.csv')
        if os.path.exists(file_path):
            # Указываем правильную кодировку cp1252
            self.rock_data = pd.read_csv(file_path, encoding='cp1252')
            print(f"Загружено {len(self.rock_data)} записей о свойствах породы.")
        else:
            print(f"Файл {file_path} не найден.")

    def load_capillary_pressure(self):
        """Загрузка данных о капиллярном давлении"""
        file_path = os.path.join(self.data_dir, 'capillary_pressure.csv')
        if os.path.exists(file_path):
            self.capillary_data = pd.read_csv(file_path)
            print(f"Загружено {len(self.capillary_data)} записей о капиллярном давлении.")
        else:
            print(f"Файл {file_path} не найден.")

    def load_relative_permeability(self):
        """Загрузка данных об относительной проницаемости"""
        file_path = os.path.join(self.data_dir, 'relative_perm.csv')
        if os.path.exists(file_path):
            self.perm_data = pd.read_csv(file_path)
            print(f"Загружено {len(self.perm_data)} записей об относительной проницаемости.")
        else:
            print(f"Файл {file_path} не найден.")

    def load_pvt_data(self):
        """Загрузка PVT-данных флюидов"""
        file_path = os.path.join(self.data_dir, 'pvt_data.csv')
        if os.path.exists(file_path):
            self.pvt_data = pd.read_csv(file_path)
            print(f"Загружено {len(self.pvt_data)} записей PVT-данных.")
        else:
            print(f"Файл {file_path} не найден.")

    def load_production_data(self):
        """Загрузка данных о добыче"""
        file_path = os.path.join(self.data_dir, 'production_data.csv')
        if os.path.exists(file_path):
            self.production_data = pd.read_csv(file_path)
            print(f"Загружено {len(self.production_data)} записей о добыче.")
        else:
            print(f"Файл {file_path} не найден.")

    def get_brooks_corey_params(self, rock_type=None):
        """Получение параметров модели Брукса-Кори для заданного типа породы"""
        if self.rock_data is None or self.capillary_data is None:
            print("Данные о породе или капиллярном давлении не загружены.")
            return None

        # Фильтрация данных по типу породы, если указан
        if rock_type:
            samples = self.rock_data[self.rock_data['Rock_Type'] == rock_type]['ID_Sample'].tolist()
        else:
            samples = self.rock_data['ID_Sample'].tolist()

        # Получение уникальных образцов
        unique_samples = set(samples).intersection(set(self.capillary_data['ID_Sample'].unique()))

        # Расчет параметров для каждого образца
        results = {}
        for sample in unique_samples:
            sample_data = self.capillary_data[self.capillary_data['ID_Sample'] == sample]

            # Только для дренирования (Pc_Drainage_MPa)
            Sw = sample_data['Water_Saturation'].values
            Pc = sample_data['Pc_Drainage_MPa'].values

            # Фильтрация нулевых значений
            valid_idx = (Pc > 0) & (Sw > 0) & (Sw < 1)
            Sw = Sw[valid_idx]
            Pc = Pc[valid_idx]

            if len(Sw) < 3:
                continue

            # Вычисление параметров Брукса-Кори
            try:
                # Нахождение параметров для Pc = Pe * (Sw_eff)^(-1/lambda)
                # Линеаризация: ln(Pc) = ln(Pe) - (1/lambda) * ln(Sw_eff)

                # Расчет эффективной насыщенности
                # Использование немного меньшего значения, чем минимум, чтобы избежать нулей в Sw_eff
                Swc = max(0.01, min(Sw) * 0.95)  # Добавляем небольшой запас
                Sw_eff = (Sw - Swc) / (1 - Swc)

                # Фильтрация слишком малых значений, чтобы избежать проблем с логарифмом
                valid_for_log = Sw_eff > 0.001
                if sum(valid_for_log) < 3:
                    continue

                Sw_eff_filtered = Sw_eff[valid_for_log]
                Pc_filtered = Pc[valid_for_log]

                # Линеаризация
                log_Pc = np.log(Pc_filtered)
                log_Sw_eff = np.log(Sw_eff_filtered)  # Теперь здесь нет нулей

                # Линейная регрессия
                valid = ~np.isnan(log_Sw_eff) & ~np.isnan(log_Pc) & ~np.isinf(log_Sw_eff) & ~np.isinf(log_Pc)
                if sum(valid) < 3:
                    continue

                slope, intercept = np.polyfit(log_Sw_eff[valid], log_Pc[valid], 1)

                # Расчет параметров
                lambda_value = -1 / slope
                Pe = np.exp(intercept)

                # Добавление в результаты
                results[sample] = {
                    'entry_pressure': Pe,
                    'pore_distribution_index': lambda_value,
                    'initial_water_saturation': Swc
                }

            except Exception as e:
                print(f"Ошибка при обработке образца {sample}: {e}")

        # Если не удалось вычислить параметры ни для одного образца, возвращаем стандартные значения
        if not results:
            print(
                f"Не удалось вычислить параметры Брукса-Кори для типа породы {rock_type}. Используем стандартные значения.")
            default_sample = "DEFAULT"
            results[default_sample] = {
                'entry_pressure': 0.5,  # МПа, стандартное давление входа
                'pore_distribution_index': 2.0,  # Стандартный индекс распределения пор
                'initial_water_saturation': 0.1  # Стандартная начальная водонасыщенность
            }

        return results

    def get_relative_permeability_data(self, sample_id):
        """Получение данных об относительной проницаемости для образца"""
        if self.perm_data is None:
            print("Данные об относительной проницаемости не загружены.")
            return None

        if sample_id not in self.perm_data['ID_Sample'].unique():
            print(f"Образец {sample_id} не найден в данных.")
            return None

        sample_data = self.perm_data[self.perm_data['ID_Sample'] == sample_id]

        # Выбираем только нужные столбцы и сортируем по насыщенности
        result = sample_data[['Water_Saturation', 'Krw_Drainage', 'Kro_Drainage']].copy()
        result.sort_values('Water_Saturation', inplace=True)

        # Преобразуем в numpy массивы для удобства
        Sw = result['Water_Saturation'].values
        Krw = result['Krw_Drainage'].values
        Kro = result['Kro_Drainage'].values

        return Sw, Krw, Kro

    def get_average_parameters(self, rock_type=None):
        """Получение усредненных параметров для заданного типа породы"""
        if self.rock_data is None:
            print("Данные о породе не загружены.")
            return None

        # Фильтрация данных по типу породы, если указан
        if rock_type:
            data = self.rock_data[self.rock_data['Rock_Type'] == rock_type]
        else:
            data = self.rock_data

        if len(data) == 0:
            print(f"Нет данных для типа породы {rock_type}.")
            return None

        # Расчет средних значений
        avg_porosity = data['Porosity_fr'].mean()
        avg_permeability = data['Permeability_mD'].mean()
        avg_wettability = data['Wettability_Index'].mean()

        return {
            'porosity': avg_porosity,
            'permeability': avg_permeability,
            'wettability_index': avg_wettability
        }

    def get_pvt_properties(self, pressure=None):
        """Получение PVT-свойств флюидов при заданном давлении"""
        if self.pvt_data is None:
            print("PVT-данные не загружены.")
            return None

        if pressure is None:
            # Возвращаем первую строку данных
            row = self.pvt_data.iloc[0]
        else:
            # Находим ближайшее давление
            idx = (self.pvt_data['Pressure_MPa'] - pressure).abs().idxmin()
            row = self.pvt_data.iloc[idx]

        return {
            'pressure': row['Pressure_MPa'],
            'oil_viscosity': row['Oil_Viscosity_cP'],
            'water_viscosity': 1.0,  # Предполагаем постоянную вязкость воды
            'oil_density': row['Oil_Density_kg_m3'],
            'gas_density': row['Gas_Density_kg_m3']
        }