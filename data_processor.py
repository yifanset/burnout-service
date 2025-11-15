# data_processor.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
from config import *

class DataProcessor:
    def __init__(self):
        self.df = None
    
    def load_raw_data(self):
        """Загрузка исходных данных"""
        self.df = pd.read_excel(RAW_DATA_PATH, sheet_name='Лист1', header=1)
        print(f"Загружено {len(self.df)} записей")
        print(f"Столбцы в данных: {list(self.df.columns)}")
        return self
    
    def _get_column_name(self, possible_names):
        """Поиск столбца по возможным названиям"""
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def clean_data(self):
        """Очистка и исправление данных"""
        # Исправление опечаток в подчинении
        sub_column = self._get_column_name([
            'В подчиненнии сотрудники', 
            'В подчинении сотрудники'
        ])
        if sub_column:
            self.df[sub_column] = self.df[sub_column].replace({
                'Сотрутник': 'Сотрудник'
            })
        return self
    
    def process_gender(self):
        """Определение пола по ФИО"""
        def detect_gender(name):
            if pd.isna(name):
                return 'не указано'
            name_parts = str(name).split()
            if not name_parts:
                return 'не указано'
            first_name = name_parts[0]
            if re.search(r'вна$|ова$|ева$|ина$|ская$', first_name):
                return 'жен'
            elif re.search(r'ов$|ев$|ин$|ский$|ой$', first_name):
                return 'муж'
            return 'не указано'
        
        self.df['пол'] = self.df['ФИО'].apply(detect_gender)
        return self
    
    def process_experience(self):
        """Преобразование стажа в месяцы"""
        def experience_to_months(exp):
            if pd.isna(exp) or exp == 'нет':
                return 0
            exp_str = str(exp)
            years = re.findall(r'(\d+)\s*год', exp_str)
            months = re.findall(r'(\d+)\s*месяц', exp_str)
            total_months = 0
            if years:
                total_months += int(years[0]) * 12
            if months:
                total_months += int(months[0])
            return total_months
        
        self.df['Стаж_месяцы'] = self.df['Стаж'].apply(experience_to_months)
        return self
    
    def process_kpi(self):
        """Обработка KPI показателей - сохраняем все значения"""
        # Сначала обрабатываем KPI столбцы
        for col in KPI_COLUMNS:
            if col in self.df.columns:
                # Исправляем FutureWarning
                self.df[col] = self.df[col].replace({'нет': np.nan})
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Вместо агрегации создаем признаки на основе всех KPI значений
        available_kpi_cols = [col for col in KPI_COLUMNS if col in self.df.columns]
        
        if available_kpi_cols:
            # Сохраняем все исходные KPI значения как отдельные признаки
            # Они уже обработаны выше, просто переименуем для ясности
            kpi_rename = {col: f'KPI_{col}' for col in available_kpi_cols}
            self.df = self.df.rename(columns=kpi_rename)
            
            # Создаем дополнительные признаки на основе KPI
            kpi_new_cols = [f'KPI_{col}' for col in available_kpi_cols]
            
            # Количество заполненных KPI показателей
            self.df['KPI_заполнено_показателей'] = self.df[kpi_new_cols].notna().sum(axis=1)
            
            # Стабильность KPI (стандартное отклонение)
            self.df['KPI_стабильность'] = self.df[kpi_new_cols].std(axis=1)
            
            # Минимальный и максимальный KPI
            self.df['KPI_мин'] = self.df[kpi_new_cols].min(axis=1)
            self.df['KPI_макс'] = self.df[kpi_new_cols].max(axis=1)
            
            # Размах KPI (макс - мин)
            self.df['KPI_размах'] = self.df['KPI_макс'] - self.df['KPI_мин']
            
            # Расчет тренда KPI (наклон линейной регрессии)
            def calculate_trend(row):
                kpi_values = row[kpi_new_cols].values
                try:
                    kpi_values_float = kpi_values.astype(float)
                except (ValueError, TypeError):
                    return 0
                    
                if np.all(np.isnan(kpi_values_float)):
                    return 0
                    
                valid_indices = ~np.isnan(kpi_values_float)
                if np.sum(valid_indices) < 2:
                    return 0
                    
                x = np.where(valid_indices)[0]
                y = kpi_values_float[valid_indices]
                try:
                    trend = np.polyfit(x, y, 1)[0]
                    return trend
                except:
                    return 0
            
            self.df['KPI_тренд'] = self.df.apply(calculate_trend, axis=1)
            
            # Последний доступный KPI
            def get_last_kpi(row):
                kpi_values = row[kpi_new_cols].values
                # Ищем последний не-NaN значение (с конца массива)
                for i in range(len(kpi_values)-1, -1, -1):
                    if not pd.isna(kpi_values[i]):
                        return kpi_values[i]
                return np.nan
            
            self.df['KPI_последний'] = self.df.apply(get_last_kpi, axis=1)
            
        else:
            # Если нет KPI данных, создаем заглушки
            self.df['KPI_заполнено_показателей'] = 0
            self.df['KPI_стабильность'] = 0
            self.df['KPI_тренд'] = 0
            self.df['KPI_мин'] = 0
            self.df['KPI_макс'] = 0
            self.df['KPI_размах'] = 0
            self.df['KPI_последний'] = 0
            
        return self
    
    def process_dates(self):
        """Обработка дат"""
        current_date = pd.to_datetime(CURRENT_DATE)
        
        # Обработка отпуска
        vacation_column = self._get_column_name([
            'Отпуск (когда ходил в последний раз)',
            'Отпуск'
        ])
        if vacation_column:
            vacation_dates = pd.to_datetime(
                self.df[vacation_column], 
                errors='coerce'
            )
            self.df['Отпуск_месяцев_назад'] = (
                (current_date - vacation_dates).dt.days // 30
            )
            # Исправляем FutureWarning
            self.df['Отпуск_месяцев_назад'] = self.df['Отпуск_месяцев_назад'].fillna(999)
        else:
            self.df['Отпуск_месяцев_назад'] = 999
            
        return self
    
    def encode_categorical(self):
        """Кодирование категориальных переменных"""
        # Бинарное кодирование аттестации
        attestation_column = self._get_column_name([
            'Прохождение аттестации (прошел/не прошел/нет аттестации)',
            'Прохождение аттестации'
        ])
        if attestation_column:
            self.df['Прохождение аттестации'] = self.df[attestation_column].map(BINARY_MAPPING)
        
        # Бинарное кодирование больничного
        sick_column = self._get_column_name([
            'Больничный (брал или нет в 2025 году)',
            'Больничный'
        ])
        if sick_column:
            self.df['Больничный'] = self.df[sick_column].map(BINARY_MAPPING)
        
        # Бинарное кодирование выговора
        reprimand_column = self._get_column_name([
            'Выговор (да/нет)',
            'Выговор'
        ])
        if reprimand_column:
            self.df['Выговор'] = self.df[reprimand_column].map(BINARY_MAPPING)
        
        # Бинарное кодирование участия в активностях
        activities_column = self._get_column_name([
            'Участие в активностях корпоративных',
            'Участие в активностях'
        ])
        if activities_column:
            self.df['Участие в активностях'] = self.df[activities_column].map(BINARY_MAPPING)
        
        # Кодирование обучения
        if 'Обучение' in self.df.columns:
            self.df['Обучение'] = self.df['Обучение'].map({
                'завершена': 1, 
                'в процессе': 0,
                'завершено': 1  # на случай опечаток
            }).fillna(0)
        
        # One-Hot Encoding для города и должности
        if 'Город' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Город'], prefix=['Город'])
        
        if 'Должность' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Должность'], prefix=['Должность'])
        
        # Признак руководства
        sub_column = self._get_column_name([
            'В подчиненнии сотрудники', 
            'В подчинении сотрудники'
        ])
        if sub_column:
            self.df['Руководитель'] = self.df[sub_column].map({
                'Руководитель': 1, 
                'Сотрудник': 0
            }).fillna(0)
        else:
            self.df['Руководитель'] = 0
            
        return self
    
    def process_target(self):
        """Обработка целевой переменной"""
        target_column = self._get_column_name([
            'Состояние выгорания (самооценка своего состояния сотрудника)',
            'Состояние выгорания'
        ])
        if target_column:
            self.df[TARGET_COLUMN] = self.df[target_column].map(BURNOUT_MAPPING)
        else:
            self.df[TARGET_COLUMN] = 0
            
        return self
    
    def finalize_dataset(self):
        """Финализация датасета"""
        # Удаление исходных столбцов, но сохраняем переименованные KPI
        columns_to_drop = [
            'ФИО', 'Стаж'
        ]
        
        # Добавляем столбцы, которые могли быть использованы
        optional_columns = [
            'В подчиненнии сотрудники', 'В подчинении сотрудники',
            'Больничный (брал или нет в 2025 году)', 'Больничный',
            'Выговор (да/нет)', 'Выговор',
            'Участие в активностях корпоративных', 'Участие в активностях',
            'Отпуск (когда ходил в последний раз)', 'Отпуск',
            'Состояние выгорания (самооценка своего состояния сотрудника)',
            'Состояние выгорания',
            'Прохождение аттестации (прошел/не прошел/нет аттестации)',
            'Прохождение аттестации'
        ]
        
        # Удаляем исходные KPI столбцы (они уже переименованы)
        original_kpi_to_drop = [col for col in KPI_COLUMNS if col in self.df.columns]
        columns_to_drop.extend(original_kpi_to_drop)
        
        for col in optional_columns:
            if col in self.df.columns:
                columns_to_drop.append(col)
        
        self.df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        # Заполнение пропусков только в признаках
        feature_columns = [col for col in self.df.columns if col != TARGET_COLUMN]
        self.df[feature_columns] = self.df[feature_columns].fillna(0)
        
        print(f"Финальный датасет: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        print(f"KPI признаки: {[col for col in self.df.columns if 'KPI' in col]}")
        print(f"Столбцы: {list(self.df.columns)}")
        
        return self
    
    def process_all(self):
        """Полный пайплайн обработки"""
        return (self.load_raw_data()
                  .clean_data()
                  .process_gender()
                  .process_experience()
                  .process_kpi()
                  .process_dates()
                  .encode_categorical()
                  .process_target()
                  .finalize_dataset())