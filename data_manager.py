# data_manager.py
import pandas as pd
import json
from datetime import datetime
from config import *

class DataManager:
    @staticmethod
    def save_processed_data(df):
        """Сохранение обработанных данных в разных форматах"""
        
        # Проверяем наличие целевой переменной
        has_target = TARGET_COLUMN in df.columns
        
        # Сохраняем CSV для ML
        df.to_csv(PROCESSED_CSV, index=False, encoding='utf-8')
        
        # Сохраняем отдельно признаки (если есть целевая переменная)
        if has_target:
            features_df = df.drop(columns=[TARGET_COLUMN])
            features_df.to_csv(FEATURES_CSV, index=False, encoding='utf-8')
        else:
            # Если целевой переменной нет, то все столбцы - это признаки
            features_df = df.copy()
            features_df.to_csv(FEATURES_CSV, index=False, encoding='utf-8')
            print(f"Внимание: целевая переменная '{TARGET_COLUMN}' отсутствует в данных")
        
        # Сохраняем JSON для удобства работы
        records = df.to_dict('records')
        
        data_structure = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_records': len(records),
                'columns': list(df.columns),
                'has_target': has_target,
                'target_column': TARGET_COLUMN if has_target else None,
                'feature_columns': list(features_df.columns)
            },
            'records': records
        }
        
        with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
            json.dump(data_structure, f, ensure_ascii=False, indent=2)
        
        print(f"Данные сохранены:")
        print(f"  - JSON: {PROCESSED_JSON} ({len(records)} записей)")
        print(f"  - CSV: {PROCESSED_CSV}")
        print(f"  - Features: {FEATURES_CSV}")
        if not has_target:
            print(f"  ⚠️  Целевая переменная '{TARGET_COLUMN}' отсутствует!")
        
        return data_structure
    
    @staticmethod
    def add_new_records(new_records):
        """Добавление новых записей в JSON и обновление CSV"""
        try:
            with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Добавляем новые записи
            data['records'].extend(new_records)
            
            # Обновляем метаданные
            data['metadata']['total_records'] = len(data['records'])
            data['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Сохраняем обновленный JSON
            with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Обновляем CSV файлы
            updated_df = pd.DataFrame(data['records'])
            updated_df.to_csv(PROCESSED_CSV, index=False)
            
            # Сохраняем признаки (с проверкой наличия целевой переменной)
            if data['metadata']['has_target']:
                features_df = updated_df.drop(columns=[TARGET_COLUMN])
            else:
                features_df = updated_df.copy()
            features_df.to_csv(FEATURES_CSV, index=False)
            
            print(f"Добавлено {len(new_records)} новых записей")
            print(f"Всего записей: {len(data['records'])}")
            
            return data
            
        except FileNotFoundError:
            print("Файл с обработанными данными не найден. Сначала выполните обработку данных.")
            return None
    
    @staticmethod
    def load_data_for_ml():
        """Загрузка данных для ML (из CSV)"""
        try:
            df = pd.read_csv(PROCESSED_CSV)
            print(f"Загружено {len(df)} записей для ML")
            
            # Проверяем наличие целевой переменной
            if TARGET_COLUMN in df.columns:
                print(f"Целевая переменная '{TARGET_COLUMN}' присутствует")
            else:
                print(f"Целевая переменная '{TARGET_COLUMN}' отсутствует")
                
            return df
        except FileNotFoundError:
            print("Файл с обработанными данными не найден. Сначала выполните обработку данных.")
            return None
    
    @staticmethod
    def load_features_for_ml():
        """Загрузка только признаков для ML"""
        try:
            df = pd.read_csv(FEATURES_CSV)
            print(f"Загружено {len(df)} записей признаков для ML")
            return df
        except FileNotFoundError:
            print("Файл с признаками не найден. Сначала выполните обработку данных.")
            return None
    
    @staticmethod
    def get_dataset_info():
        """Получение информации о датасете"""
        try:
            with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['records'])
            
            info = {
                'total_records': len(df),
                'columns': list(df.columns),
                'has_target': data['metadata']['has_target'],
                'metadata': data['metadata']
            }
            
            # Добавляем информацию о целевой переменной, если она есть
            if info['has_target']:
                info['target_missing'] = df[TARGET_COLUMN].isna().sum()
                info['target_distribution'] = df[TARGET_COLUMN].value_counts().to_dict()
            else:
                info['target_missing'] = 'N/A'
                info['target_distribution'] = 'N/A'
                info['warning'] = f"Целевая переменная '{TARGET_COLUMN}' отсутствует"
            
            return info
        except FileNotFoundError:
            print("Файл с обработанными данными не найден.")
            return None
    
    @staticmethod
    def check_target_presence():
        """Проверка наличия целевой переменной в данных"""
        try:
            df = pd.read_csv(PROCESSED_CSV)
            has_target = TARGET_COLUMN in df.columns
            
            if has_target:
                target_stats = df[TARGET_COLUMN].describe()
                print(f"Целевая переменная '{TARGET_COLUMN}' присутствует:")
                print(f"  - Распределение: {df[TARGET_COLUMN].value_counts().to_dict()}")
                print(f"  - Пропуски: {df[TARGET_COLUMN].isna().sum()}")
            else:
                print(f"Целевая переменная '{TARGET_COLUMN}' отсутствует в данных")
                print(f"Доступные столбцы: {list(df.columns)}")
            
            return has_target
        except FileNotFoundError:
            print("Файл с обработанными данными не найден.")
            return False