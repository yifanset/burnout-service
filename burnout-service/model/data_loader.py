# data_loader.py
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    @staticmethod
    def load_splits():
        """Загрузка готовых train/val/test наборов"""
        base_path = 'data/splits/'
        
        try:
            X_train = pd.read_csv(f'{base_path}X_train.csv')
            X_val = pd.read_csv(f'{base_path}X_val.csv')
            X_test = pd.read_csv(f'{base_path}X_test.csv')
            
            y_train = pd.read_csv(f'{base_path}y_train.csv').squeeze()
            y_val = pd.read_csv(f'{base_path}y_val.csv').squeeze()
            y_test = pd.read_csv(f'{base_path}y_test.csv').squeeze()
            
            # Очищаем данные от строковых признаков
            X_train = DataLoader._clean_dataframe(X_train)
            X_val = DataLoader._clean_dataframe(X_val)
            X_test = DataLoader._clean_dataframe(X_test)
            
            print(f"✅ Наборы данных загружены:")
            print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except FileNotFoundError as e:
            print(f"❌ Файлы наборов данных не найдены: {e}")
            print("Сначала выполните предобработку данных.")
            return None
    
    @staticmethod
    def _clean_dataframe(df):
        """Очистка DataFrame от строковых признаков"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    # Пробуем преобразовать в числа
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean[col] = df_clean[col].fillna(0)
                except:
                    # Если не получается, используем Label Encoding
                    le = LabelEncoder()
                    df_clean[col] = df_clean[col].fillna('unknown')
                    df_clean[col] = le.fit_transform(df_clean[col])
        
        return df_clean
    
    @staticmethod
    def list_available_splits():
        """Показать доступные наборы данных"""
        base_path = 'data/splits/'
        if os.path.exists(base_path):
            files = os.listdir(base_path)
            print("📁 Доступные наборы данных:")
            for file in sorted(files):
                print(f"   {file}")
        else:
            print("📁 Папка data/splits/ не существует")

# Пример использования
if __name__ == "__main__":
    DataLoader.list_available_splits()
    splits = DataLoader.load_splits()